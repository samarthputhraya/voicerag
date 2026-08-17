"""Chunking and fusion ablations against real relevance judgements.

An ablation is the difference between "we chose sentence windows" and "we
measured that sentence windows win, here is by how much, and here is what they
cost". This module produces that table.

Two axes:

* **Chunking** (:func:`run_ablation`) -- every strategy in
  :data:`voicerag.chunking.registry.STRATEGIES`, holding retrieval, fusion and
  the embedder fixed. Reports Recall@1/5/10, MRR@10 and nDCG@10 *plus* the
  operational cost that a quality-only table hides: chunk count, mean chunk
  length, chunk/embed/build time, serialised index size and per-query latency.
  A strategy that buys +0.01 nDCG for 3x the index is a bad trade, and the
  table has to make that visible.
* **Fusion** (:func:`run_fusion_ablation`) -- RRF against weighted-score fusion
  (min-max and z-score) and against the dense-only and sparse-only controls,
  holding the chunking fixed. The controls matter: without them, "hybrid
  retrieval" is an assertion.

**One embedder, fitted once.** The LSA projection is fitted on the *document*
corpus and shared by every strategy. Refitting per strategy would change the
representation and the split together, and the table would no longer isolate
chunking. This is also why the embedder is an argument: swap in the neural
static embedder and every row moves, comparably.

**Relevance is judged on passages, not chunks.** Retrieval returns chunk ids;
each is mapped back to its ``doc_id`` (the passage content hash from
:mod:`eval.dataset`) and the ranking is deduplicated in place, so a strategy
that emits ten chunks per passage cannot flood the top-k with one document and
fake a recall win.

Runs are cached per row: with ``cache_dir`` set, an interrupted ablation
resumes and only the missing strategies are recomputed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Mapping, Sequence

from .dataset import Query
from .metrics import QueryResult, aggregate

__all__ = [
    "AblationConfig",
    "BuiltIndex",
    "ablation_to_csv",
    "ablation_to_json",
    "ablation_to_markdown",
    "build_indexes",
    "default_embedder_spec",
    "fit_embedder",
    "run_ablation",
    "run_fusion_ablation",
    "run_queries",
]

#: Columns shown, in order, in the markdown ablation table. Quality first
#: (that is what the choice is made on), cost second (that is what the choice
#: costs), latency last.
_MD_COLUMNS: tuple[tuple[str, str], ...] = (
    ("strategy", "Strategy"),
    ("n_chunks", "Chunks"),
    ("mean_chunk_chars", "Mean chars"),
    ("recall@1", "R@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    ("mrr@10", "MRR@10"),
    ("ndcg@10", "nDCG@10"),
    ("chunk_ms", "Chunk ms"),
    ("embed_ms", "Embed ms"),
    ("build_ms", "Build ms"),
    ("index_mb", "Index MB"),
    ("query_p50_ms", "Query p50"),
    ("query_p95_ms", "Query p95"),
)


def default_embedder_spec(n_texts: int, *, max_dim: int = 256) -> str:
    """Pick an LSA dimensionality that the corpus can actually support.

    ``HashingLSAEmbedder`` cannot estimate a rank-``d`` subspace from fewer than
    ``d`` documents, so a 40-document offline fixture must not ask for 256
    components. The rule is ``dim = clamp(n_texts // 4, 16, max_dim)`` rounded
    down to a power of two, which keeps the fixture honest (a real subspace,
    not a memorised one) and the production run at the benchmarked 256.

    Args:
        n_texts: Size of the corpus the embedder will be fitted on.
        max_dim: Ceiling; 256 matches the dimension the FAISS numbers in the
            README were measured at.

    Returns:
        A spec string accepted by
        :func:`voicerag.embed.base.resolve_embedder`, e.g. ``"lsa:64"``.

    Raises:
        ValueError: If ``n_texts`` is below the 16-document floor, where LSA is
            meaningless and the caller should use a different embedder.
    """
    if n_texts < 16:
        raise ValueError(
            f"need at least 16 documents to fit an LSA projection, got {n_texts}"
        )
    dim = max(16, min(max_dim, n_texts // 4))
    dim = 1 << (int(dim).bit_length() - 1)
    return f"lsa:{dim}"


def fit_embedder(spec: str, corpus: Sequence[str]) -> Any:
    """Resolve an embedder spec and fit it if the backend needs fitting.

    Args:
        spec: Spec string for :func:`voicerag.embed.base.resolve_embedder`.
        corpus: Texts to fit on. Ignored by backends that carry weights.

    Returns:
        A ready-to-use embedder.

    Raises:
        ValueError: If a fittable backend is handed a corpus it cannot support.
    """
    from voicerag.embed.base import resolve_embedder

    embedder = resolve_embedder(spec)
    fit = getattr(embedder, "fit", None)
    if callable(fit) and not getattr(embedder, "is_fitted", True):
        fit(list(corpus))
    return embedder


# --- configuration ------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AblationConfig:
    """Everything held fixed across ablation rows.

    Attributes:
        ks: Cutoffs reported for recall/precision/hit-rate.
        k_retrieve: Depth actually retrieved. Must be at least ``max(ks)``.
        fusion: Fusion method for the chunking axis.
        ef_search: HNSW query breadth. 64 is the benchmarked default: 1.1 ms
            p50 at 200k vectors, and recall indistinguishable from exact.
        hnsw_m / ef_construction: Graph build parameters.
        candidate_multiplier: Candidates per run before fusion.
        query_field: Which query text to retrieve with -- ``"eng_query"`` for
            the monolingual table, ``"indic_query"`` for the cross-lingual one.
    """

    ks: tuple[int, ...] = (1, 5, 10)
    k_retrieve: int = 10
    fusion: str = "rrf"
    ef_search: int = 64
    hnsw_m: int = 32
    ef_construction: int = 80
    candidate_multiplier: int = 3
    query_field: str = "eng_query"

    def __post_init__(self) -> None:
        if self.k_retrieve < max(self.ks):
            raise ValueError(
                f"k_retrieve={self.k_retrieve} is below max(ks)={max(self.ks)}; "
                "metrics would be truncated by retrieval depth rather than by k"
            )
        if self.query_field not in ("eng_query", "indic_query"):
            raise ValueError(f"unknown query_field {self.query_field!r}")

    def fingerprint(self) -> str:
        """Short stable hash of the configuration, used as the cache key."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=6).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ks": list(self.ks),
            "k_retrieve": self.k_retrieve,
            "fusion": self.fusion,
            "ef_search": self.ef_search,
            "hnsw_m": self.hnsw_m,
            "ef_construction": self.ef_construction,
            "candidate_multiplier": self.candidate_multiplier,
            "query_field": self.query_field,
        }


# --- index construction -------------------------------------------------------


def _corpus_fingerprint(documents: Sequence[Any], queries: Sequence[Query]) -> str:
    """Hash the corpus identity so a cached row can never be reused across corpora."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(len(documents)).encode())
    for d in documents:
        h.update(d.doc_id.encode())
    h.update(b"|")
    for q in queries:
        h.update(str(q.query_id).encode())
    return h.hexdigest()


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@dataclass(slots=True)
class BuiltIndex:
    """A built retrieval stack plus the cost of building it.

    Attributes:
        hybrid: The searchable :class:`~voicerag.index.hybrid.HybridIndex`.
        store: Chunk payloads, used to map chunk ids back to passage ids.
        chunk_ms / embed_ms / build_ms: Wall-clock for each ingest stage.
        index_bytes: Serialised size of dense + sparse + store. Measured by
            actually saving to a temporary directory, because an analytic
            estimate of a FAISS HNSW graph is guesswork and this number is used
            to argue about deployment cost.
        n_chunks / chunk_chars: Corpus shape after chunking.
    """

    hybrid: Any
    store: Any
    chunk_ms: float
    embed_ms: float
    build_ms: float
    index_bytes: int
    n_chunks: int
    chunk_chars: list[int] = field(default_factory=list)

    def close(self) -> None:
        """Release the hybrid retriever's thread pool."""
        self.hybrid.close()


def build_indexes(
    documents: Sequence[Any],
    strategy: Any,
    embedder: Any,
    config: AblationConfig,
    *,
    measure_size: bool = True,
) -> BuiltIndex:
    """Chunk, embed and index one corpus under one chunking strategy.

    Args:
        documents: Source passages.
        strategy: An instantiated
            :class:`~voicerag.chunking.base.ChunkingStrategy`.
        embedder: Fitted embedder, shared across strategies.
        config: Index parameters.
        measure_size: Serialise to a temporary directory to measure index size.
            Skippable because it costs a full write of the index.

    Returns:
        A :class:`BuiltIndex`. The caller owns it and should call
        :meth:`BuiltIndex.close`.

    Raises:
        ValueError: If the strategy produced no chunks at all.
    """
    from voicerag.index.dense import DenseIndex
    from voicerag.index.hybrid import HybridIndex
    from voicerag.index.sparse import SparseIndex
    from voicerag.index.store import ChunkStore

    t0 = time.perf_counter()
    chunks = list(strategy.split_many(documents))
    chunk_ms = (time.perf_counter() - t0) * 1e3
    if not chunks:
        raise ValueError(f"strategy {strategy.name!r} produced no chunks")

    store = ChunkStore(chunks)
    ids = store.ids()
    texts = store.texts("embed_text")

    t0 = time.perf_counter()
    vectors = embedder.encode(texts)
    embed_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    dense = DenseIndex(ef_search=config.ef_search).build(
        vectors, ids, m=config.hnsw_m, ef_construction=config.ef_construction
    )
    sparse = SparseIndex().build(texts, ids)
    build_ms = (time.perf_counter() - t0) * 1e3

    hybrid = HybridIndex(
        dense=dense,
        sparse=sparse,
        fusion=config.fusion,  # type: ignore[arg-type]
        candidate_multiplier=config.candidate_multiplier,
    )

    index_bytes = 0
    if measure_size:
        with TemporaryDirectory(prefix="voicerag-ablation-") as tmp:
            root = Path(tmp)
            hybrid.save(root)
            store.save(root / "store")
            index_bytes = _dir_bytes(root)

    return BuiltIndex(
        hybrid=hybrid,
        store=store,
        chunk_ms=round(chunk_ms, 3),
        embed_ms=round(embed_ms, 3),
        build_ms=round(build_ms, 3),
        index_bytes=index_bytes,
        n_chunks=len(chunks),
        chunk_chars=[len(c.text) for c in chunks],
    )


# --- query execution ----------------------------------------------------------


def _query_text(query: Query, field_name: str) -> str:
    return query.eng_query if field_name == "eng_query" else query.indic_query


def run_queries(
    built: BuiltIndex,
    embedder: Any,
    queries: Sequence[Query],
    qrels: Mapping[int, Iterable[str]],
    config: AblationConfig,
    *,
    fusion: str | None = None,
) -> list[QueryResult]:
    """Retrieve for every query and map chunk hits back to passage ids.

    Args:
        built: A built index.
        embedder: The same embedder the index was built with.
        queries: Evaluation queries.
        qrels: ``query_id -> gold passage ids``.
        config: Retrieval parameters.
        fusion: Override the configured fusion method for this pass.

    Returns:
        One :class:`~eval.metrics.QueryResult` per query that has qrels.
        Queries without gold are skipped -- they belong to the abstention
        evaluation, not the retrieval one.

    The per-query latency recorded here covers **query embedding + hybrid
    search + id mapping**, i.e. everything retrieval owes the request path. It
    deliberately excludes generation, which :mod:`eval.latency` measures.
    """
    results: list[QueryResult] = []
    for query in queries:
        gold = qrels.get(query.query_id)
        if not gold:
            continue
        text = _query_text(query, config.query_field)

        t0 = time.perf_counter()
        vec = embedder.encode([text])[0]
        hits = built.hybrid.search(
            query_text=text,
            query_vec=vec,
            k=config.k_retrieve,
            fusion=fusion or config.fusion,  # type: ignore[arg-type]
            ef_search=config.ef_search,
        )
        ranked = [
            chunk.doc_id
            for chunk in built.store.get_many([h.chunk_id for h in hits])
        ]
        latency_ms = (time.perf_counter() - t0) * 1e3

        results.append(
            QueryResult(
                query_id=query.query_id,
                ranked=tuple(dict.fromkeys(ranked)),  # passage-level, order kept
                relevant=frozenset(gold),
                latency_ms=latency_ms,
                query_type=query.query_type,
            )
        )
    return results


# --- the runners --------------------------------------------------------------


def _row_from(
    label: str,
    rationale: str,
    built: BuiltIndex,
    results: Sequence[QueryResult],
    config: AblationConfig,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = aggregate(results, ks=config.ks)
    row: dict[str, Any] = {
        "strategy": label,
        "rationale": rationale,
        "n_chunks": built.n_chunks,
        "mean_chunk_chars": round(statistics.fmean(built.chunk_chars), 1),
        "p50_chunk_chars": round(statistics.median(built.chunk_chars), 1),
        "chunk_ms": built.chunk_ms,
        "embed_ms": built.embed_ms,
        "build_ms": built.build_ms,
        "index_bytes": built.index_bytes,
        "index_mb": round(built.index_bytes / (1 << 20), 2),
        "n_queries": summary.n_queries,
        **summary.metrics,
        "query_p50_ms": round(summary.latency.pcts.get("p50", 0.0), 3),
        "query_p95_ms": round(summary.latency.pcts.get("p95", 0.0), 3),
        "query_mean_ms": summary.latency.mean,
        "by_type": summary.by_type,
        "config": config.to_dict(),
    }
    row.update(extra or {})
    return row


def _cache_path(cache_dir: Path, kind: str, key: str, label: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return cache_dir / f"{kind}.{key}.{safe}.json"


def run_ablation(
    documents: Sequence[Any],
    queries: Sequence[Query],
    qrels: Mapping[int, Iterable[str]],
    *,
    strategies: Sequence[str] | None = None,
    embedder: Any | None = None,
    embedder_spec: str | None = None,
    config: AblationConfig | None = None,
    cache_dir: str | Path | None = None,
    measure_size: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the chunking ablation and return one row per strategy.

    Args:
        documents: Deduplicated passages from :func:`eval.dataset.build_corpus`.
        queries: Answerable evaluation queries.
        qrels: ``query_id -> gold passage ids``.
        strategies: Strategy names from the registry. ``None`` runs all of them,
            including ``semantic``, which needs the embedder and gets it.
        embedder: A pre-fitted embedder. Takes precedence over ``embedder_spec``.
        embedder_spec: Spec to resolve and fit on the document corpus. Defaults
            to :func:`default_embedder_spec` for the corpus size.
        config: Fixed retrieval configuration.
        cache_dir: Directory for per-row JSON. When set, rows already present
            are loaded instead of recomputed, which makes the run resumable
            after an interrupt and makes re-running to regenerate the markdown
            free.
        measure_size: Measure serialised index size (costs one full write).
        progress: Called with a short status string before each strategy.

    Returns:
        Rows sorted by ``nDCG@10`` descending, each a flat JSON-serialisable
        dict. Sorted rather than registry-ordered so the winner is the first
        line of the table.

    Raises:
        ValueError: If ``documents`` or ``queries`` is empty.
    """
    from voicerag.chunking.registry import STRATEGIES, build

    if not documents:
        raise ValueError("ablation needs a non-empty document corpus")
    if not queries:
        raise ValueError("ablation needs at least one query")

    cfg = config or AblationConfig()
    names = list(strategies) if strategies is not None else list(STRATEGIES)
    doc_texts = [d.text for d in documents]
    emb = embedder or fit_embedder(
        embedder_spec or default_embedder_spec(len(doc_texts)), doc_texts
    )

    cache = Path(cache_dir) if cache_dir else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    key = f"{cfg.fingerprint()}.{_corpus_fingerprint(documents, queries)}"

    rows: list[dict[str, Any]] = []
    for name in names:
        if name not in STRATEGIES:
            raise KeyError(f"unknown strategy {name!r}; have {sorted(STRATEGIES)}")
        cached = _cache_path(cache, "chunking", key, name) if cache else None
        if cached is not None and cached.exists():
            if progress:
                progress(f"{name}: cached")
            rows.append(json.loads(cached.read_text()))
            continue

        if progress:
            progress(f"{name}: chunking + indexing {len(documents)} docs")
        strategy = build(name, embedder=emb)
        built = build_indexes(documents, strategy, emb, cfg, measure_size=measure_size)
        try:
            if progress:
                progress(f"{name}: {built.n_chunks} chunks, running {len(queries)} queries")
            results = run_queries(built, emb, queries, qrels, cfg)
            row = _row_from(name, STRATEGIES[name][2], built, results, cfg)
        finally:
            built.close()

        if cached is not None:
            cached.write_text(json.dumps(row, indent=2))
        rows.append(row)

    rows.sort(key=lambda r: r.get(f"ndcg@{max(cfg.ks)}", 0.0), reverse=True)
    return rows


def run_fusion_ablation(
    documents: Sequence[Any],
    queries: Sequence[Query],
    qrels: Mapping[int, Iterable[str]],
    *,
    strategy: str = "sentence_window",
    methods: Sequence[str] = ("rrf", "minmax", "zscore", "dense", "sparse"),
    embedder: Any | None = None,
    embedder_spec: str | None = None,
    config: AblationConfig | None = None,
    cache_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Compare fusion methods on a single, fixed chunking strategy.

    ``dense`` and ``sparse`` are not fusion methods -- they are the single-run
    controls, and they are in the default list on purpose. If RRF does not beat
    both of them, the hybrid retriever is complexity with no payoff and the
    table should say so.

    Args:
        documents: Passages.
        queries: Evaluation queries.
        qrels: Gold labels.
        strategy: Chunking strategy held fixed across the sweep.
        methods: Fusion methods (and controls) to compare.
        embedder: Pre-fitted embedder, or ``None`` to build one.
        embedder_spec: Spec used when ``embedder`` is ``None``.
        config: Fixed retrieval configuration.
        cache_dir: Per-row cache, as in :func:`run_ablation`.
        progress: Status callback.

    Returns:
        Rows sorted by ``nDCG@10`` descending. The index is built **once** and
        re-queried per method, because fusion is a query-time decision -- which
        also means the build-cost columns are identical by construction and are
        reported once in the row's ``build_ms``.
    """
    from voicerag.chunking.registry import build

    cfg = config or AblationConfig()
    doc_texts = [d.text for d in documents]
    emb = embedder or fit_embedder(
        embedder_spec or default_embedder_spec(len(doc_texts)), doc_texts
    )
    cache = Path(cache_dir) if cache_dir else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    key = f"{cfg.fingerprint()}.{strategy}.{_corpus_fingerprint(documents, queries)}"

    pending = [
        m
        for m in methods
        if not (cache and _cache_path(cache, "fusion", key, m).exists())
    ]
    rows: list[dict[str, Any]] = []
    if cache is not None:
        rows.extend(
            json.loads(_cache_path(cache, "fusion", key, m).read_text())
            for m in methods
            if m not in pending
        )

    if pending:
        if progress:
            progress(f"fusion: building {strategy} index once for {len(pending)} methods")
        built = build_indexes(documents, build(strategy, embedder=emb), emb, cfg)
        try:
            for method in pending:
                if progress:
                    progress(f"fusion={method}: {len(queries)} queries")
                results = run_queries(built, emb, queries, qrels, cfg, fusion=method)
                row = _row_from(
                    method,
                    _FUSION_RATIONALE.get(method, ""),
                    built,
                    results,
                    cfg,
                    {"chunking": strategy, "fusion": method},
                )
                if cache is not None:
                    _cache_path(cache, "fusion", key, method).write_text(
                        json.dumps(row, indent=2)
                    )
                rows.append(row)
        finally:
            built.close()

    rows.sort(key=lambda r: r.get(f"ndcg@{max(cfg.ks)}", 0.0), reverse=True)
    return rows


_FUSION_RATIONALE: dict[str, str] = {
    "rrf": "Reciprocal Rank Fusion: rank-only, so a run with an uncalibrated "
    "score scale cannot dominate.",
    "minmax": "Weighted sum of min-max normalised scores: keeps within-run "
    "margins, but one outlier rescales the whole run.",
    "zscore": "Weighted sum of z-scored runs: margin-preserving and "
    "outlier-tolerant, assumes roughly symmetric score distributions.",
    "dense": "Control: HNSW only. Isolates what the sparse run contributes.",
    "sparse": "Control: BM25 only. Isolates what the dense run contributes.",
}


# --- writers ------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def ablation_to_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str = "Chunking ablation",
    notes: Sequence[str] = (),
) -> str:
    """Render rows as a GitHub-flavoured markdown table with its rationale key.

    Args:
        rows: Rows from :func:`run_ablation` or :func:`run_fusion_ablation`.
        title: Heading text.
        notes: Extra lines emitted under the heading -- corpus size, embedder
            spec, anything a reader needs to interpret the numbers.

    Returns:
        Markdown. Empty rows produce a heading and an explicit "no rows"
        line rather than a malformed table.
    """
    out: list[str] = [f"### {title}", ""]
    out.extend(f"{n}  " for n in notes)
    if notes:
        out.append("")
    if not rows:
        out.append("_No rows: the ablation produced no results._")
        return "\n".join(out) + "\n"

    cols = [(k, h) for k, h in _MD_COLUMNS if any(k in r for r in rows)]
    out.append("| " + " | ".join(h for _, h in cols) + " |")
    out.append("|" + "|".join("---:" if i else "---" for i, _ in enumerate(cols)) + "|")
    for row in rows:
        out.append("| " + " | ".join(_fmt(row.get(k, "")) for k, _ in cols) + " |")

    out.extend(["", "**Why each row exists**", ""])
    for row in rows:
        if row.get("rationale"):
            out.append(f"- `{row['strategy']}` -- {row['rationale']}")
    return "\n".join(out) + "\n"


def ablation_to_json(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write rows verbatim as JSON (the machine-readable artefact)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(list(rows), indent=2, sort_keys=False))
    return p


def ablation_to_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the flat (non-nested) columns as CSV for spreadsheet consumers.

    Nested values (``by_type``, ``config``) are dropped rather than stringified:
    a CSV cell containing a JSON blob is not usable by the tool the CSV exists
    for.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return p
    keys = [
        k
        for k in rows[0]
        if not isinstance(rows[0][k], (dict, list))
    ]
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return p
