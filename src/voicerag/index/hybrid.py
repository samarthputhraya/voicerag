"""Hybrid retrieval: dense + sparse, fused.

Two fusion families are implemented because the ablation compares them, and the
comparison is cheap enough that shipping only one would be a wasted result:

**Reciprocal Rank Fusion (RRF).** ``score = sum_runs w_r / (k + rank_r)``.
It reads only ranks, so it is immune to the fact that cosine similarity lives in
``[-1, 1]`` while BM25 scores are unbounded and corpus-dependent. That
scale-invariance is why RRF is the default: it needs no tuning and cannot be
destabilised by one retriever's score distribution drifting after a re-index.
The constant ``k`` (60, from Cormack et al. 2009) damps the top of the curve --
with ``k=60`` the gap between rank 1 and rank 2 is ~1.6 %, so a single confident
retriever cannot unilaterally decide the answer.

**Weighted score fusion.** Normalise each run's scores (min-max or z-score),
then take a weighted sum. This keeps *margin* information RRF throws away: a
dense hit at 0.91 next to a runner-up at 0.42 is a far stronger signal than two
hits at 0.66, and RRF sees both as "rank 1 and rank 2". The cost is fragility --
min-max over a top-k slice is a normalisation over a truncated, self-selected
sample, so the same document can normalise differently depending on what else
was retrieved.

Whichever fusion runs, every :class:`RetrievalHit` carries the raw per-run
scores and ranks. The guardrail layer decides whether to answer or abstain from
exactly those numbers -- an absolute dense score below threshold, or a document
that only one retriever found, are both reasons to hedge -- so discarding them
at fusion time would be throwing away the abstention signal.

Dense and sparse retrieval are independent, so they run concurrently on a small
thread pool. This is not a token gesture: both inner loops are C/numpy code that
releases the GIL (faiss graph traversal, bm25s sparse slicing), so the wall
clock is genuinely ``max(dense, sparse)`` rather than the sum.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from .dense import DenseIndex
from .sparse import SparseIndex

__all__ = [
    "RetrievalHit",
    "HybridIndex",
    "FusionMethod",
    "rrf_fuse",
    "score_fuse",
    "DEFAULT_RRF_K",
]

#: Cormack et al. (2009). Large enough that no single run dominates rank 1.
DEFAULT_RRF_K = 60

FusionMethod = Literal["rrf", "minmax", "zscore", "dense", "sparse"]

_Run = list[tuple[str, float]]


@dataclass(slots=True)
class RetrievalHit:
    """One fused retrieval result, with its provenance intact.

    Attributes:
        chunk_id: Id of the retrieved chunk; resolve payload via ``ChunkStore``.
        score: Fused score. Comparable within one query only -- RRF scores are
            not calibrated across queries and min-max scores are relative to the
            candidate slice.
        rank: 1-based position in the fused list.
        dense_score: Raw cosine similarity, or ``None`` if the dense run did not
            return this chunk.
        sparse_score: Raw BM25 score, or ``None`` if the sparse run did not.
        dense_rank: 1-based rank within the dense run, or ``None``.
        sparse_rank: 1-based rank within the sparse run, or ``None``.
        source: Which runs found it -- ``"dense"``, ``"sparse"`` or ``"both"``.
            ``"both"`` is a meaningful agreement signal for the guardrails.
    """

    chunk_id: str
    score: float
    rank: int
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    source: str = "both"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "dense_score": None if self.dense_score is None else round(self.dense_score, 6),
            "sparse_score": None if self.sparse_score is None else round(self.sparse_score, 6),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "source": self.source,
        }


# --- fusion primitives --------------------------------------------------------


def rrf_fuse(
    runs: Mapping[str, _Run],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked runs by Reciprocal Rank Fusion.

    Args:
        runs: Mapping of run name -> ``(doc_id, score)`` list, already sorted
            best-first. Scores are ignored; only order matters.
        k: RRF damping constant. Smaller values sharpen the top of the list.
        weights: Optional per-run multipliers. Missing runs default to 1.0.

    Returns:
        ``(doc_id, fused_score)`` sorted best-first, ties broken by doc id so
        the output is deterministic across runs and platforms.

    Raises:
        ValueError: If ``k`` is negative (it would make early ranks explode or
            divide by zero).
    """
    if k < 0:
        raise ValueError(f"rrf k must be non-negative, got {k}")
    acc: dict[str, float] = {}
    for name, run in runs.items():
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        if w == 0.0:
            continue
        for rank, (doc_id, _score) in enumerate(run, start=1):
            acc[doc_id] = acc.get(doc_id, 0.0) + w / (k + rank)
    return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))


def _normalise(scores: Sequence[float], method: str) -> tuple[list[float], float]:
    """Normalise one run's scores; also return the value to impute for misses."""
    if not scores:
        return [], 0.0
    arr = np.asarray(scores, dtype=np.float64)
    if method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-12:
            # Degenerate run: every candidate scored the same, so they are all
            # equally good. Imputing 0.0 for misses still ranks them below.
            return [1.0] * len(scores), 0.0
        return ((arr - lo) / (hi - lo)).tolist(), 0.0
    if method == "zscore":
        mu, sd = float(arr.mean()), float(arr.std())
        if sd < 1e-12:
            return [0.0] * len(scores), 0.0
        z = ((arr - mu) / sd).tolist()
        # Misses get the run's minimum z-score rather than 0.0: under z-scoring
        # 0.0 means "average", which would reward a document for being absent.
        return z, float(min(z))
    raise ValueError(f"unknown normalisation {method!r}; use 'minmax' or 'zscore'")


def score_fuse(
    runs: Mapping[str, _Run],
    *,
    method: Literal["minmax", "zscore"] = "minmax",
    weights: Mapping[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked runs by weighted sum of per-run normalised scores.

    Unlike :func:`rrf_fuse` this preserves the *margin* between candidates
    within a run, which is the whole reason to try it -- see the module
    docstring for when that helps and when it misleads.

    Args:
        runs: Mapping of run name -> ``(doc_id, score)`` list, best-first.
        method: ``"minmax"`` maps each run to ``[0, 1]``; ``"zscore"`` centres
            and scales it.
        weights: Optional per-run multipliers. Missing runs default to 1.0.

    Returns:
        ``(doc_id, fused_score)`` sorted best-first, ties broken by doc id.
    """
    acc: dict[str, float] = {}
    seen: list[str] = []
    per_run: dict[str, tuple[dict[str, float], float, float]] = {}
    for name, run in runs.items():
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        if w == 0.0 or not run:
            continue
        norm, miss = _normalise([s for _, s in run], method)
        per_run[name] = ({d: v for (d, _), v in zip(run, norm)}, miss, w)
        for doc_id, _ in run:
            if doc_id not in acc:
                acc[doc_id] = 0.0
                seen.append(doc_id)

    for doc_id in seen:
        total = 0.0
        for values, miss, w in per_run.values():
            total += w * values.get(doc_id, miss)
        acc[doc_id] = total
    return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))


# --- the retriever ------------------------------------------------------------


class HybridIndex:
    """Dense + sparse retrieval with pluggable fusion.

    Args:
        dense: A built :class:`DenseIndex`, or ``None`` for sparse-only.
        sparse: A built :class:`SparseIndex`, or ``None`` for dense-only.
        fusion: Default fusion method.
        rrf_k: Default RRF damping constant.
        dense_weight / sparse_weight: Default per-run weights. Applied by both
            fusion families, so an ablation can sweep the mix once and compare
            RRF against score fusion at the same operating point.
        candidate_multiplier: Each run retrieves ``k * multiplier`` candidates
            before fusion. Fusion can only promote a document one run ranked
            poorly if that document was retrieved at all, so a fusion depth
            equal to ``k`` would silently degrade to "dense wins ties". Three is
            the usual choice and costs nothing measurable at these latencies.
    """

    def __init__(
        self,
        dense: DenseIndex | None = None,
        sparse: SparseIndex | None = None,
        *,
        fusion: FusionMethod = "rrf",
        rrf_k: int = DEFAULT_RRF_K,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        candidate_multiplier: int = 3,
    ) -> None:
        if dense is None and sparse is None:
            raise ValueError("HybridIndex needs at least one of dense/sparse")
        self.dense = dense
        self.sparse = sparse
        self.fusion: FusionMethod = fusion
        self.rrf_k = int(rrf_k)
        self.weights = {"dense": float(dense_weight), "sparse": float(sparse_weight)}
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self._pool: ThreadPoolExecutor | None = None

    # -- lifecycle ------------------------------------------------------------

    def __len__(self) -> int:
        if self.dense is not None:
            return len(self.dense)
        assert self.sparse is not None
        return len(self.sparse)

    def __repr__(self) -> str:
        return (
            f"HybridIndex(n={len(self)}, fusion={self.fusion!r}, "
            f"weights={self.weights}, rrf_k={self.rrf_k})"
        )

    def __enter__(self) -> "HybridIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _executor(self) -> ThreadPoolExecutor:
        """Lazily create the two-thread pool used for concurrent retrieval.

        Created on first concurrent use rather than in ``__init__`` so that
        purely sequential users (tests, batch evaluation) never spawn threads.
        Exactly two workers: there are exactly two independent runs, and extra
        threads would only add contention on a 2 vCPU box.
        """
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="voicerag-retrieve"
            )
        return self._pool

    def close(self) -> None:
        """Shut the retrieval thread pool down. Safe to call repeatedly."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def config(self) -> dict[str, Any]:
        """Configuration snapshot for the ablation report."""
        return {
            "fusion": self.fusion,
            "rrf_k": self.rrf_k,
            "weights": dict(self.weights),
            "candidate_multiplier": self.candidate_multiplier,
            "dense": None if self.dense is None else self.dense.config(),
            "sparse": None if self.sparse is None else self.sparse.config(),
        }

    # -- individual runs ------------------------------------------------------

    def _dense_run(
        self, query_vec: np.ndarray | None, n: int, ef_search: int | None
    ) -> _Run:
        if self.dense is None or query_vec is None:
            return []
        return self.dense.search(query_vec, n, ef_search=ef_search)

    def _sparse_run(self, query_text: str | None, n: int) -> _Run:
        if self.sparse is None or not query_text:
            return []
        return self.sparse.search(query_text, n)

    # -- fusion ---------------------------------------------------------------

    def _assemble(
        self,
        dense_run: _Run,
        sparse_run: _Run,
        k: int,
        fusion: FusionMethod,
        rrf_k: int,
        weights: Mapping[str, float],
    ) -> list[RetrievalHit]:
        """Fuse two runs and attach per-run provenance to each surviving hit."""
        d_score = {d: s for d, s in dense_run}
        s_score = {d: s for d, s in sparse_run}
        d_rank = {d: i for i, (d, _) in enumerate(dense_run, start=1)}
        s_rank = {d: i for i, (d, _) in enumerate(sparse_run, start=1)}

        runs = {"dense": dense_run, "sparse": sparse_run}
        if fusion == "rrf":
            fused = rrf_fuse(runs, k=rrf_k, weights=weights)
        elif fusion in ("minmax", "zscore"):
            fused = score_fuse(runs, method=fusion, weights=weights)
        elif fusion == "dense":
            fused = list(dense_run)
        elif fusion == "sparse":
            fused = list(sparse_run)
        else:  # pragma: no cover - guarded by the Literal type
            raise ValueError(f"unknown fusion {fusion!r}")

        hits: list[RetrievalHit] = []
        for rank, (doc_id, score) in enumerate(fused[:k], start=1):
            in_d, in_s = doc_id in d_score, doc_id in s_score
            hits.append(
                RetrievalHit(
                    chunk_id=doc_id,
                    score=float(score),
                    rank=rank,
                    dense_score=d_score.get(doc_id),
                    sparse_score=s_score.get(doc_id),
                    dense_rank=d_rank.get(doc_id),
                    sparse_rank=s_rank.get(doc_id),
                    source="both" if in_d and in_s else ("dense" if in_d else "sparse"),
                )
            )
        return hits

    def _candidate_n(self, k: int, candidate_k: int | None) -> int:
        return max(1, int(candidate_k or k * self.candidate_multiplier))

    # -- public search --------------------------------------------------------

    def search(
        self,
        query_text: str | None = None,
        query_vec: np.ndarray | None = None,
        k: int = 10,
        *,
        fusion: FusionMethod | None = None,
        rrf_k: int | None = None,
        weights: Mapping[str, float] | None = None,
        candidate_k: int | None = None,
        ef_search: int | None = None,
        concurrent: bool = False,
    ) -> list[RetrievalHit]:
        """Retrieve and fuse.

        Args:
            query_text: Raw query, used by the sparse run. ``None`` skips it.
            query_vec: Query embedding, used by the dense run. ``None`` skips
                it, which is how the speculative path serves a cache hit.
            k: Number of fused results to return.
            fusion: Override the default fusion method.
            rrf_k: Override the RRF constant.
            weights: Override per-run weights.
            candidate_k: Candidates retrieved per run before fusion.
            ef_search: Per-call HNSW breadth override.
            concurrent: Run the two retrievers on the thread pool. Off by
                default because for a small in-process corpus the pool dispatch
                (~0.1 ms) can exceed the work; on a real corpus it wins. The
                async entry point :meth:`asearch` always runs concurrently.

        Returns:
            Up to ``k`` :class:`RetrievalHit` objects, best first. Empty list if
            both runs came back empty.
        """
        if k <= 0:
            return []
        n = self._candidate_n(k, candidate_k)
        if concurrent and self.dense is not None and self.sparse is not None:
            pool = self._executor()
            fut = pool.submit(self._sparse_run, query_text, n)
            dense_run = self._dense_run(query_vec, n, ef_search)
            sparse_run = fut.result()
        else:
            dense_run = self._dense_run(query_vec, n, ef_search)
            sparse_run = self._sparse_run(query_text, n)
        return self._assemble(
            dense_run,
            sparse_run,
            k,
            fusion or self.fusion,
            self.rrf_k if rrf_k is None else int(rrf_k),
            self.weights if weights is None else {**self.weights, **weights},
        )

    async def asearch(
        self,
        query_text: str | None = None,
        query_vec: np.ndarray | None = None,
        k: int = 10,
        *,
        fusion: FusionMethod | None = None,
        rrf_k: int | None = None,
        weights: Mapping[str, float] | None = None,
        candidate_k: int | None = None,
        ef_search: int | None = None,
    ) -> list[RetrievalHit]:
        """Async :meth:`search` that always overlaps the two retrievers.

        Both runs are dispatched to the retrieval pool and awaited together, so
        the event loop is never blocked by faiss or bm25s and wall-clock cost is
        ``max(dense, sparse)`` plus fusion.

        Args: See :meth:`search`.

        Returns:
            Up to ``k`` :class:`RetrievalHit` objects, best first.
        """
        if k <= 0:
            return []
        n = self._candidate_n(k, candidate_k)
        loop = asyncio.get_running_loop()
        pool = self._executor()
        dense_task = loop.run_in_executor(pool, self._dense_run, query_vec, n, ef_search)
        sparse_task = loop.run_in_executor(pool, self._sparse_run, query_text, n)
        dense_run, sparse_run = await asyncio.gather(dense_task, sparse_task)
        return self._assemble(
            dense_run,
            sparse_run,
            k,
            fusion or self.fusion,
            self.rrf_k if rrf_k is None else int(rrf_k),
            self.weights if weights is None else {**self.weights, **weights},
        )

    # -- persistence ----------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Persist both sub-indexes under one directory.

        Args:
            directory: Created if absent. Sub-indexes go in ``dense/`` and
                ``sparse/``; fusion settings go in ``hybrid.json``.

        Returns:
            The directory written to.
        """
        import json

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self.dense is not None:
            self.dense.save(path / "dense")
        if self.sparse is not None:
            self.sparse.save(path / "sparse")
        (path / "hybrid.json").write_text(
            json.dumps(
                {
                    "fusion": self.fusion,
                    "rrf_k": self.rrf_k,
                    "weights": self.weights,
                    "candidate_multiplier": self.candidate_multiplier,
                    "has_dense": self.dense is not None,
                    "has_sparse": self.sparse is not None,
                }
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: str | Path, *, mmap_sparse: bool = True) -> "HybridIndex":
        """Restore an index written by :meth:`save`.

        Args:
            directory: Directory produced by :meth:`save`.
            mmap_sparse: Memory-map the BM25 arrays. See
                :meth:`SparseIndex.load`.

        Returns:
            A ready-to-search :class:`HybridIndex`.
        """
        import json

        path = Path(directory)
        cfg = json.loads((path / "hybrid.json").read_text(encoding="utf-8"))
        dense = DenseIndex.load(path / "dense") if cfg.get("has_dense") else None
        sparse = (
            SparseIndex.load(path / "sparse", mmap=mmap_sparse)
            if cfg.get("has_sparse")
            else None
        )
        weights = cfg.get("weights", {})
        return cls(
            dense,
            sparse,
            fusion=cfg.get("fusion", "rrf"),
            rrf_k=int(cfg.get("rrf_k", DEFAULT_RRF_K)),
            dense_weight=float(weights.get("dense", 1.0)),
            sparse_weight=float(weights.get("sparse", 1.0)),
            candidate_multiplier=int(cfg.get("candidate_multiplier", 3)),
        )
