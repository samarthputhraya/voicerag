#!/usr/bin/env python
"""Build and persist a retrieval index from one MSMARCO-XI shard.

    python scripts/ingest.py --limit 20000 --strategy sentence_window --out data/index

Streams the pinned 440 MB validation shard rather than downloading the 55.6 GB
repository, deduplicates passages by content hash, chunks with the chosen
strategy, fits (or loads) the embedder, builds the dense HNSW and BM25 indexes
and writes everything plus a manifest describing exactly how it was made.

Offline machines can build the identical layout from the synthetic fixture with
``--synthetic 2000``, which is what makes ``scripts/bench_latency.py`` runnable
with no network.

The output directory is self-describing::

    <out>/dense/      faiss HNSW graph + ids
    <out>/sparse/     BM25 matrix + vocabulary
    <out>/store/      chunk payloads (text, offsets, provenance)
    <out>/hybrid.json fusion settings
    <out>/embedder.pkl fitted LSA projection (LSA backends only)
    <out>/manifest.json  everything needed to reproduce or serve it
    <out>/rows.jsonl.gz  the consumed rows, when --cache-rows is set
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from _bootstrap import bootstrap

ROOT = bootstrap()

from eval.dataset import (  # noqa: E402 - after sys.path bootstrap
    DEFAULT_REPO_ID,
    DEFAULT_SHARD,
    DatasetUnavailable,
    build_corpus,
    dump_jsonl,
    load_jsonl,
    load_msmarco_xi,
    synthetic_rows,
)


def _progress(label: str, every: int = 2000):
    """Return a callback printing a heartbeat every ``every`` items."""

    def report(i: int) -> None:
        if i and i % every == 0:
            print(f"  {label}: {i:,}", flush=True)

    return report


def _rows_from_parquet(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    """Stream rows straight out of a local parquet shard.

    The ``datasets`` streaming path resolves data files against the hub before
    the first row appears, and on a 55 GB repository that step has been observed
    to hang for tens of minutes with no output. Reading a downloaded shard with
    pyarrow removes the hub, the resolution step and the ``datasets`` dependency
    from the critical path -- which matters most inside a container build, where
    a hang is indistinguishable from progress until the builder times out.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    seen = 0
    for batch in pf.iter_batches(batch_size=2000):
        for row in batch.to_pylist():
            yield row
            seen += 1
            if limit is not None and seen >= limit:
                return


def _download_shard(repo_id: str, shard: str, dest: Path) -> Path:
    """Fetch one parquet shard from the hub, returning its local path."""
    from huggingface_hub import hf_hub_download

    print(f"downloading {repo_id}:{shard}", flush=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=shard,
            repo_type="dataset",
            local_dir=str(dest),
        )
    )


def _rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    """Resolve the row source in priority order: cache, parquet, synthetic, hub."""
    if args.rows:
        print(f"reading cached rows from {args.rows}", flush=True)
        yield from load_jsonl(args.rows, limit=args.limit)
        return
    if args.parquet:
        print(f"reading {args.parquet} (limit={args.limit})", flush=True)
        yield from _rows_from_parquet(args.parquet, args.limit)
        return
    if args.synthetic:
        print(f"generating {args.synthetic} synthetic rows (offline fixture)", flush=True)
        yield from synthetic_rows(args.synthetic, seed=args.seed)
        return
    if args.download:
        local = _download_shard(args.repo_id, args.shard, args.out.parent / "raw")
        yield from _rows_from_parquet(local, args.limit)
        return
    print(f"streaming {args.repo_id}:{args.shard} (limit={args.limit})", flush=True)
    yield from load_msmarco_xi(args.shard, limit=args.limit, repo_id=args.repo_id)


def _counted(rows: Iterable[dict[str, Any]], report) -> Iterator[dict[str, Any]]:
    for i, row in enumerate(rows, start=1):
        report(i)
        yield row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_argument_group("source")
    src.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                     help="hub repo, or a local directory holding the same "
                          "<split>/<name>.parquet layout. Streaming the shard "
                          "straight off the hub is one long-lived HTTP read and "
                          "dies on a slow link; download it once and point here.")
    src.add_argument("--shard", default=DEFAULT_SHARD, help="parquet path inside the repo")
    src.add_argument("--limit", type=int, default=20_000, help="rows to consume")
    src.add_argument("--synthetic", type=int, default=0, metavar="N",
                     help="build from N synthetic rows instead of the hub (offline)")
    src.add_argument("--rows", type=Path, help="read rows from a cached .jsonl(.gz) instead")
    src.add_argument("--parquet", type=Path,
                     help="read rows from an already-downloaded shard, skipping the hub entirely")
    src.add_argument("--download", action="store_true",
                     help="download the shard with hf_hub_download, then read it with pyarrow. "
                          "Preferred over the default `datasets` streaming path in a container "
                          "build: streaming resolves data files against a 55GB repo first, which "
                          "has been observed to hang with no output.")
    src.add_argument("--cache-rows", action="store_true",
                     help="also write the consumed rows to <out>/rows.jsonl.gz")

    build = parser.add_argument_group("build")
    build.add_argument("--strategy", default="sentence_window", help="chunking strategy")
    build.add_argument("--embedder", default="", help="embedder spec; default picks an LSA dim for the corpus")
    build.add_argument("--ef-construction", type=int, default=80)
    build.add_argument("--ef-search", type=int, default=64)
    build.add_argument("--hnsw-m", type=int, default=32)
    build.add_argument("--seed", type=int, default=0)

    out = parser.add_argument_group("output")
    out.add_argument("--out", type=Path, default=ROOT / "data" / "index")
    out.add_argument("--resume", action="store_true",
                     help="exit successfully if <out>/manifest.json already describes this build")
    args = parser.parse_args(argv)

    from eval.ablation import AblationConfig, build_indexes, default_embedder_spec, fit_embedder
    from voicerag.chunking.registry import build as build_strategy

    manifest_path = args.out / "manifest.json"
    wanted = {
        "strategy": args.strategy,
        "limit": args.limit,
        "shard": args.shard if not args.synthetic else f"synthetic:{args.synthetic}",
    }
    if args.resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if all(existing.get(k) == v for k, v in wanted.items()):
            print(f"resume: {manifest_path} already matches this configuration; nothing to do")
            return 0
        print("resume: existing manifest differs; rebuilding", flush=True)

    t_start = time.perf_counter()
    try:
        rows = list(_counted(_rows(args), _progress("rows")))
    except DatasetUnavailable as exc:
        print(f"\ndataset unavailable: {exc}", file=sys.stderr)
        print("hint: run with --synthetic 2000 to build an offline index", file=sys.stderr)
        return 2
    if not rows:
        print("no rows consumed; nothing to build", file=sys.stderr)
        return 2

    corpus = build_corpus(rows)
    print(f"corpus: {json.dumps(corpus.stats())}", flush=True)

    spec = args.embedder or default_embedder_spec(len(corpus.documents))
    print(f"fitting embedder {spec} on {len(corpus.documents):,} passages", flush=True)
    embedder = fit_embedder(spec, [d.text for d in corpus.documents])

    config = AblationConfig(
        ef_search=args.ef_search,
        ef_construction=args.ef_construction,
        hnsw_m=args.hnsw_m,
    )
    print(f"chunking with {args.strategy!r} and building indexes", flush=True)
    built = build_indexes(
        corpus.documents,
        build_strategy(args.strategy, embedder=embedder),
        embedder,
        config,
        measure_size=False,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    built.hybrid.save(args.out)
    built.store.save(args.out / "store")
    save = getattr(embedder, "save", None)
    embedder_file = ""
    if callable(save):
        # The LSA projection is corpus-specific: an index built with one
        # projection is meaningless under another, so the two must travel
        # together or the served results are silently wrong.
        embedder_file = str(save(args.out / "embedder.pkl").name)
    if args.cache_rows:
        dump_jsonl(rows, args.out / "rows.jsonl.gz")

    manifest = {
        **wanted,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_id": args.repo_id,
        "upstream_repo_id": DEFAULT_REPO_ID,
        "embedder_spec": spec,
        "embedder_file": embedder_file,
        "embedder_dim": int(embedder.dim),
        "n_documents": len(corpus.documents),
        "n_chunks": built.n_chunks,
        "corpus": corpus.stats(),
        "index": config.to_dict(),
        "chunk_ms": built.chunk_ms,
        "embed_ms": built.embed_ms,
        "build_ms": built.build_ms,
        "queries": [q.to_dict() for q in corpus.queries[:2000]],
        "qrels": {str(k): sorted(v) for k, v in list(corpus.qrels.items())[:2000]},
        "unanswerable": [q.to_dict() for q in corpus.unanswerable[:2000]],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    built.close()

    total = time.perf_counter() - t_start
    print(
        f"done in {total:.1f}s: {built.n_chunks:,} chunks from "
        f"{len(corpus.documents):,} passages -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
