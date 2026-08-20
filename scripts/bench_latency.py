#!/usr/bin/env python
"""Measure the pipeline against the 200 ms bar and write the latency report.

    python scripts/bench_latency.py --index data/index_20k --iterations 200
    python scripts/bench_latency.py --synthetic 300 --iterations 100   # offline

Measures transcript in -> **final** answer token out, which is the quantity the
brief scopes to 200 ms, and reports P50/P70/P90/P95/P99/P100 by nearest rank.
TTFT, the per-stage breakdown, the cold (warmup) runs and any client-measured
STT/network series are reported beside it, never folded into it.

Generation uses Groq when ``GROQ_API_KEY`` is set. Without it the run falls back
to :class:`eval.latency.SimulatedGenerator` and **every artefact is stamped
``simulated``** -- the retrieval and guardrail numbers stay real either way.

``--stt-ms`` and ``--network-ms`` accept comma-separated client-side
measurements (from the browser HUD) and render them in their own section, which
is how the report stays honest about the parts of the user-visible latency that
the server budget does not own.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

ROOT = bootstrap()

from eval.dataset import build_corpus, synthetic_rows  # noqa: E402
from eval.latency import (  # noqa: E402
    GROQ_GPT_OSS_20B_PROFILE,
    PipelineBenchmark,
    SimulatedGenerator,
    benchmark,
)


def _floats(spec: str) -> list[float]:
    """Parse a comma-separated list of milliseconds."""
    return [float(x) for x in spec.replace(" ", "").split(",") if x]


def _load_index(directory: Path) -> tuple[Any, Any, Any, list[str], dict[str, Any]]:
    """Load a persisted index built by ``scripts/ingest.py``.

    Returns:
        ``(embedder, hybrid, store, queries, manifest)``.

    Raises:
        FileNotFoundError: If the directory has no manifest.
        RuntimeError: If the manifest describes an embedder that cannot be
            restored -- serving an index with a different projection than it was
            built with returns confidently wrong passages, so this fails loudly
            rather than degrading.
    """
    from voicerag.config import get_settings
    from voicerag.embed.base import resolve_embedder
    from voicerag.index.hybrid import HybridIndex
    from voicerag.index.store import ChunkStore

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found; build one with `python scripts/ingest.py --out {directory}`"
        )
    manifest = json.loads(manifest_path.read_text())

    spec = manifest.get("embedder_spec", "")
    embedder_file = manifest.get("embedder_file") or ""
    if embedder_file:
        from voicerag.embed.lsa import HashingLSAEmbedder

        embedder = HashingLSAEmbedder.load(directory / embedder_file)
    else:
        embedder = resolve_embedder(spec)
        if not getattr(embedder, "is_fitted", True):
            raise RuntimeError(
                f"embedder {spec!r} needs fitting but no fitted file was saved with the "
                "index; rebuild with scripts/ingest.py"
            )

    # Load the index the way the SERVER loads it. This used to call
    # HybridIndex.load(directory) bare, which takes mmap_sparse=True by default
    # while api/state.py passes Settings.mmap_sparse -- so the published
    # benchmark measured a configuration the deployment does not run. It
    # measured the slower one: memory-mapped BM25 page-faults on first touch
    # against the 956k index, costing ~130 ms of retrieve that a preloaded
    # index does not pay. The headline latency number was therefore understating
    # the served system by roughly 5x on the sparse leg.
    hybrid = HybridIndex.load(directory, mmap_sparse=get_settings().mmap_sparse)
    store = ChunkStore.load(directory / "store")
    queries = [q["eng_query"] for q in manifest.get("queries", []) if q.get("eng_query")]
    return embedder, hybrid, store, queries, manifest


def _build_synthetic(n_rows: int, seed: int, strategy: str) -> tuple[Any, Any, Any, list[str], dict[str, Any]]:
    """Build an in-memory index from the offline fixture."""
    from eval.ablation import AblationConfig, build_indexes, default_embedder_spec, fit_embedder
    from voicerag.chunking.registry import build as build_strategy

    corpus = build_corpus(synthetic_rows(n_rows, seed=seed))
    embedder = fit_embedder(
        default_embedder_spec(len(corpus.documents)), [d.text for d in corpus.documents]
    )
    built = build_indexes(
        corpus.documents,
        build_strategy(strategy, embedder=embedder),
        embedder,
        AblationConfig(),
        measure_size=False,
    )
    manifest = {
        "source": f"synthetic:{n_rows}",
        "strategy": strategy,
        "n_documents": len(corpus.documents),
        "n_chunks": built.n_chunks,
        "queries": [q.to_dict() for q in corpus.queries],
        "unanswerable": [q.to_dict() for q in corpus.unanswerable],
    }
    return embedder, built.hybrid, built.store, [q.eng_query for q in corpus.queries], manifest


def _calibrated_policy(
    embedder: Any, hybrid: Any, manifest: dict[str, Any], *, cap: int = 150
) -> Any | None:
    """Fit the abstention gate on this corpus before timing it.

    The shipped prior thresholds are reasoned, not fitted, and the fused score
    scale depends on the corpus and the embedder. Benchmarking with a mismatched
    gate measures the wrong system in the worst possible direction: an
    over-eager gate skips generation and *improves* the latency numbers while
    refusing to answer. So the gate is calibrated on the manifest's labelled
    queries first, exactly as a deployment would.

    Args:
        embedder: The index's embedder.
        hybrid: The built retriever.
        manifest: Index manifest, which carries the labelled query sets.
        cap: Maximum queries per class used for calibration, to bound the cost.

    Returns:
        A :class:`~voicerag.guardrails.policy.GuardrailPolicy` with a calibrated
        gate, or ``None`` when the manifest lacks either class (in which case
        the caller keeps the default policy rather than fitting on one class).
    """
    from eval.abstention_eval import collect_examples
    from eval.dataset import Query
    from voicerag.guardrails.abstention import AbstentionGate
    from voicerag.guardrails.policy import GuardrailPolicy

    answerable = [Query(**q) for q in manifest.get("queries", [])[:cap]]
    unanswerable = [Query(**q) for q in manifest.get("unanswerable", [])[:cap]]
    if not answerable or not unanswerable:
        return None

    def retrieve(text: str) -> Any:
        return hybrid.search(query_text=text, query_vec=embedder.encode([text])[0], k=10)

    examples = collect_examples(answerable + unanswerable, retrieve, k=10)
    gate = AbstentionGate(k=10)
    result = gate.calibrate(
        [e.signals for e in examples if not e.unanswerable],
        [e.signals for e in examples if e.unanswerable],
    )
    print(
        f"calibrated abstention gate on {len(examples)} labelled queries: "
        f"F1 {result.before['f1']:.3f} -> {result.after['f1']:.3f} (in-sample), "
        f"threshold {result.model.decision_threshold:.3f}",
        flush=True,
    )
    return GuardrailPolicy(abstention=gate)


def _generator(args: argparse.Namespace) -> tuple[Any, str | None]:
    """Pick the real generator when credentials exist, else the simulated one.

    Returns:
        ``(generator, simulated_reason)``. ``simulated_reason`` is ``None`` for a
        real provider and otherwise a phrase naming *why* the run was simulated,
        so the published report can say so accurately instead of assuming.
    """
    key = os.getenv("GROQ_API_KEY", "")
    if key and not args.force_simulated:
        from voicerag.generate.groq import GroqGenerator

        return GroqGenerator(api_key=key, max_tokens=args.max_tokens), None
    reason = (
        "The decoder was pinned to the simulated profile with --force-simulated, "
        "so the run is reproducible and consumes no provider quota."
        if args.force_simulated
        else "No LLM credentials were available."
    )
    return SimulatedGenerator(GROQ_GPT_OSS_20B_PROFILE, max_tokens=args.max_tokens), reason


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=Path, default=None, help="index directory from scripts/ingest.py")
    parser.add_argument("--synthetic", type=int, default=0, metavar="N",
                        help="build an in-memory index from N synthetic rows instead")
    parser.add_argument("--strategy", default="sentence_window", help="chunking for --synthetic")
    parser.add_argument("--iterations", type=int, default=200, help="warm measured runs")
    parser.add_argument("--warmup", type=int, default=5, help="cold runs, excluded from percentiles")
    parser.add_argument("--k", type=int, default=5, help="passages passed to the generator")
    parser.add_argument("--max-tokens", type=int, default=80,
                        help="output ceiling; every extra token costs ~1 ms of the budget")
    parser.add_argument("--target-ms", type=float, default=200.0)
    parser.add_argument("--no-guardrails", action="store_true",
                        help="measure without the input guard and abstention gate")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="keep the uncalibrated prior thresholds in the abstention gate")
    parser.add_argument("--force-simulated", action="store_true",
                        help="use the simulated generator even if GROQ_API_KEY is set")
    parser.add_argument("--stt-ms", default="", help="comma-separated client-measured STT latencies")
    parser.add_argument("--network-ms", default="", help="comma-separated client round-trip latencies")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "latency.json")
    args = parser.parse_args(argv)

    if args.index and args.synthetic:
        print("choose either --index or --synthetic, not both", file=sys.stderr)
        return 2
    if not args.index and not args.synthetic:
        args.synthetic = 300
        print("no --index given; benchmarking an in-memory synthetic index", flush=True)

    if args.index:
        embedder, hybrid, store, queries, manifest = _load_index(args.index)
    else:
        embedder, hybrid, store, queries, manifest = _build_synthetic(
            args.synthetic, args.seed, args.strategy
        )
    if not queries:
        print("index manifest carries no queries to benchmark with", file=sys.stderr)
        return 2

    policy = None
    if not args.no_guardrails and not args.no_calibrate:
        policy = _calibrated_policy(embedder, hybrid, manifest)

    generator, simulated_reason = _generator(args)
    pipeline = PipelineBenchmark(
        embedder,
        hybrid,
        store,
        generator,
        k=args.k,
        use_guardrails=not args.no_guardrails,
        policy=policy,
        max_tokens=args.max_tokens,
    )
    print(
        f"benchmarking {args.iterations} warm runs (+{args.warmup} cold) over "
        f"{len(queries)} queries with {pipeline.generator.name}",
        flush=True,
    )

    report = await benchmark(
        pipeline,
        queries,
        iterations=args.iterations,
        warmup=args.warmup,
        target_ms=args.target_ms,
        meta={"index": manifest, "simulated_reason": simulated_reason},
        progress=lambda msg: print(f"  {msg}", flush=True),
    )
    report.with_client_measurements(
        stt_ms=_floats(args.stt_ms), network_ms=_floats(args.network_ms)
    )
    await pipeline.aclose()

    markdown = report.to_markdown()
    print()
    print(markdown)
    report.write_json(args.out)
    args.out.with_suffix(".md").write_text(markdown)
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
