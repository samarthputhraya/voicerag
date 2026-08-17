#!/usr/bin/env python
"""End-to-end offline smoke test: no network, no credentials, no downloads.

Runs the whole system on synthetic data that has the exact MSMARCO-XI row
schema -- chunking, embedding, dense + sparse indexing, hybrid fusion,
retrieval metrics, guardrails, a simulated generation, and the abstention
evaluation -- and fails loudly if any stage misbehaves.

This exists because "it works on my machine with my API keys" is not
verifiable. A judge (or a teammate on a plane) can run this on a fresh clone
and see every seam exercised in a few seconds::

    python scripts/smoke.py            # summary
    python scripts/smoke.py --verbose  # plus the retrieved passages

Exit code is 0 only if every assertion holds.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from _bootstrap import bootstrap

ROOT = bootstrap()

from eval.ablation import (  # noqa: E402 - after sys.path bootstrap
    AblationConfig,
    build_indexes,
    default_embedder_spec,
    fit_embedder,
    run_queries,
)
from eval.abstention_eval import (  # noqa: E402
    calibrate_and_evaluate,
    collect_examples,
    evaluate_abstention,
)
from eval.dataset import build_corpus, synthetic_rows  # noqa: E402
from eval.latency import PipelineBenchmark, SimulatedGenerator, benchmark  # noqa: E402
from eval.metrics import aggregate  # noqa: E402


class SmokeFailure(AssertionError):
    """A stage produced a result the system's own contracts forbid."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _say(step: str, detail: str) -> None:
    print(f"  {step:<22} {detail}", flush=True)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", type=int, default=60, help="synthetic rows to generate")
    parser.add_argument("--strategy", default="sentence_window", help="chunking strategy")
    parser.add_argument("--k", type=int, default=5, help="passages retrieved per query")
    parser.add_argument("--iterations", type=int, default=12, help="latency iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    from voicerag.chunking.registry import build as build_strategy
    from voicerag.guardrails.abstention import AbstentionGate
    from voicerag.guardrails.policy import GuardrailPolicy

    print("VoiceRAG offline smoke test")
    print("=" * 72)

    # 1. Corpus ---------------------------------------------------------------
    rows = synthetic_rows(args.queries, seed=args.seed)
    corpus = build_corpus(rows)
    stats = corpus.stats()
    _check(stats["n_documents"] > 0, "corpus produced no documents")
    _check(stats["n_queries"] > 0, "corpus produced no answerable queries")
    _check(stats["n_unanswerable"] > 0, "fixture produced no unanswerable queries")
    _check(
        stats["n_duplicates_collapsed"] > 0,
        "fixture produced no duplicate passages, so the dedup path is untested",
    )
    _say("corpus", f"{stats['n_documents']} docs, {stats['n_queries']} queries, "
                   f"{stats['n_unanswerable']} unanswerable, "
                   f"{stats['n_duplicates_collapsed']} dupes collapsed")

    # Every gold label must resolve to a surviving document. This is the
    # invariant that silently breaks every retrieval metric when it fails.
    doc_ids = {d.doc_id for d in corpus.documents}
    missing = {
        qid: sorted(gold - doc_ids) for qid, gold in corpus.qrels.items() if gold - doc_ids
    }
    _check(not missing, f"gold passages lost during dedup: {missing}")
    _say("qrels", f"{sum(len(v) for v in corpus.qrels.values())} gold labels, all resolvable")

    # 2. Embedder + index -----------------------------------------------------
    spec = default_embedder_spec(len(corpus.documents))
    embedder = fit_embedder(spec, [d.text for d in corpus.documents])
    _say("embedder", f"{spec}, dim={embedder.dim}, fitted offline")

    config = AblationConfig(k_retrieve=max(10, args.k))
    built = build_indexes(
        corpus.documents, build_strategy(args.strategy, embedder=embedder), embedder, config
    )
    _check(built.n_chunks >= len(corpus.documents), "chunking lost documents")
    _say("index", f"{built.n_chunks} chunks, build {built.build_ms:.0f} ms, "
                  f"{built.index_bytes / (1 << 20):.2f} MB serialised")

    try:
        # 3. Retrieval quality -------------------------------------------------
        results = run_queries(built, embedder, corpus.queries, corpus.qrels, config)
        summary = aggregate(results, ks=(1, 5, 10))
        _check(summary.n_queries == len(corpus.queries), "some queries were not scored")
        _check(
            summary.metrics["recall@10"] > 0.5,
            f"recall@10 = {summary.metrics['recall@10']:.3f}; retrieval is broken, "
            "the fixture's gold passage is findable by construction",
        )
        _say(
            "retrieval",
            f"R@1={summary.metrics['recall@1']:.3f} R@5={summary.metrics['recall@5']:.3f} "
            f"R@10={summary.metrics['recall@10']:.3f} "
            f"nDCG@10={summary.metrics['ndcg@10']:.3f} "
            f"MRR@10={summary.metrics['mrr@10']:.3f} "
            f"(p50 {summary.latency.pcts['p50']:.2f} ms)",
        )

        # 4. Guardrails: abstention on the labelled unanswerable set -----------
        def retrieve(text: str) -> Any:
            vec = embedder.encode([text])[0]
            return built.hybrid.search(query_text=text, query_vec=vec, k=10)

        examples = collect_examples(
            list(corpus.queries) + list(corpus.unanswerable), retrieve, k=10
        )
        abst = evaluate_abstention(examples, sweep=True)
        _check(abst.n_unanswerable > 0, "no unanswerable examples reached the gate")
        _check(len(abst.sweep) > 3, "threshold sweep produced too few points")
        _say(
            "abstention",
            f"prior rules: P={abst.metrics['precision']:.3f} R={abst.metrics['recall']:.3f} "
            f"F1={abst.metrics['f1']:.3f} "
            f"(tp={abst.confusion['tp']} fp={abst.confusion['fp']} "
            f"tn={abst.confusion['tn']} fn={abst.confusion['fn']})",
        )

        calib = calibrate_and_evaluate(examples, test_fraction=0.4, seed=args.seed)
        # Asserted: calibration ran, on a real split, and produced a usable
        # model. NOT asserted: that held-out F1 improved. On a few dozen
        # fixture examples that is a coin flip, and a smoke test that fails on
        # sampling noise trains people to ignore it. The delta is printed so a
        # real run over MS MARCO can be compared against it.
        _check(calib["n_test"] > 0 and calib["n_train"] > 0, "empty calibration split")
        _check(bool(calib["coefficients"]), "calibration produced no coefficients")
        _check(0.0 < calib["fitted_threshold"] < 1.0, "fitted threshold outside (0, 1)")
        _say(
            "  calibrated",
            f"held-out F1 {calib['prior']['metrics']['f1']:.3f} -> "
            f"{calib['calibrated']['metrics']['f1']:.3f} "
            f"(threshold {calib['fitted_threshold']:.3f}, "
            f"{calib['n_train']} train / {calib['n_test']} test)",
        )

        # 5. Full pipeline with simulated generation --------------------------
        # The gate is calibrated on this corpus first. The shipped priors are
        # tuned for real MS MARCO score scales; running them unchanged over a
        # fixture with a different distribution would abstain on almost
        # everything and measure the wrong pipeline.
        gate = AbstentionGate(k=10)
        answerable = [e.signals for e in examples if not e.unanswerable]
        unanswerable = [e.signals for e in examples if e.unanswerable]
        gate.calibrate(answerable, unanswerable)
        pipeline = PipelineBenchmark(
            embedder,
            built.hybrid,
            built.store,
            SimulatedGenerator(),
            k=args.k,
            policy=GuardrailPolicy(abstention=gate),
        )
        answerable_queries = [
            q.eng_query
            for q in corpus.queries
            if not gate.judge(retrieve(q.eng_query)).should_abstain
        ]
        _check(
            len(answerable_queries) >= 5,
            f"the calibrated gate answers only {len(answerable_queries)} of "
            f"{len(corpus.queries)} answerable queries; nothing left to time",
        )
        probe = await pipeline.run(answerable_queries[0])
        _check(probe.error is None, f"pipeline errored: {probe.error}")
        _check(probe.answer != "" or probe.abstained, "pipeline produced neither answer nor abstention")
        _say(
            "pipeline",
            f"{'abstained' if probe.abstained else 'answered'} in {probe.pipeline_ms:.1f} ms "
            f"(ttft {probe.ttft_ms:.1f} ms), stages: "
            + ", ".join(f"{k}={v:.2f}" for k, v in sorted(probe.stages.items())),
        )

        report = await benchmark(
            pipeline,
            answerable_queries[:10],
            iterations=args.iterations,
            warmup=3,
        )
        head = report.summaries["pipeline_total"]
        _check(head.n > 0, "no answered runs in the latency benchmark")
        _say(
            "latency",
            f"p50={head.pcts['p50']:.1f} p70={head.pcts['p70']:.1f} "
            f"p100={head.pcts['p100']:.1f} ms over {head.n} warm runs "
            f"({report.meta['generation']})",
        )

        if args.verbose:
            first = corpus.queries[0]
            print("\nSample retrieval for:", first.eng_query)
            for hit in retrieve(first.eng_query)[:3]:
                chunk = built.store.get(hit.chunk_id)
                gold = chunk.doc_id in corpus.qrels[first.query_id]
                print(f"  [{hit.rank}] {'GOLD' if gold else '    '} "
                      f"score={hit.score:.4f} src={hit.source} :: {chunk.text[:90]}")
    finally:
        built.close()

    print("=" * 72)
    print("OK - every stage ran offline with no network and no credentials.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except SmokeFailure as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
