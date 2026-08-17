#!/usr/bin/env python
"""Run the chunking and fusion ablations and write the report.

    python scripts/run_ablation.py --limit 2000 --out reports/ablation.md
    python scripts/run_ablation.py --synthetic 300 --out reports/ablation.md

Every strategy in the registry is chunked, indexed and evaluated against the
same queries, the same qrels and the same fitted embedder, so the only thing
that varies between rows is the chunking. A second pass holds the chunking
fixed and sweeps the fusion method against dense-only and sparse-only controls.

Results are written as markdown (for the README), JSON (machine-readable) and
CSV (spreadsheets). With ``--cache-dir`` set the run is resumable: rows already
computed are loaded instead of recomputed, so an interrupted 20k-passage
ablation does not start from zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from eval.dataset import (  # noqa: E402 - after sys.path bootstrap
    DEFAULT_SHARD,
    DatasetUnavailable,
    build_corpus,
    load_jsonl,
    load_msmarco_xi,
    synthetic_rows,
)


def _load_rows(args: argparse.Namespace) -> list[dict]:
    if args.rows:
        return list(load_jsonl(args.rows, limit=args.limit))
    if args.synthetic:
        return synthetic_rows(args.synthetic, seed=args.seed)
    return list(load_msmarco_xi(args.shard, limit=args.limit))


def _abstention_section(
    corpus,
    queries,
    embedder,
    config,
    strategy: str,
    out: Path,
    seed: int,
) -> str:
    """Evaluate abstention on the labelled unanswerable rows and render it.

    Uses the same corpus and the same retriever the ablation just measured, so
    the guardrail numbers describe the system the table describes. Both the
    uncalibrated priors and a model calibrated on a train split are reported on
    the *same held-out* test split -- reporting only the calibrated number, or
    reporting it on its own training data, would overstate the guardrail.
    """
    from eval.ablation import build_indexes
    from eval.abstention_eval import (
        calibrate_and_evaluate,
        collect_examples,
        evaluate_abstention,
        sweep_to_markdown,
        to_markdown,
    )
    from voicerag.chunking.registry import build as build_strategy

    print(f"  abstention: building {strategy} index for the guardrail evaluation", flush=True)
    built = build_indexes(
        corpus.documents,
        build_strategy(strategy, embedder=embedder),
        embedder,
        config,
        measure_size=False,
    )
    try:
        def retrieve(text: str):
            return built.hybrid.search(
                query_text=text, query_vec=embedder.encode([text])[0], k=10
            )

        examples = collect_examples(
            list(queries) + list(corpus.unanswerable),
            retrieve,
            k=10,
            query_field=config.query_field,
            progress=lambda msg: print(f"  {msg}", flush=True),
        )
        prior = evaluate_abstention(examples, sweep=True)
        calibrated = calibrate_and_evaluate(examples, test_fraction=0.4, seed=seed)
    finally:
        built.close()

    section = to_markdown(
        prior,
        notes=[
            f"Retriever: `{strategy}` chunking, hybrid `{config.fusion}` fusion, "
            "identical to the ablation above.",
            "Labels are MS MARCO's own: a query is unanswerable when every "
            "candidate passage is `is_selected == 0` and the gold answer is "
            "`\"No Answer Present.\"`.",
        ],
    )
    section += (
        "\n**Calibrated on a train split, scored on held-out data**\n\n"
        f"| Gate | Precision | Recall | F1 | False-abstention |\n|---|---:|---:|---:|---:|\n"
        f"| prior thresholds | {calibrated['prior']['metrics']['precision']:.3f} | "
        f"{calibrated['prior']['metrics']['recall']:.3f} | "
        f"{calibrated['prior']['metrics']['f1']:.3f} | "
        f"{calibrated['prior']['metrics']['false_abstention_rate']:.3f} |\n"
        f"| calibrated (threshold {calibrated['fitted_threshold']:.3f}) | "
        f"{calibrated['calibrated']['metrics']['precision']:.3f} | "
        f"{calibrated['calibrated']['metrics']['recall']:.3f} | "
        f"{calibrated['calibrated']['metrics']['f1']:.3f} | "
        f"{calibrated['calibrated']['metrics']['false_abstention_rate']:.3f} |\n\n"
        f"_{calibrated['n_train']} training / {calibrated['n_test']} held-out examples._\n\n"
        "**Threshold sweep (prior gate)**\n\n" + sweep_to_markdown(prior.sweep)
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    prior.write_json(out.parent / "abstention.json")
    (out.parent / "abstention.md").write_text(section)
    return section


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    parser.add_argument("--limit", type=int, default=2000, help="rows to evaluate on")
    parser.add_argument("--synthetic", type=int, default=0, metavar="N",
                        help="use N synthetic rows instead of the hub (offline)")
    parser.add_argument("--rows", type=Path, help="cached .jsonl(.gz) of rows")
    parser.add_argument("--strategies", nargs="*", default=None,
                        help="subset of registry strategies; default is all of them")
    parser.add_argument("--fusion-strategy", default="sentence_window",
                        help="chunking held fixed during the fusion sweep")
    parser.add_argument("--no-fusion", action="store_true", help="skip the fusion axis")
    parser.add_argument("--embedder", default="", help="embedder spec; default picks an LSA dim")
    parser.add_argument("--query-field", default="eng_query",
                        choices=("eng_query", "indic_query"),
                        help="'indic_query' produces the cross-lingual table")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="cap the evaluation queries (0 = all)")
    parser.add_argument("--no-abstention", action="store_true",
                        help="skip the abstention evaluation on the labelled unanswerable set")
    parser.add_argument("--cache-dir", type=Path, default=None, help="per-row result cache")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "ablation.md")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from eval.ablation import (
        AblationConfig,
        ablation_to_csv,
        ablation_to_json,
        ablation_to_markdown,
        default_embedder_spec,
        fit_embedder,
        run_ablation,
        run_fusion_ablation,
    )

    try:
        rows = _load_rows(args)
    except DatasetUnavailable as exc:
        print(f"dataset unavailable: {exc}", file=sys.stderr)
        print("hint: --synthetic 300 runs the full ablation offline", file=sys.stderr)
        return 2

    corpus = build_corpus(rows)
    queries = corpus.queries[: args.max_queries] if args.max_queries else corpus.queries
    print(f"corpus: {json.dumps(corpus.stats())}", flush=True)
    if not queries:
        print("no answerable queries; nothing to evaluate", file=sys.stderr)
        return 2

    spec = args.embedder or default_embedder_spec(len(corpus.documents))
    print(f"fitting {spec} on {len(corpus.documents):,} passages", flush=True)
    embedder = fit_embedder(spec, [d.text for d in corpus.documents])
    config = AblationConfig(query_field=args.query_field)

    t0 = time.perf_counter()
    chunking_rows = run_ablation(
        corpus.documents,
        queries,
        corpus.qrels,
        strategies=args.strategies,
        embedder=embedder,
        config=config,
        cache_dir=args.cache_dir,
        progress=lambda msg: print(f"  {msg}", flush=True),
    )
    fusion_rows: list[dict] = []
    if not args.no_fusion:
        fusion_rows = run_fusion_ablation(
            corpus.documents,
            queries,
            corpus.qrels,
            strategy=args.fusion_strategy,
            embedder=embedder,
            config=config,
            cache_dir=args.cache_dir,
            progress=lambda msg: print(f"  {msg}", flush=True),
        )
    notes = [
        f"Corpus: {len(corpus.documents):,} deduplicated passages from "
        f"{corpus.n_rows:,} rows "
        f"({'synthetic fixture' if args.synthetic else f'{args.shard}'}); "
        f"{len(queries):,} queries, {corpus.n_duplicates:,} duplicate passages collapsed.",
        f"Embedder: `{spec}` (dim {embedder.dim}), fitted once on the passage corpus "
        "and shared by every row so the table isolates chunking.",
        f"Retrieval: hybrid HNSW + BM25, fusion `{config.fusion}`, "
        f"efSearch {config.ef_search}, query field `{config.query_field}`. "
        "Relevance is judged on passages: chunk hits are mapped back to their "
        "source passage id before scoring.",
    ]

    abstention_md = ""
    if not args.no_abstention and corpus.unanswerable:
        abstention_md = _abstention_section(
            corpus, queries, embedder, config, args.fusion_strategy, args.out, args.seed
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    markdown = ablation_to_markdown(chunking_rows, title="Chunking ablation", notes=notes)
    if fusion_rows:
        markdown += "\n" + ablation_to_markdown(
            fusion_rows,
            title=f"Fusion ablation (chunking fixed at `{args.fusion_strategy}`)",
            notes=[
                "`dense` and `sparse` are single-run controls, not fusion methods: "
                "if RRF does not beat both, the hybrid retriever is not earning its "
                "complexity."
            ],
        )
    if abstention_md:
        markdown += "\n" + abstention_md
    args.out.write_text(markdown)
    ablation_to_json(chunking_rows + fusion_rows, args.out.with_suffix(".json"))
    ablation_to_csv(chunking_rows + fusion_rows, args.out.with_suffix(".csv"))

    print(markdown)
    print(f"wrote {args.out}, {args.out.with_suffix('.json')}, "
          f"{args.out.with_suffix('.csv')} in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
