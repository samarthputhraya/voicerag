"""Find the questions this deployment can actually answer, and publish the list.

Why this exists
---------------

``GET /examples`` offers a judge a row of chips captioned "this corpus can
answer questions like". Until now those chips were sampled from the shard's
own ``answerable: True`` labels -- a claim about the *dataset*, not about this
pipeline. The two disagree, and the disagreement is visible on camera:

    "How is caffeine metabolized?"  is labelled answerable, retrieves cleanly
    (abstention confidence 0.14, dense_max 0.78), and is then refused by the
    model -- correctly. Its gold answer in MS MARCO is about "fat burning
    supplements ... increase metabolism", not the biochemistry the question
    asks. The passages genuinely do not contain the answer.

A chip is a promise. Offering a question the system will decline is the single
most avoidable way to look broken, and no cheap signal predicts it: the
abstention gate passes every chip in the current set, so only a real generation
call tells you the truth. So we pay for that once, offline, and serve the
result.

This is the same habit as the rest of the repo: measure, publish the artifact,
and serve exactly what was measured.

Usage
-----

    python scripts/verify_examples.py                 # verify 24 candidates
    python scripts/verify_examples.py --candidates 40 --keep 8

Writes ``reports/examples.json`` (served by ``GET /examples``) and
``reports/examples.md`` (the evidence, including what was rejected and why).
Needs a generation credential; costs one LLM call per candidate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

ROOT = bootstrap()

from voicerag.api.main import _presentable_question  # noqa: E402
from voicerag.api.state import AppState  # noqa: E402
from voicerag.config import REPO_ROOT, get_settings  # noqa: E402

#: Seconds between calls. Groq's free tier is ~8,000 tokens/minute and one RAG
#: answer costs roughly 1,400, so anything faster fails for a reason that has
#: nothing to do with whether the question is answerable.
PACE_S = 13.0


async def _verify_one(pipeline: Any, question: str) -> dict[str, Any]:
    """Run one candidate all the way through, exactly as a user would."""
    response = await pipeline.answer(question)
    guard = response.guardrails
    return {
        "question": question,
        "answered": not response.abstained,
        "answer": response.answer,
        "grounding": getattr(guard, "grounding_score", None),
        "abstain_confidence": getattr(guard, "abstain_confidence", None),
        "n_citations": len(response.citations or []),
        "provider": response.provider,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=24,
                        help="How many presentable queries to try.")
    parser.add_argument("--keep", type=int, default=8,
                        help="How many verified questions to publish.")
    parser.add_argument("--out", default=str(REPO_ROOT / "reports"))
    args = parser.parse_args()

    cfg = get_settings()
    state = AppState(cfg)
    await state.startup()
    pipeline = state.pipeline
    if pipeline is None or state.bundle is None:
        print("no index loaded; build one with scripts/ingest.py first")
        return 1
    if not state.provider_names():
        print("no generation provider configured; this script needs one to tell")
        print("an answerable question from one the model will decline")
        return 1

    manifest = state.bundle.manifest
    pool: list[str] = []
    seen: set[str] = set()
    for raw in manifest.get("example_queries", []):
        q = _presentable_question(raw)
        if q and q.lower() not in seen:
            seen.add(q.lower())
            pool.append(q)

    print(f"{len(pool)} presentable candidates in the manifest; trying {args.candidates}")
    print()

    results: list[dict[str, Any]] = []
    for i, question in enumerate(pool[: args.candidates]):
        if i:
            time.sleep(PACE_S)
        try:
            verdict = await _verify_one(pipeline, question)
        except Exception as exc:  # noqa: BLE001 - a failed candidate is just rejected
            print(f"  [err ] {question[:58]:<60} {type(exc).__name__}")
            continue
        results.append(verdict)
        mark = "keep" if verdict["answered"] else "drop"
        detail = (
            f"grounding={verdict['grounding']:.2f}"
            if isinstance(verdict["grounding"], (int, float))
            else "declined"
        )
        print(f"  [{mark}] {question[:58]:<60} {detail}")

    # Grounding is the tiebreak, not the filter: every kept question was
    # answered, and a better-supported answer is a better thing to put in front
    # of a judge.
    kept = sorted(
        (r for r in results if r["answered"]),
        key=lambda r: (r["grounding"] if isinstance(r["grounding"], (int, float)) else 0.0),
        reverse=True,
    )[: args.keep]
    dropped = [r for r in results if not r["answered"]]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_from": {
            "index_dir": str(cfg.index_dir),
            "n_chunks": pipeline.n_chunks,
            "embedder_spec": state.bundle.embedder_spec,
            "strategy": state.bundle.strategy,
        },
        "verified": [r["question"] for r in kept],
        "rejected": [
            {"question": r["question"], "reason": (r["answer"] or "")[:200]} for r in dropped
        ],
        "detail": kept,
    }
    (out_dir / "examples.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Verified example questions",
        "",
        "Every question below was run through the **whole** pipeline -- retrieval,",
        "abstention gate, generation and grounding -- against the index this",
        "deployment serves, and answered. They are what `GET /examples` offers.",
        "",
        "The rejected list matters more than the kept one. Those questions are",
        "labelled `answerable: True` by MS MARCO and are refused anyway, because",
        "the labelled gold answer does not address what the question asks. A chip",
        "is a promise; these are the promises this corpus cannot keep.",
        "",
        f"- index: `{cfg.index_dir}` ({pipeline.n_chunks:,} chunks, "
        f"{state.bundle.strategy}, `{state.bundle.embedder_spec}`)",
        f"- candidates tried: {len(results)}",
        f"- answered: {len(results) - len(dropped)} · declined: {len(dropped)}",
        "",
        "## Published",
        "",
        "| question | grounding | citations |",
        "|---|---:|---:|",
    ]
    for r in kept:
        g = f"{r['grounding']:.2f}" if isinstance(r["grounding"], (int, float)) else "-"
        lines.append(f"| {r['question']} | {g} | {r['n_citations']} |")
    if dropped:
        lines += ["", "## Rejected — labelled answerable, declined anyway", "",
                  "| question | what the system said |", "|---|---|"]
        for r in dropped:
            reason = (r["answer"] or "").replace("\n", " ")[:120]
            lines.append(f"| {r['question']} | {reason} |")
    (out_dir / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"kept {len(kept)}, rejected {len(dropped)} -> {out_dir / 'examples.json'}")
    await state.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
