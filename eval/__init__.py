"""Evaluation harness: datasets, IR metrics, ablations, latency, abstention.

This package is the evidence layer of the project. Everything it produces is
meant to be read by someone who does not trust us: the numbers in the README,
the ablation table that justifies the chunking choice, and the latency report
that is checked against the 200 ms requirement all originate here.

Three rules hold throughout:

* **Offline first.** Every module runs with no network. The real dataset loader
  lazily imports :mod:`datasets` and fails with a remediation message; the
  synthetic fixture generator in :mod:`eval.dataset` produces rows with the
  identical schema, so the full ablation, the latency benchmark and the test
  suite work on a machine that has never seen the internet.
* **One definition per number.** Percentiles use nearest-rank, defined once in
  :mod:`eval.metrics` and consumed by the report, the API and the frontend HUD.
* **Honest labelling.** Anything simulated (notably generation latency when no
  LLM key is present) is marked as simulated in the emitted JSON *and* in the
  markdown, because a benchmark that quietly reports a fake number is worse
  than no benchmark.

Submodules are imported lazily via :func:`__getattr__` so that
``import eval.metrics`` does not drag in faiss, bm25s or the chunking registry.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "abstention_eval",
    "ablation",
    "dataset",
    "latency",
    "metrics",
]


def __getattr__(name: str) -> Any:
    """Import submodules on first attribute access.

    Keeps ``eval.metrics`` (pure stdlib + numpy) cheap even though
    ``eval.ablation`` pulls in the whole retrieval stack.
    """
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
