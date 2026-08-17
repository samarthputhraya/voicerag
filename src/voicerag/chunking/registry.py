"""The ablation grid.

One place that knows every strategy and its default configuration, so the
benchmark, the API and the docs can never drift out of sync.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import ChunkingStrategy
from .semantic import ContextualChunker, Embedder, SemanticChunker
from .strategies import (
    FixedOverlapChunker,
    MetadataAwareChunker,
    RecursiveChunker,
    SentenceWindowChunker,
)

__all__ = ["STRATEGIES", "build", "build_all", "describe_all"]


#: name -> (factory, needs_embedder, one-line rationale for the report)
STRATEGIES: dict[str, tuple[Callable[..., ChunkingStrategy], bool, str]] = {
    "fixed": (
        FixedOverlapChunker,
        False,
        "Control. Fixed 120-word windows, 24-word overlap. The naive split every "
        "other strategy is measured against.",
    ),
    "recursive": (
        RecursiveChunker,
        False,
        "Descends a separator hierarchy so chunks break on paragraphs and "
        "sentences rather than mid-clause.",
    ),
    "sentence_window": (
        SentenceWindowChunker,
        False,
        "Retrieve small, read big: indexes one sentence, serves a 7-sentence "
        "window to the generator.",
    ),
    "metadata": (
        MetadataAwareChunker,
        False,
        "Fixed geometry plus a title/section prefix on the embedded text only, "
        "restoring referents that bare passages lose.",
    ),
    "semantic": (
        SemanticChunker,
        True,
        "Boundaries placed at per-document similarity percentiles, so cuts land "
        "where the topic actually changes.",
    ),
    "contextual": (
        ContextualChunker,
        False,
        "Overlay that prepends a situating blurb (title + lead) to each chunk "
        "before embedding, without altering cited text.",
    ),
}


def build(
    name: str,
    *,
    embedder: Embedder | None = None,
    **overrides: Any,
) -> ChunkingStrategy:
    """Instantiate one strategy by name."""
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; have {sorted(STRATEGIES)}")
    factory, needs_embedder, _ = STRATEGIES[name]

    if name == "contextual":
        base = overrides.pop("base", None) or RecursiveChunker()
        return ContextualChunker(base=base, **overrides)
    if needs_embedder:
        return factory(embedder=embedder, **overrides)
    return factory(**overrides)


def build_all(
    *,
    embedder: Embedder | None = None,
    skip: tuple[str, ...] = (),
) -> dict[str, ChunkingStrategy]:
    """Instantiate the full ablation grid.

    Strategies that need an embedder are skipped (not crashed) when none is
    supplied, so a lightweight run still produces a valid partial table.
    """
    out: dict[str, ChunkingStrategy] = {}
    for name, (_, needs_embedder, _) in STRATEGIES.items():
        if name in skip or (needs_embedder and embedder is None):
            continue
        out[name] = build(name, embedder=embedder)
    return out


def describe_all() -> list[dict[str, str]]:
    """Rows for the benchmark report's strategy table."""
    return [{"name": n, "rationale": r} for n, (_, _, r) in STRATEGIES.items()]
