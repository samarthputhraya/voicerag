"""Chunking abstractions.

Every strategy produces `Chunk` objects that carry enough provenance to
(a) cite the exact source passage, and (b) expand to a wider context window at
generation time without re-reading the corpus.

The contract is deliberately narrow so the ablation harness can swap strategies
without touching indexing, retrieval, or generation.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

__all__ = ["Document", "Chunk", "ChunkingStrategy", "sentences", "word_tokens"]


@dataclass(slots=True, frozen=True)
class Document:
    """A source document from the corpus, before any splitting."""

    doc_id: str
    text: str
    title: str = ""
    url: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """An indexable unit of text plus the provenance needed to cite it.

    Attributes:
        chunk_id: Stable, content-addressed id. Identical text under the same
            strategy always yields the same id, which makes ablation runs
            comparable and makes the index idempotent to rebuild.
        embed_text: What actually gets embedded. May differ from `text` when a
            strategy prepends context (metadata-aware, contextual).
        text: The verbatim span, used for citation display.
        context_text: A wider window returned to the generator. Enables the
            "retrieve small, read big" pattern without a second corpus lookup.
        char_start / char_end: Offsets into the parent document, so a citation
            can be resolved back to an exact span.
    """

    chunk_id: str
    doc_id: str
    text: str
    embed_text: str
    context_text: str
    char_start: int
    char_end: int
    ordinal: int
    strategy: str
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)

    def to_row(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "context_text": self.context_text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "ordinal": self.ordinal,
            "strategy": self.strategy,
            "title": self.title,
        }


def _chunk_id(strategy: str, doc_id: str, ordinal: int, text: str) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return f"{strategy}:{doc_id}:{ordinal}:{h}"


# --- lightweight tokenisation -------------------------------------------------
# Deliberately dependency-free. A full sentence splitter (pysbd/spaCy) costs
# 10-40ms per document at ingest scale and buys us very little on MS MARCO
# passages, which are already clean prose.

_SENT_RE = re.compile(
    r"""(?<=[.!?])["')\]]*\s+(?=[A-Z0-9"'(\[])   # terminator + space + capital
        |\n{2,}                                  # or a hard paragraph break
    """,
    re.VERBOSE,
)
_WORD_RE = re.compile(r"\S+")


def sentences(text: str) -> list[tuple[int, int, str]]:
    """Split into sentences, returning (start, end, text) with real offsets.

    Offsets are preserved so downstream chunks can cite exact spans.
    """
    out: list[tuple[int, int, str]] = []
    pos = 0
    for m in _SENT_RE.finditer(text):
        end = m.start()
        seg = text[pos:end]
        if seg.strip():
            out.append((pos, end, seg))
        pos = m.end()
    tail = text[pos:]
    if tail.strip():
        out.append((pos, len(text), tail))
    return out or ([(0, len(text), text)] if text.strip() else [])


def word_tokens(text: str) -> list[tuple[int, int, str]]:
    """Whitespace tokens with offsets. Used as a cheap proxy for model tokens.

    On English prose, 1 word ~= 1.3 BPE tokens; the strategies express their
    budgets in words and the registry documents the conversion.
    """
    return [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]


class ChunkingStrategy(ABC):
    """Base class for every chunking strategy in the ablation."""

    #: Short, stable identifier used in chunk ids, index names and result tables.
    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def split(self, doc: Document) -> Iterator[Chunk]:
        """Yield chunks for a single document."""

    def split_many(self, docs: Iterable[Document]) -> Iterator[Chunk]:
        for d in docs:
            yield from self.split(d)

    # -- helpers available to every subclass ---------------------------------

    def _emit(
        self,
        doc: Document,
        ordinal: int,
        start: int,
        end: int,
        *,
        embed_text: str | None = None,
        context_text: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Chunk:
        text = doc.text[start:end].strip()
        return Chunk(
            chunk_id=_chunk_id(self.name, doc.doc_id, ordinal, text),
            doc_id=doc.doc_id,
            text=text,
            embed_text=(embed_text if embed_text is not None else text),
            context_text=(context_text if context_text is not None else text),
            char_start=start,
            char_end=end,
            ordinal=ordinal,
            strategy=self.name,
            title=doc.title,
            meta={**doc.meta, **(meta or {})},
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


def window_spans(
    units: Sequence[tuple[int, int, str]],
    size: int,
    overlap: int,
) -> Iterator[tuple[int, int]]:
    """Slide a window of `size` units with `overlap`, yielding index ranges.

    Guarantees forward progress even when overlap >= size (a common config bug),
    and never emits a trailing window that is fully contained in its predecessor.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    step = max(1, size - max(0, overlap))
    n = len(units)
    i = 0
    while i < n:
        j = min(i + size, n)
        yield i, j
        if j >= n:
            return
        i += step
