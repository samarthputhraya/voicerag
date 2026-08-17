"""The four model-free chunking strategies.

Semantic and contextual chunking live in `semantic.py` because they need an
embedder / LLM at ingest time.

Design note
-----------
Every strategy here is O(n) in document length and allocation-light, because
ingest runs over hundreds of thousands of passages. None of them call a model,
so they cost microseconds per document and can be re-run freely during ablation.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from .base import Chunk, ChunkingStrategy, Document, sentences, window_spans, word_tokens

__all__ = [
    "FixedOverlapChunker",
    "RecursiveChunker",
    "SentenceWindowChunker",
    "MetadataAwareChunker",
]


class FixedOverlapChunker(ChunkingStrategy):
    """Fixed-width word windows with a fixed overlap.

    This is the naive baseline the brief warns against shipping alone. We keep
    it because an ablation without a baseline proves nothing -- it is the
    control, and every other strategy is measured against it.

    Args:
        size: Window width in whitespace words.
        overlap: Words of overlap between consecutive windows.
    """

    name = "fixed"

    def __init__(self, size: int = 120, overlap: int = 24, **kw: Any) -> None:
        super().__init__(size=size, overlap=overlap, **kw)
        self.size, self.overlap = size, overlap

    def split(self, doc: Document) -> Iterator[Chunk]:
        toks = word_tokens(doc.text)
        if not toks:
            return
        for ordinal, (i, j) in enumerate(window_spans(toks, self.size, self.overlap)):
            yield self._emit(doc, ordinal, toks[i][0], toks[j - 1][1])


class RecursiveChunker(ChunkingStrategy):
    """Split on the most semantic separator that yields small-enough pieces.

    Walks a separator hierarchy (paragraph -> line -> sentence -> clause -> word)
    and only descends when a piece still exceeds the budget. This keeps natural
    boundaries intact wherever the document structure allows it, which is why it
    is the default in most production RAG stacks.

    Args:
        max_chars: Hard ceiling per chunk.
        min_chars: Pieces below this are merged forward, preventing a tail of
            near-empty chunks that pollute the index and dilute BM25 stats.
        overlap_chars: Characters of tail context prepended to the next chunk.
    """

    name = "recursive"
    SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")

    def __init__(
        self,
        max_chars: int = 700,
        min_chars: int = 120,
        overlap_chars: int = 120,
        **kw: Any,
    ) -> None:
        super().__init__(
            max_chars=max_chars, min_chars=min_chars, overlap_chars=overlap_chars, **kw
        )
        self.max_chars, self.min_chars, self.overlap_chars = max_chars, min_chars, overlap_chars

    def _split_span(self, text: str, base: int, depth: int = 0) -> list[tuple[int, int]]:
        """Recursively split [base, base+len(text)) into spans under the ceiling."""
        if len(text) <= self.max_chars or depth >= len(self.SEPARATORS):
            return [(base, base + len(text))]

        sep = self.SEPARATORS[depth]
        parts = text.split(sep)
        if len(parts) == 1:
            return self._split_span(text, base, depth + 1)

        spans: list[tuple[int, int]] = []
        cursor = base
        buf_start, buf_len = cursor, 0
        for k, part in enumerate(parts):
            piece = part + (sep if k < len(parts) - 1 else "")
            # Flush when adding this piece would breach the ceiling.
            if buf_len and buf_len + len(piece) > self.max_chars:
                spans.extend(
                    self._split_span(text[buf_start - base : buf_start - base + buf_len],
                                     buf_start, depth + 1)
                )
                buf_start, buf_len = cursor, 0
            buf_len += len(piece)
            cursor += len(piece)
        if buf_len:
            spans.extend(
                self._split_span(text[buf_start - base : buf_start - base + buf_len],
                                 buf_start, depth + 1)
            )
        return spans

    def split(self, doc: Document) -> Iterator[Chunk]:
        if not doc.text.strip():
            return
        spans = self._split_span(doc.text, 0)

        # Merge undersized spans forward so we never index a 20-char fragment.
        merged: list[tuple[int, int]] = []
        for s, e in spans:
            if merged and (e - s) < self.min_chars:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        for ordinal, (s, e) in enumerate(merged):
            ctx_start = max(0, s - self.overlap_chars)
            ctx_end = min(len(doc.text), e + self.overlap_chars)
            embed = doc.text[ctx_start:e].strip() if ctx_start < s else None
            yield self._emit(
                doc, ordinal, s, e,
                embed_text=embed,
                context_text=doc.text[ctx_start:ctx_end].strip(),
            )


class SentenceWindowChunker(ChunkingStrategy):
    """Retrieve small, read big.

    Indexes one sentence (or a tight group) but attaches a wider surrounding
    window as `context_text`. Small units embed sharply -- a single sentence has
    one topic, so its vector is not smeared across three -- while the generator
    still receives enough surrounding prose to answer without truncation.

    This is usually the strongest strategy on short factoid queries, which is
    exactly the MS MARCO distribution.

    Args:
        group: Sentences per indexed unit.
        window: Sentences of context on each side.
    """

    name = "sentence_window"

    def __init__(self, group: int = 1, window: int = 3, **kw: Any) -> None:
        super().__init__(group=group, window=window, **kw)
        self.group, self.window = group, window

    def split(self, doc: Document) -> Iterator[Chunk]:
        sents = sentences(doc.text)
        if not sents:
            return
        for ordinal, (i, j) in enumerate(window_spans(sents, self.group, 0)):
            lo = max(0, i - self.window)
            hi = min(len(sents), j + self.window)
            yield self._emit(
                doc, ordinal, sents[i][0], sents[j - 1][1],
                context_text=doc.text[sents[lo][0] : sents[hi - 1][1]].strip(),
                meta={"ctx_sentences": hi - lo},
            )


class MetadataAwareChunker(ChunkingStrategy):
    """Prepend document metadata to every chunk before embedding.

    A passage that says "It was signed in 1919" is unretrievable on its own --
    the vector has no idea what "it" is. Prefixing the title (and any structural
    metadata) restores the referent, which measurably lifts recall on documents
    whose body text relies on the title for context.

    Crucially the prefix goes into `embed_text` only. `text` stays verbatim, so
    citations never show synthetic text the source document does not contain.

    Args:
        size / overlap: Window geometry, in words, same as the fixed baseline
            so the comparison isolates the effect of the prefix alone.
        meta_keys: Additional `doc.meta` keys to fold into the prefix.
    """

    name = "metadata"

    def __init__(
        self,
        size: int = 120,
        overlap: int = 24,
        meta_keys: tuple[str, ...] = ("section", "url"),
        **kw: Any,
    ) -> None:
        super().__init__(size=size, overlap=overlap, meta_keys=list(meta_keys), **kw)
        self.size, self.overlap, self.meta_keys = size, overlap, meta_keys

    def _prefix(self, doc: Document, ordinal: int, total: int) -> str:
        bits: list[str] = []
        if doc.title:
            bits.append(doc.title.strip())
        for k in self.meta_keys:
            v = doc.meta.get(k)
            if v:
                bits.append(f"{k}: {v}")
        if total > 1:
            bits.append(f"part {ordinal + 1} of {total}")
        return " | ".join(bits)

    def split(self, doc: Document) -> Iterator[Chunk]:
        toks = word_tokens(doc.text)
        if not toks:
            return
        spans = list(window_spans(toks, self.size, self.overlap))
        for ordinal, (i, j) in enumerate(spans):
            s, e = toks[i][0], toks[j - 1][1]
            body = doc.text[s:e].strip()
            pre = self._prefix(doc, ordinal, len(spans))
            yield self._emit(
                doc, ordinal, s, e,
                embed_text=f"{pre}\n{body}" if pre else body,
                meta={"prefix": pre},
            )
