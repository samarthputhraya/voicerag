"""Model-assisted chunking: semantic breakpoints and contextual enrichment.

Both strategies need something beyond string manipulation, so they take an
injected dependency and degrade gracefully when it is absent -- ingest of a
500k-passage corpus must never hard-fail because one model is unavailable.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Protocol, Sequence

import numpy as np

from .base import Chunk, ChunkingStrategy, Document, sentences

__all__ = ["SemanticChunker", "ContextualChunker", "Embedder"]


class Embedder(Protocol):
    """Minimal structural type so chunking never imports a concrete backend."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SemanticChunker(ChunkingStrategy):
    """Cut where the topic actually changes, not every N characters.

    Embeds each sentence, measures cosine similarity between consecutive
    sentences, and places a boundary wherever similarity falls into the bottom
    `percentile` of that document's own distribution.

    Using a *per-document percentile* rather than a global threshold is the
    important detail: absolute cosine values drift with document length, style
    and embedding model, so a fixed 0.7 cutoff that works on encyclopedic prose
    silently over-splits conversational text. A percentile adapts per document.

    A `buffer` of neighbouring sentences is averaged into each comparison to
    damp single-sentence noise (one short quote should not trigger a boundary).

    Args:
        percentile: Similarity percentile below which a boundary is placed.
            Lower -> fewer, larger chunks.
        buffer: Sentences averaged on each side when computing similarity.
        min_sentences / max_sentences: Hard floor and ceiling per chunk, so a
            pathological document cannot produce 1-sentence or 200-sentence
            chunks regardless of what the similarity curve says.
    """

    name = "semantic"

    def __init__(
        self,
        embedder: Embedder | None = None,
        percentile: float = 25.0,
        buffer: int = 1,
        min_sentences: int = 2,
        max_sentences: int = 12,
        **kw: Any,
    ) -> None:
        super().__init__(
            percentile=percentile, buffer=buffer,
            min_sentences=min_sentences, max_sentences=max_sentences, **kw,
        )
        self.embedder = embedder
        self.percentile = percentile
        self.buffer = buffer
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def _boundaries(self, sents: list[str]) -> list[int]:
        """Indices *after* which a chunk boundary falls."""
        n = len(sents)
        if n <= self.min_sentences or self.embedder is None:
            return []

        # Combine each sentence with its neighbours to reduce noise.
        combined = [
            " ".join(sents[max(0, i - self.buffer) : min(n, i + self.buffer + 1)])
            for i in range(n)
        ]
        V = np.asarray(self.embedder.encode(combined), dtype=np.float32)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        V /= norms

        sims = np.einsum("ij,ij->i", V[:-1], V[1:])  # cosine, consecutive pairs
        if sims.size == 0:
            return []
        cutoff = float(np.percentile(sims, self.percentile))

        cuts: list[int] = []
        run = 1
        for i, s in enumerate(sims):
            run += 1
            hard_cut = run >= self.max_sentences
            soft_cut = s <= cutoff and run >= self.min_sentences
            if hard_cut or soft_cut:
                cuts.append(i)  # boundary after sentence i
                run = 0
        return cuts

    def split(self, doc: Document) -> Iterator[Chunk]:
        sents = sentences(doc.text)
        if not sents:
            return
        texts = [t for _, _, t in sents]
        cuts = set(self._boundaries(texts))

        ordinal, start_i = 0, 0
        for i in range(len(sents)):
            if i in cuts or i == len(sents) - 1:
                yield self._emit(doc, ordinal, sents[start_i][0], sents[i][1],
                                 meta={"n_sentences": i - start_i + 1})
                ordinal += 1
                start_i = i + 1


class ContextualChunker(ChunkingStrategy):
    """Situate every chunk inside its document before embedding it.

    Follows the "contextual retrieval" idea: a chunk embedded in isolation loses
    the discourse context that makes it findable. We prepend a short situating
    blurb so the vector encodes *what this passage is about* alongside its
    literal content.

    Two modes:

    - ``extractive`` (default): the blurb is built deterministically from the
      title plus the document's lead sentence. Costs microseconds, runs over the
      whole corpus, and captures most of the benefit.
    - ``generated``: the blurb comes from an injected LLM callable. Higher
      quality, but one call per chunk -- only viable on a sampled subset, so the
      ablation reports it on a subset and says so.

    As with metadata chunking, the blurb only enters `embed_text`; `text` stays
    verbatim so citations remain faithful to the source.

    Args:
        base: The strategy that produces the underlying spans. Contextual
            enrichment is an *overlay*, so it composes with any splitter.
        mode: ``"extractive"`` or ``"generated"``.
        summarize: Callable ``(doc_text, chunk_text) -> str`` for generated mode.
        lead_chars: How much of the document lead to use in extractive mode.
    """

    name = "contextual"

    def __init__(
        self,
        base: ChunkingStrategy,
        mode: str = "extractive",
        summarize: Callable[[str, str], str] | None = None,
        lead_chars: int = 240,
        **kw: Any,
    ) -> None:
        super().__init__(base=base.name, mode=mode, lead_chars=lead_chars, **kw)
        if mode not in {"extractive", "generated"}:
            raise ValueError("mode must be 'extractive' or 'generated'")
        if mode == "generated" and summarize is None:
            raise ValueError("generated mode requires a `summarize` callable")
        self.base = base
        self.mode = mode
        self.summarize = summarize
        self.lead_chars = lead_chars
        self._cache: dict[str, str] = {}

    def _lead(self, doc: Document) -> str:
        sents = sentences(doc.text)
        lead, total = [], 0
        for _, _, t in sents:
            t = t.strip()
            if total + len(t) > self.lead_chars and lead:
                break
            lead.append(t)
            total += len(t)
        return " ".join(lead)

    def _context_for(self, doc: Document, chunk: Chunk) -> str:
        if self.mode == "generated":
            key = chunk.chunk_id
            if key not in self._cache:
                assert self.summarize is not None
                try:
                    self._cache[key] = self.summarize(doc.text, chunk.text).strip()
                except Exception:
                    # Never let enrichment failure abort a corpus ingest.
                    self._cache[key] = self._lead(doc)
            return self._cache[key]

        parts = [p for p in (doc.title.strip(), self._lead(doc)) if p]
        return " — ".join(parts)

    def split(self, doc: Document) -> Iterator[Chunk]:
        for chunk in self.base.split(doc):
            ctx = self._context_for(doc, chunk)
            chunk.strategy = self.name
            chunk.embed_text = f"{ctx}\n\n{chunk.text}" if ctx else chunk.text
            chunk.meta = {**chunk.meta, "situating_context": ctx, "base": self.base.name}
            yield chunk
