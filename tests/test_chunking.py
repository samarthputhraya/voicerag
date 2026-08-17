"""Correctness tests for the chunking layer.

These assert the invariants the rest of the pipeline relies on:
offsets resolve, no chunk is empty, coverage is complete, and citation text is
never contaminated by synthetic prefixes.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicerag.chunking.base import Document, sentences, window_spans, word_tokens
from voicerag.chunking.registry import STRATEGIES, build, build_all
from voicerag.chunking.semantic import ContextualChunker, SemanticChunker
from voicerag.chunking.strategies import RecursiveChunker

LONG = (
    "The Treaty of Versailles was one of the peace treaties that ended the First "
    "World War. It was signed on 28 June 1919 in the Hall of Mirrors. The treaty "
    "required Germany to disarm and make territorial concessions.\n\n"
    "Economists criticised the reparations clauses immediately. John Maynard "
    "Keynes argued the sums were unpayable. His book sold widely and shaped "
    "opinion in Britain. Later historians have partly revised that view.\n\n"
    "Photosynthesis is unrelated to treaties. Plants convert light energy into "
    "chemical energy. Chlorophyll absorbs light most strongly in the blue and red "
    "parts of the spectrum. Oxygen is released as a by-product."
)

DOC = Document(
    doc_id="d1",
    text=LONG,
    title="Treaty of Versailles",
    meta={"section": "Overview", "url": "https://example.org/versailles"},
)


class FakeEmbedder:
    """Deterministic bag-of-words embedder. No downloads, no network.

    Real enough that semantically different sentences get low cosine similarity,
    which is all SemanticChunker needs to be exercised meaningfully.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return out / n


# --- primitives ---------------------------------------------------------------

def test_sentence_offsets_resolve_exactly():
    for s, e, text in sentences(LONG):
        assert LONG[s:e] == text


def test_word_token_offsets_resolve_exactly():
    for s, e, tok in word_tokens(LONG):
        assert LONG[s:e] == tok


def test_window_spans_progress_even_when_overlap_exceeds_size():
    spans = list(window_spans([(0, 1, "x")] * 10, size=3, overlap=99))
    assert len(spans) < 50, "overlap >= size must not loop forever"
    assert spans[-1][1] == 10


def test_window_spans_cover_everything():
    units = [(i, i + 1, "x") for i in range(23)]
    spans = list(window_spans(units, size=5, overlap=2))
    covered = set()
    for i, j in spans:
        covered |= set(range(i, j))
    assert covered == set(range(23))


# --- every strategy -----------------------------------------------------------

@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_produces_valid_chunks(name):
    strat = build(name, embedder=FakeEmbedder())
    chunks = list(strat.split(DOC))
    assert chunks, f"{name} produced no chunks"

    for c in chunks:
        assert c.text.strip(), f"{name} produced an empty chunk"
        assert c.embed_text.strip()
        assert c.context_text.strip()
        assert 0 <= c.char_start < c.char_end <= len(DOC.text)
        assert c.doc_id == "d1"
        # The verbatim span must really appear in the source document.
        assert c.text in DOC.text, f"{name} chunk text is not a substring of the doc"

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), f"{name} produced duplicate chunk ids"


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_chunk_ids_are_deterministic(name):
    a = [c.chunk_id for c in build(name, embedder=FakeEmbedder()).split(DOC)]
    b = [c.chunk_id for c in build(name, embedder=FakeEmbedder()).split(DOC)]
    assert a == b, f"{name} is not reproducible across runs"


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_empty_and_whitespace_documents_are_safe(name):
    strat = build(name, embedder=FakeEmbedder())
    assert list(strat.split(Document(doc_id="e", text=""))) == []
    assert list(strat.split(Document(doc_id="w", text="   \n\n  "))) == []


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_single_short_sentence_yields_one_chunk(name):
    doc = Document(doc_id="s", text="Paris is the capital of France.", title="France")
    chunks = list(build(name, embedder=FakeEmbedder()).split(doc))
    assert len(chunks) == 1
    assert "Paris" in chunks[0].text


# --- strategy-specific behaviour ---------------------------------------------

def test_fixed_chunks_respect_the_word_budget():
    chunks = list(build("fixed", size=30, overlap=5).split(DOC))
    for c in chunks:
        assert len(c.text.split()) <= 32  # +slack for boundary whitespace


def test_recursive_respects_the_char_ceiling():
    chunks = list(RecursiveChunker(max_chars=200, min_chars=40).split(DOC))
    assert chunks
    # min_chars merging can push one span slightly over; allow a single ceiling.
    assert all(len(c.text) <= 400 for c in chunks)


def test_sentence_window_context_is_wider_than_the_indexed_text():
    chunks = list(build("sentence_window", group=1, window=3).split(DOC))
    assert chunks
    wider = [c for c in chunks if len(c.context_text) > len(c.text)]
    assert len(wider) >= len(chunks) - 2, "windows should expand context"


def test_metadata_prefix_enters_embed_text_but_never_cited_text():
    chunks = list(build("metadata").split(DOC))
    assert chunks
    assert all("Treaty of Versailles" in c.embed_text for c in chunks)
    # The prefix must not leak into what we show as a quotation.
    prefixed = [c for c in chunks if c.meta.get("prefix")]
    assert prefixed
    for c in prefixed:
        assert c.text in DOC.text


def test_semantic_finds_the_topic_shift():
    chunks = list(SemanticChunker(embedder=FakeEmbedder(), percentile=40).split(DOC))
    assert len(chunks) > 1, "a document with three distinct topics should split"


def test_semantic_without_embedder_degrades_to_one_chunk():
    chunks = list(SemanticChunker(embedder=None).split(DOC))
    assert len(chunks) == 1


def test_contextual_prepends_situating_text_only_to_the_embedding():
    chunks = list(ContextualChunker(base=RecursiveChunker()).split(DOC))
    assert chunks
    for c in chunks:
        assert c.meta["situating_context"]
        assert c.embed_text.endswith(c.text)
        assert c.text in DOC.text


def test_contextual_generated_mode_survives_a_failing_summarizer():
    def boom(_doc, _chunk):
        raise RuntimeError("provider down")

    chunks = list(
        ContextualChunker(base=RecursiveChunker(), mode="generated", summarize=boom).split(DOC)
    )
    assert chunks, "ingest must not abort when enrichment fails"
    assert all(c.embed_text.endswith(c.text) for c in chunks)


# --- the grid -----------------------------------------------------------------

def test_build_all_skips_embedder_strategies_when_none_supplied():
    grid = build_all(embedder=None)
    assert "semantic" not in grid
    assert "fixed" in grid and "recursive" in grid


def test_build_all_is_complete_with_an_embedder():
    grid = build_all(embedder=FakeEmbedder())
    assert set(grid) == set(STRATEGIES)
