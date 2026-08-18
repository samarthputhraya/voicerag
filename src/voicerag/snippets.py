"""Turn a retrieved chunk into something worth showing a human.

MS MARCO passages are scraped web pages. Alongside the sentence that actually
answers the question, a chunk routinely carries the page's navigation furniture:
an A-Z index strip (``a b c d e f g h i j k l m n o p q r s t u v w x y z``), a
bare source URL, or the "related questions" rail rendered as one dash-run of
question fragments. A full-store scan of the served index found 136 chunks with
an alphabet run, 419 beginning with a URL, and 8,550 containing one.

Those chunks are not retrieval errors. They rank because the *other* half of the
passage genuinely matches -- so filtering them out of retrieval would throw away
correct evidence. The junk is a **presentation** problem, and this is the
presentation fix.

Why this is safe to apply to the citation text and nowhere else:
:meth:`RagPipeline._contexts` returns two parallel lists, ``(chunks,
prompt_texts)``. The model is prompted with ``prompt_texts`` and grounding
verifies against ``prompt_texts``; only the ``Citation.text`` the UI renders is
taken from ``chunk.text``. Cleaning here therefore cannot move a retrieval
score, change what the model reads, or alter a grounding verdict -- it changes
exactly what a person sees.

The cleaner is deliberately conservative. It removes only patterns that carry no
information for a reader, and if cleaning would leave too little to be useful it
returns the original text instead. A citation that looks scruffy is a much
smaller failure than a citation that has been emptied of its evidence.
"""

from __future__ import annotations

import re

__all__ = ["clean_snippet", "looks_like_junk"]

#: Bare URLs. They are never the evidence -- the passage text around them is.
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)

#: Six or more isolated single letters in a row. Real prose does not do this;
#: an A-Z index strip does. Six is comfortably above the longest run initials
#: produce ("J. R. R. Tolkien" is three) and below a full alphabet.
_ALPHABET_RUN = re.compile(r"(?:\b[a-z]\b[\s.,|·–—-]*){6,}", re.IGNORECASE)

#: A "people also asked" rail flattened into one line, where each question is
#: introduced by a dash: ``-How much is excreted?-What enzymes break it down?``
#: The dash is the separator, so it becomes a space rather than being deleted.
_QUESTION_RAIL = re.compile(r"\s*[-–—]\s*(?=[A-Z][^?]{5,}\?)")

#: Runs of the punctuation that survives stripping the above.
_ORPHAN_PUNCT = re.compile(r"\s+([,.;:!?])")
_REPEAT_PUNCT = re.compile(r"([.,;:|·–—-])\1{2,}")
_WS = re.compile(r"\s+")

#: Below this many "real" words a cleaned snippet is not worth showing on its
#: own merits. Used only by :func:`looks_like_junk`, never to blank a citation
#: the answer actually cited. Tuned down from 8 after it rejected "The lethal
#: dose of caffeine is 2 grams per day." -- a short sentence that fully answers
#: its question. Six keeps the nav-rail leftovers out and lets real prose past.
_MIN_CONTENT_WORDS = 6


def clean_snippet(text: str, *, max_chars: int | None = None) -> str:
    """Strip scraped furniture from a passage, without touching its meaning.

    Args:
        text: The raw chunk text.
        max_chars: Optional clip length. The clip lands on a word boundary and
            appends a single-character ellipsis. ``None`` leaves the length
            alone, which is the right default for an API response -- clipping
            evidence is a display decision and belongs to the display.

    Returns:
        The cleaned text, or the whitespace-normalised original when cleaning
        would leave less than a quarter of what it started with. That guard is
        what stops an over-eager pattern from silently emptying a citation.
    """
    if not text:
        return ""

    cleaned = _QUESTION_RAIL.sub(" ", text)
    cleaned = _URL.sub(" ", cleaned)
    cleaned = _ALPHABET_RUN.sub(" ", cleaned)
    cleaned = _REPEAT_PUNCT.sub(r"\1", cleaned)
    cleaned = _WS.sub(" ", cleaned).strip()
    cleaned = _ORPHAN_PUNCT.sub(r"\1", cleaned)
    # Stripping a leading URL or index strip usually orphans its punctuation.
    cleaned = cleaned.lstrip(" .,;:|-–—")

    original = _WS.sub(" ", text).strip()
    # Never trade real evidence for tidiness.
    if len(cleaned) < len(original) // 4:
        cleaned = original

    if max_chars is not None and len(cleaned) > max_chars:
        cut = cleaned.rfind(" ", 0, max_chars)
        cleaned = cleaned[: cut if cut > max_chars * 2 // 3 else max_chars].rstrip(" .,;:")
        cleaned += "\u2026"

    return cleaned


def looks_like_junk(text: str) -> bool:
    """True when a *cleaned* snippet has too little prose to be worth showing.

    Only ever applied to the uncited fallback list, where the panel is showing
    retrieved context rather than evidence the answer relied on. A passage the
    answer cited is always shown, however scruffy, because its ``[n]`` marker
    has to point at something.
    """
    words = [w for w in text.split() if len(w) > 2]
    return len(words) < _MIN_CONTENT_WORDS
