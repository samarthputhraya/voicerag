"""The grounded-answer prompt, and the parser for what comes back.

Four constraints shape this prompt, and each one is a requirement rather than a
preference:

**Grounding.** The model may use only the numbered passages. Every factual
answer must carry an inline ``[n]`` citation, which turns "is this grounded?"
from a judgement call into a check the guardrail layer can run mechanically:
a citation that points at a passage which does not support the claim, or an
answer with no citation at all, is a detectable failure rather than a plausible
paragraph.

**Abstention.** The system must know when *not* to answer, so there is exactly
one abstention token, :data:`ABSTAIN_TOKEN`, and the prompt requires it verbatim
with nothing else. A single literal token is trivially detectable; "I'm sorry,
but the provided context doesn't seem to contain..." is not, and it also costs
twenty tokens of decode time to say nothing.

**Brevity is a latency constraint, not a style preference.** The 200 ms bar is
measured to the *last* output token, and each token costs about a millisecond.
A three-sentence answer is a failed run. The prompt therefore demands one short
sentence, and :data:`~voicerag.generate.base.DEFAULT_MAX_TOKENS` enforces a hard
ceiling underneath it. MS MARCO gold answers are themselves short factoids, so
terseness costs no measurable quality here.

**Prompt-cache alignment.** Groq caches on a byte-identical prefix and bills
cached input at roughly half price with a materially better TTFT. So
:data:`SYSTEM_PROMPT` is a module constant with no interpolation of any kind --
no timestamp, no query, no retrieved text, not even a chunk count -- and every
variable byte lives in the user turn that follows it. A single formatted
character in the system prompt would silently cost every request its cache hit,
which is exactly the kind of regression no test catches unless the invariant is
asserted; :func:`render` guarantees it structurally and the tests assert it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

__all__ = [
    "SYSTEM_PROMPT",
    "ABSTAIN_TOKEN",
    "ParsedAnswer",
    "PassageLike",
    "render",
    "format_passages",
    "usable_passages",
    "parse_answer",
    "MAX_CONTEXT_CHUNKS",
    "MAX_CHUNK_CHARS",
]

#: The one string that means "I will not answer this". Uppercase and
#: underscored so it cannot appear by accident in an English sentence.
ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"

#: Passages beyond this add prefill latency and dilute attention faster than
#: they add recall. Hybrid retrieval already puts the supporting passage in the
#: top few; the tail is mostly Bing hard negatives.
MAX_CONTEXT_CHUNKS = 5

#: Per-passage character cap. MS MARCO passages run ~400 characters, so this
#: rarely bites -- it exists to bound the worst case, since one pathological
#: chunk would otherwise blow the prefill budget for the whole request.
MAX_CHUNK_CHARS = 600

#: Byte-identical across every request. See the module docstring; do not
#: interpolate anything into this string, ever.
SYSTEM_PROMPT = """You are VoiceRAG, a voice question-answering assistant. Your answer is spoken aloud, so it must be short, plain and immediately useful.

You answer strictly from the numbered passages supplied in the user message.

Rules:
1. Use ONLY those passages. Never use prior knowledge and never infer beyond what is written.
2. Answer in ONE short sentence of at most 25 words. No preamble, no restating the question, no follow-up offers, no markdown.
3. Cite each passage you used inline as [n], for example: Paris [2].
4. If the passages do not contain the answer, reply with exactly INSUFFICIENT_CONTEXT and nothing else.
5. If the question is unsafe, nonsensical, or asks for anything the passages cannot support, reply with exactly INSUFFICIENT_CONTEXT and nothing else."""


@runtime_checkable
class PassageLike(Protocol):
    """Anything with retrievable text.

    Structural rather than a concrete import of
    :class:`~voicerag.chunking.base.Chunk`, so this module stays usable with
    plain strings, with database rows, and in tests -- and so the prompt layer
    never drags the chunking package into the import graph.
    """

    @property
    def text(self) -> str: ...


@dataclass(slots=True, frozen=True)
class ParsedAnswer:
    """A model answer decomposed into the parts the guardrails check.

    Args:
        text: The answer with any trailing prompt echo removed. Empty when the
            model abstained.
        citations: Passage numbers cited, in first-appearance order, 1-based.
        abstained: The model emitted :data:`ABSTAIN_TOKEN`.
        raw: The model output, untouched.
    """

    text: str
    citations: tuple[int, ...]
    abstained: bool
    raw: str

    def invalid_citations(self, n_passages: int) -> tuple[int, ...]:
        """Citations pointing outside the supplied passage list.

        A non-empty result is a hard hallucination signal: the model invented a
        source. Cheaper and far more reliable than any semantic check, because
        it is arithmetic.

        Args:
            n_passages: How many passages were actually supplied.

        Returns:
            The out-of-range citation numbers, in order.
        """
        return tuple(c for c in self.citations if c < 1 or c > n_passages)

    @property
    def is_grounded_shape(self) -> bool:
        """True if the answer has the *shape* of a grounded answer.

        Shape only -- it says the model abstained, or answered and cited
        something. Whether the cited passage actually supports the claim is the
        guardrail layer's job; this is the cheap structural precondition.
        """
        return self.abstained or bool(self.citations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": list(self.citations),
            "abstained": self.abstained,
        }


def _passage_text(passage: str | PassageLike) -> str:
    """Extract prompt text from a string or a chunk-like object.

    Prefers ``context_text`` when present: contextual chunking stores a
    retrieval-time enriched form there (document title, surrounding summary),
    and that enrichment is exactly what makes an isolated passage answerable.
    """
    if isinstance(passage, str):
        return passage
    for attr in ("context_text", "text"):
        value = getattr(passage, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _truncate(text: str, limit: int) -> str:
    """Clip to ``limit`` characters on a word boundary.

    Cutting mid-word produces a fragment the model may cite as if it were whole;
    an ellipsis makes the truncation visible to the model instead.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut or text[:limit]}..."


def usable_passages(
    passages: Sequence[str | PassageLike],
    *,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[tuple[int, str]]:
    """Select and clip the passages that will actually appear in the prompt.

    Returns ``(original_index, prompt_text)`` pairs rather than bare strings so
    a caller can map citation ``[n]`` back to the retrieved chunk it came from:
    the model cites the *n*-th line of the prompt, and blank or dropped
    passages mean that is not the *n*-th element of the input list. Handing the
    caller the index removes an off-by-one that would silently attribute an
    answer to the wrong source document.

    Args:
        passages: Retrieved passages, best first.
        max_chunks: Keep at most this many.
        max_chars: Per-passage character cap.

    Returns:
        Included passages in prompt order; ``result[n - 1]`` corresponds to
        citation ``[n]``.
    """
    out: list[tuple[int, str]] = []
    for i, passage in enumerate(passages[:max_chunks]):
        body = _truncate(_passage_text(passage), max_chars)
        if body:
            out.append((i, body))
    return out


def format_passages(
    passages: Sequence[str | PassageLike],
    *,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> str:
    """Render passages as a numbered block, 1-based to match the citation format.

    Numbering is contiguous over the passages actually included -- see
    :func:`usable_passages` for mapping a citation back to its source.

    Args:
        passages: Retrieved passages, best first.
        max_chunks: Keep at most this many.
        max_chars: Per-passage character cap.

    Returns:
        One ``[n] text`` line per passage, or ``"(none)"`` when there are no
        usable passages. That literal matters: it leaves the model with nothing
        to cite, so rule 4 deterministically forces an abstention instead of
        inviting a fluent guess from parametric memory.
    """
    lines = [
        f"[{n}] {body}"
        for n, (_, body) in enumerate(
            usable_passages(passages, max_chunks=max_chunks, max_chars=max_chars),
            start=1,
        )
    ]
    return "\n".join(lines) if lines else "(none)"


def render(
    question: str,
    chunks: Sequence[str | PassageLike],
    *,
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> tuple[str, str]:
    """Build the ``(system, user)`` pair for one grounded answer.

    Args:
        question: The user's question, as transcribed.
        chunks: Retrieved passages, best first. Strings or chunk-like objects.
        max_chunks: Passage cap; see :data:`MAX_CONTEXT_CHUNKS`.
        max_chars: Per-passage character cap.

    Returns:
        ``(system, user)``. The system string is always :data:`SYSTEM_PROMPT`
        itself -- returned by identity, not rebuilt -- so the provider's prompt
        cache sees the same prefix bytes on every request.
    """
    body = format_passages(chunks, max_chunks=max_chunks, max_chars=max_chars)
    user = f"PASSAGES:\n{body}\n\nQUESTION: {' '.join(question.split())}\nANSWER:"
    return SYSTEM_PROMPT, user


_CITATION = re.compile(r"\[(\d{1,2})\]")
_ANSWER_ECHO = re.compile(r"^\s*ANSWER\s*:\s*", re.IGNORECASE)


def parse_answer(raw: str) -> ParsedAnswer:
    """Decompose a model answer into text, citations and the abstention flag.

    Abstention is detected on a *substring* rather than an exact match. A model
    that emits ``INSUFFICIENT_CONTEXT.`` or prefixes a stray space has still
    refused to answer, and treating that as a normal answer -- an uncited one,
    which the guardrails would then reject as ungrounded -- would report the
    right outcome for entirely the wrong reason and corrupt the abstention
    metrics.

    Args:
        raw: The model's full output.

    Returns:
        A :class:`ParsedAnswer`.
    """
    text = _ANSWER_ECHO.sub("", raw or "").strip()
    if ABSTAIN_TOKEN in text.upper():
        return ParsedAnswer(text="", citations=(), abstained=True, raw=raw or "")

    seen: list[int] = []
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return ParsedAnswer(
        text=text, citations=tuple(seen), abstained=False, raw=raw or ""
    )
