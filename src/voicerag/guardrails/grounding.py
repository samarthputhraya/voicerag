"""Post-generation grounding: does the answer actually say what the passages say?

Abstention decides whether to speak. Grounding checks what was said. They catch
different failures, and a system with only one of them is broken in a way the
other cannot see: retrieval can surface a perfect passage and the model can
still invent a number from it.

**Two layers, and it matters which is which.**

*Lexical (always on, sub-millisecond).* Token overlap plus longest common
subsequence between each answer sentence and each retrieved passage, an exact
check on every number and date, and verification of every inline citation
index. This layer runs on every request. It is not a semantic entailment check
and does not claim to be -- it catches the failures that actually dominate in
extractive short-answer RAG: a fabricated figure, a citation pointing at a
passage that says something else, or a sentence assembled from words that appear
nowhere in the context.

*Model (optional, ~15-30 ms).* A natural language inference model --
``vectara/hallucination_evaluation_model`` (HHEM-2.1-Open, a flan-t5-base
encoder, measured at 118 ms for 256 tokens on this box) or
``cross-encoder/nli-deberta-v3-xsmall`` (measured at 17 ms for 192 tokens) --
scores real entailment and catches paraphrase and negation, which lexical
overlap provably cannot. It is off by default and injected through
:class:`NLIScorer`; nothing here imports torch.

**Why numbers get their own check.** Numerals are where short-answer models
hallucinate most, and they are the cheapest thing in the world to verify: a date
or a quantity either appears in the retrieved text or it does not. Overlap
scoring dilutes them -- "the population was 4.7 million" shares seven of eight
tokens with a passage that says 3.2 million -- so an exact set membership test on
extracted numerals runs separately and can fail a claim on its own.

**What the lexical layer cannot do, stated plainly.** It cannot check polarity.
A bare ``"Yes."`` contains no content token, so it is marked ``trivial`` and
passes -- there is nothing lexical to compare against a passage, and pretending
otherwise would mean failing every correct yes/no answer. Negation and
paraphrase are the same class of problem. All three are precisely what the
optional NLI layer exists for, and a deployment that serves yes/no questions
should enable it rather than trust the overlap score.

**Where this fits in the latency budget.** The lexical layer costs under a
millisecond on a five-passage context and runs after the last token, so it is
inside the 200 ms. The NLI layer must not be: :meth:`GroundingChecker.averify`
offloads it to a thread so it overlaps generation, and
:meth:`GroundingChecker.verify_claim` is public so a streaming caller can check
each sentence the moment it completes rather than waiting for the full answer.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..chunking.base import sentences

__all__ = [
    "ClaimVerdict",
    "GroundingVerdict",
    "GroundingConfig",
    "GroundingChecker",
    "NLIScorer",
    "GroundingBackendUnavailable",
    "load_nli",
    "extract_numbers",
]


class GroundingBackendUnavailable(RuntimeError):
    """Raised when an optional NLI backend cannot be constructed.

    Mirrors ``EmbeddingBackendUnavailable``: optional accelerators announce
    their absence loudly at construction time and never at request time.
    """


@runtime_checkable
class NLIScorer(Protocol):
    """Anything that can score entailment of a hypothesis by a premise."""

    def entails(self, premise: str, hypothesis: str) -> float:
        """Return ``P(premise entails hypothesis)`` in ``[0, 1]``."""
        ...


def load_nli(spec: str) -> NLIScorer:
    """Construct an NLI scorer from a spec string.

    Args:
        spec: ``"onnx:<path>"`` for an exported cross-encoder, or
            ``"hf:<model_id>"`` for a transformers checkpoint.

    Returns:
        An :class:`NLIScorer`.

    Raises:
        GroundingBackendUnavailable: Always, in this build. The seam is real and
            typed, but no NLI weights ship with the repo and no network is
            available at build time, so pretending otherwise would put a
            failing import on the serving path. Supply your own object to
            :class:`GroundingChecker` -- the protocol is one method.
    """
    raise GroundingBackendUnavailable(
        f"no NLI backend is bundled for spec {spec!r}; pass any object with "
        "entails(premise, hypothesis) -> float to GroundingChecker(nli=...)"
    )


# --- tokenisation -------------------------------------------------------------

#: Unicode-aware on purpose. An ASCII-only class (``[a-z0-9]+``) tokenises a
#: Devanagari, Bengali or Tamil sentence to the **empty list**, which does not
#: merely weaken this check -- it inverts it. Zero claim tokens take the
#: "nothing to verify" branch, and an entirely fabricated non-Latin answer is
#: certified ``grounded=True, score=1.0``, taking the citation-validity check
#: down with it. The corpus is ai4bharat's, so non-Latin text reaching this
#: function is the expected case, not an exotic one.
#:
#: ``\w`` under ``re.UNICODE`` (Python 3 default) also admits underscore, which
#: is excluded so identifiers do not read as words.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
#: Currency/percent aware numeral. Handles ``1,234.5``, ``$40``, ``12%``, years.
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_MONTHS = frozenset(
    "january february march april may june july august september october "
    "november december jan feb mar apr jun jul aug sep sept oct nov dec".split()
)

#: Function words carry no grounding evidence -- every English sentence shares
#: them, so counting them inflates overlap towards a constant and destroys the
#: dynamic range of the score.
_STOP = frozenset(
    """a an the of in on at to for from by with and or but is are was were be been
    being as that this these those it its he she they we you i him her them us my
    your their our his hers what which who whom whose when where why how than then
    there here not no nor do does did done have has had having will would shall
    should can could may might must about into over under after before between
    during also more most such some any each other into per via
    yes yeah ok okay correct""".split()
)

#: Number words that a generator may substitute for a numeral. Small on purpose:
#: full spelled-number parsing is a rabbit hole, and MS MARCO answers are
#: overwhelmingly digits. Kept so "two" in the answer matches "2" in the passage.
_NUM_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90", "hundred": "100", "thousand": "1000", "million": "1000000",
    "billion": "1000000000",
}


def _stemmer() -> Any:
    """The Snowball English stemmer, or ``None`` when PyStemmer is absent.

    Cached on the function object: constructing a Stemmer is cheap but not
    free, and this runs per claim per passage inside the request path.
    """
    cached = getattr(_stemmer, "_cache", "unset")
    if cached != "unset":
        return cached
    try:  # pragma: no cover - exercised by whichever branch the env provides
        import Stemmer  # type: ignore[import-untyped]

        cached = Stemmer.Stemmer("english")
    except Exception:
        cached = None
    _stemmer._cache = cached  # type: ignore[attr-defined]
    return cached


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens, stopped and stemmed.

    Stemmed **because the retriever stems**. The served BM25 index is built
    with ``{"stemmer": "english"}``, so a passage is retrieved on the strength
    of ``anticonvulsants`` matching a question about an ``anticonvulsant`` --
    and grounding then compared the raw surface forms and called the resulting
    answer unsupported. Measured: "Gabapentin is an anticonvulsant [1]" against
    a passage reading "in a class of medications called anticonvulsants" scored
    0.50 against a 0.55 bar and was refused, on a single inflectional ``s``.

    Grounding was therefore stricter than the retrieval that found the
    evidence, which is not a defensible place for the two to disagree. Using
    the same Snowball stemmer the index uses makes the two consistent. It is
    applied symmetrically to claim and context, so it can only merge
    morphological variants of the same word -- it cannot invent support between
    unrelated ones.

    Degrades to unstemmed tokens if PyStemmer is unavailable, since it arrives
    transitively via ``bm25s`` rather than as a declared dependency.
    """
    raw = [t for t in _WORD.findall(text.lower()) if t not in _STOP]
    stem = _stemmer()
    if stem is None:
        return raw
    return list(stem.stemWords(raw))


def _normalise_number(raw: str) -> str:
    """Canonical form of a numeral so ``1,200``, ``1200`` and ``1200.0`` agree.

    Percent signs are dropped: a claim of "40%" is grounded by a passage saying
    "40 percent", and keeping the sign would make the check fail on formatting
    rather than on facts.
    """
    s = raw.replace(",", "").rstrip("%").lstrip("+")
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else repr(f)


def extract_numbers(text: str) -> set[str]:
    """All numeric and month tokens in ``text``, canonicalised.

    Args:
        text: Any prose.

    Returns:
        A set of normalised numerals plus lowercased month names. Spelled-out
        numbers are mapped to digits so the answer and the passage can disagree
        on style without disagreeing on fact.

    Example:
        >>> sorted(extract_numbers("about 1,200 in March 2019"))
        ['1200', '2019', 'march']
    """
    out = {_normalise_number(m.group(0)) for m in _NUMBER.finditer(text)}
    for word in _WORD.findall(text.lower()):
        if word in _MONTHS:
            out.add(word)
        elif word in _NUM_WORDS:
            out.add(_NUM_WORDS[word])
    return out


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length, O(len(a) * len(b)) time, O(len(b)) space.

    Bounded by the caller to keep this genuinely sub-millisecond; see
    :attr:`GroundingConfig.max_context_tokens`.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


# --- results ------------------------------------------------------------------


@dataclass(slots=True)
class ClaimVerdict:
    """Grounding outcome for one sentence of the answer.

    Attributes:
        text: The sentence, verbatim, so it can be quoted back to the user.
        supported: Final decision after every layer that ran.
        score: Lexical support in ``[0, 1]`` against the best-matching passage.
        coverage: Fraction of the claim's content tokens present in that passage.
        lcs: Longest-common-subsequence ratio -- word *order* evidence, which
            separates a real quote from a bag of words that happens to overlap.
        best_context: Index of the best-matching passage, or ``None`` if there
            were no passages.
        best_chunk_id: Id of that passage when the caller supplied objects with
            ids, for citation display.
        citations: Citation indices parsed from the sentence (1-based, as
            written).
        invalid_citations: Cited indices that do not exist in the context list.
        unsupported_citations: Cited indices that exist but whose passage does
            not support this sentence.
        unsupported_numbers: Numerals and dates in the claim that appear in no
            retrieved passage. Any entry here fails the claim outright.
        method: Which layer decided -- ``"lexical"``, ``"nli"`` or
            ``"trivial"`` (a sentence with no content tokens, e.g. "Yes.").
        reason: Why it failed, empty when supported.
    """

    text: str
    supported: bool
    score: float
    coverage: float = 0.0
    lcs: float = 0.0
    best_context: int | None = None
    best_chunk_id: str | None = None
    citations: tuple[int, ...] = ()
    invalid_citations: tuple[int, ...] = ()
    unsupported_citations: tuple[int, ...] = ()
    unsupported_numbers: tuple[str, ...] = ()
    method: str = "lexical"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "supported": self.supported,
            "score": round(self.score, 4),
            "coverage": round(self.coverage, 4),
            "lcs": round(self.lcs, 4),
            "best_context": self.best_context,
            "best_chunk_id": self.best_chunk_id,
            "citations": list(self.citations),
            "invalid_citations": list(self.invalid_citations),
            "unsupported_citations": list(self.unsupported_citations),
            "unsupported_numbers": list(self.unsupported_numbers),
            "method": self.method,
            "reason": self.reason,
        }


@dataclass(slots=True)
class GroundingVerdict:
    """Grounding outcome for a whole answer.

    Attributes:
        grounded: ``True`` if the answer may be shown as-is.
        per_claim: One :class:`ClaimVerdict` per sentence, in order.
        unsupported_claims: The sentences that failed, verbatim. This is the
            field the UI renders, so it holds sentences and not indices.
        score: Length-weighted mean claim score in ``[0, 1]``. Weighting by
            content-token count stops a trailing "That's it." from dominating a
            substantive sentence.
        latency_ms: Cost of the check.
        used_nli: Whether the model layer actually ran.
    """

    grounded: bool
    per_claim: list[ClaimVerdict] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    score: float = 0.0
    latency_ms: float = 0.0
    used_nli: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "score": round(self.score, 4),
            "unsupported_claims": list(self.unsupported_claims),
            "per_claim": [c.to_dict() for c in self.per_claim],
            "latency_ms": round(self.latency_ms, 4),
            "used_nli": self.used_nli,
        }


# --- configuration ------------------------------------------------------------


@dataclass(slots=True)
class GroundingConfig:
    """Thresholds for the grounding stage.

    Attributes:
        claim_threshold: Lexical support at or above which a claim passes.
            0.55 with the default 0.75/0.25 coverage/LCS mix means roughly
            "two thirds of the content words appear in one passage, in a
            recognisable order".
        answer_threshold: Length-weighted mean at or above which the *answer* is
            considered grounded, given no hard failure. Set below
            ``claim_threshold`` on purpose: one weak sentence in an otherwise
            well-supported answer is a style problem, not a hallucination.
        coverage_weight: Share of the claim score from token coverage; the
            remainder comes from the LCS ratio.
        check_numbers: Enforce exact numeral/date presence. A fabricated number
            fails its claim regardless of how well the words overlap.
        check_citations: Validate inline ``[n]`` markers.
        citation_support_threshold: Coverage a *cited* passage must reach for
            the citation to count as honest. Lower than ``claim_threshold``
            because a citation is a pointer, not a quotation.
        max_context_tokens: Passages are truncated to this many tokens for the
            LCS pass, bounding it at ``claim_tokens * 400`` operations.
        require_all_claims: If ``True``, any unsupported claim makes the whole
            answer ungrounded. Off by default; the answer-level threshold plus
            hard failures is the shipped policy.
        min_claim_tokens: Distinct content tokens a claim needs before lexical
            overlap is allowed to *support* it. Below this the measure is not
            evidence: with one token, coverage can only be 0 or 1, and 1 means
            merely that the word occurs somewhere in some passage -- not that
            the passage asserts the claim. Measured failure this exists to stop:
            asked "what is the capital of India", the model answered "Mumbai."
            against five passages that never say Mumbai is the capital (one
            mentions the city in passing). Coverage was 1/1, the answer scored
            **1.0**, and a fabrication was served as fully grounded. Claims below
            the bar are marked ``unverifiable`` and cannot pass; two tokens is
            the smallest value that makes coverage a ratio rather than a
            presence test.
        nli_threshold: Entailment probability at which the model layer rescues a
            claim the lexical layer failed.
    """

    claim_threshold: float = 0.55
    answer_threshold: float = 0.5
    min_claim_tokens: int = 2
    coverage_weight: float = 0.75
    check_numbers: bool = True
    check_citations: bool = True
    citation_support_threshold: float = 0.35
    max_context_tokens: int = 400
    require_all_claims: bool = False
    nli_threshold: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_threshold": self.claim_threshold,
            "answer_threshold": self.answer_threshold,
            "coverage_weight": self.coverage_weight,
            "check_numbers": self.check_numbers,
            "check_citations": self.check_citations,
            "citation_support_threshold": self.citation_support_threshold,
            "max_context_tokens": self.max_context_tokens,
            "require_all_claims": self.require_all_claims,
            "nli_threshold": self.nli_threshold,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GroundingConfig":
        known = {f for f in cls.__slots__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _context_text(ctx: Any) -> str:
    """Accept a plain string, a ``Chunk``-like object or a mapping."""
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, Mapping):
        return str(ctx.get("text") or ctx.get("context_text") or "")
    return str(getattr(ctx, "text", "") or "")


def _context_id(ctx: Any) -> str | None:
    if isinstance(ctx, str):
        return None
    if isinstance(ctx, Mapping):
        cid = ctx.get("chunk_id")
        return None if cid is None else str(cid)
    cid = getattr(ctx, "chunk_id", None)
    return None if cid is None else str(cid)


# --- the checker --------------------------------------------------------------


class GroundingChecker:
    """Verifies a generated answer against the passages it was generated from.

    Args:
        config: Thresholds.
        nli: Optional entailment model. When supplied it is consulted **only**
            for claims the lexical layer rejected, which keeps its cost
            proportional to the number of suspicious sentences rather than to
            the length of the answer.

    Example:
        >>> ctx = ["The Eiffel Tower is 330 metres tall."]
        >>> GroundingChecker().verify("The Eiffel Tower is 330 metres tall.", ctx).grounded
        True
        >>> GroundingChecker().verify("The Eiffel Tower is 450 metres tall.", ctx).grounded
        False
    """

    __slots__ = ("config", "nli")

    def __init__(
        self, config: GroundingConfig | None = None, *, nli: NLIScorer | None = None
    ) -> None:
        self.config = config or GroundingConfig()
        self.nli = nli

    def __repr__(self) -> str:
        return f"GroundingChecker(nli={'yes' if self.nli else 'no'})"

    # -- preparation ----------------------------------------------------------

    def _prepare(self, contexts: Sequence[Any]) -> tuple[list[list[str]], set[str], list[str | None]]:
        """Tokenise the context once, not once per claim.

        Returns:
            ``(token lists, union of all numerals, chunk ids)``.
        """
        cap = self.config.max_context_tokens
        texts = [_context_text(c) for c in contexts]
        toks = [_tokens(t)[:cap] for t in texts]
        numbers: set[str] = set()
        for t in texts:
            numbers |= extract_numbers(t)
        return toks, numbers, [_context_id(c) for c in contexts]

    @staticmethod
    def split_claims(answer: str) -> list[str]:
        """Split an answer into checkable claims.

        Uses the same sentence splitter as the chunking layer, so a claim and
        the passage it is checked against were segmented by identical rules --
        one fewer source of spurious mismatch. Sentences with no content token
        (``"Yes."``) survive here and are marked ``trivial`` downstream rather
        than dropped, because dropping them would silently shrink the
        denominator of the grounding score.

        A fragment that is *only* citation markers is folded back into the
        sentence before it. The splitter ends a sentence at the full stop in
        ``"Mumbai. [2]"``, which stranded ``"[2]"`` as a claim of its own: it
        has no content tokens, so it scored a ``trivial`` 1.0 and pulled the
        answer-level mean up, while the citation it carries was detached from
        the sentence it actually belongs to -- so the cited-passage check ran
        against the wrong claim. Merging fixes both, and cannot merge across a
        real sentence because a fragment with any content token is left alone.

        Args:
            answer: Generated answer text.

        Returns:
            Non-empty, stripped sentences in order.
        """
        out: list[str] = []
        for _, _, raw in sentences(answer or ""):
            s = raw.strip()
            if not s:
                continue
            if out and not _CITATION.sub(" ", s).strip():
                out[-1] = f"{out[-1]} {s}"
                continue
            out.append(s)
        return out

    # -- one claim ------------------------------------------------------------

    def verify_claim(
        self,
        claim: str,
        contexts: Sequence[Any],
        *,
        prepared: tuple[list[list[str]], set[str], list[str | None]] | None = None,
    ) -> ClaimVerdict:
        """Check a single sentence. Public so a streaming caller can check
        sentences as they complete instead of waiting for the final token.

        Args:
            claim: One answer sentence.
            contexts: Retrieved passages -- strings, ``Chunk``-like objects or
                mappings with a ``text`` key.
            prepared: Output of the internal context preparation, passed in by
                :meth:`verify` so a multi-sentence answer tokenises the context
                once. Computed on demand when omitted.

        Returns:
            A :class:`ClaimVerdict`. Hard failures (a fabricated number, a
            citation index that does not exist) reject the claim regardless of
            lexical score; a claim can only *pass* by clearing the threshold.
        """
        cfg = self.config
        ctx_toks, ctx_numbers, ctx_ids = prepared or self._prepare(contexts)

        # Citation markers are addressing metadata, not content: leaving "[2]"
        # in inflates neither overlap nor numeral checks if we strip it first.
        cites = _parse_citations(claim)
        body = _CITATION.sub(" ", claim)
        claim_toks = _tokens(body)

        if not claim_toks:
            return ClaimVerdict(
                text=claim, supported=True, score=1.0, coverage=1.0, lcs=1.0,
                citations=cites, method="trivial",
            )

        # One content token is a presence test, not evidence. Coverage can only
        # come out 0 or 1, and a 1 says the word occurs in some passage -- which
        # is exactly how "Mumbai." was scored 1.0 against passages that never
        # claim Mumbai is the capital of India. Refuse to certify what the
        # measure cannot distinguish; the NLI layer may still rescue it, and a
        # model that writes whole sentences (as SYSTEM_PROMPT instructs) never
        # lands here.
        if len(set(claim_toks)) < cfg.min_claim_tokens:
            return ClaimVerdict(
                text=claim, supported=False, score=0.0, coverage=0.0, lcs=0.0,
                citations=cites, method="unverifiable",
                reason=(
                    f"{len(set(claim_toks))} content token(s) is too few to verify "
                    f"by lexical overlap"
                ),
            )

        claim_set = set(claim_toks)
        best_i, best_score, best_cov, best_lcs = None, 0.0, 0.0, 0.0
        per_ctx_cov: list[float] = []
        for i, toks in enumerate(ctx_toks):
            if not toks:
                per_ctx_cov.append(0.0)
                continue
            ctx_set = set(toks)
            cov = len(claim_set & ctx_set) / len(claim_set)
            per_ctx_cov.append(cov)
            lcs = _lcs_length(claim_toks, toks) / len(claim_toks)
            score = cfg.coverage_weight * cov + (1.0 - cfg.coverage_weight) * lcs
            if score > best_score or best_i is None:
                best_i, best_score, best_cov, best_lcs = i, score, cov, lcs

        # A claim citing several passages asserts that they support it
        # TOGETHER -- which is precisely what SYSTEM_PROMPT rule 2 instructs
        # ("You MAY combine facts stated across passages"). Scoring it only
        # against the best SINGLE passage measured something the model was
        # never asked to do: a correctly synthesised sentence drawing half its
        # content from each of two passages scores ~0.50 against a 0.55 bar and
        # is refused, with both halves individually "unsupported".
        #
        # So the union of the cited passages becomes an additional scoring
        # candidate. Note what this does NOT do: it does not lower
        # claim_threshold, and it does not apply to uncited or single-citation
        # claims. It fixes what the threshold is measured against, which is the
        # actual defect -- a weakened bar would have let fabrication through.
        cited_idx = [c - 1 for c in dict.fromkeys(cites) if 1 <= c <= len(ctx_toks)]
        if len(cited_idx) > 1:
            union_toks: list[str] = []
            for i in cited_idx:
                union_toks.extend(ctx_toks[i])
            if union_toks:
                cov = len(claim_set & set(union_toks)) / len(claim_set)
                lcs = _lcs_length(claim_toks, union_toks) / len(claim_toks)
                score = cfg.coverage_weight * cov + (1.0 - cfg.coverage_weight) * lcs
                if score > best_score:
                    # best_i deliberately keeps pointing at the best real
                    # passage: it is reported as best_context/best_chunk_id and
                    # must stay a citable index, not a synthetic union.
                    best_score, best_cov, best_lcs = score, cov, lcs

        reasons: list[str] = []
        bad_numbers: tuple[str, ...] = ()
        if cfg.check_numbers:
            missing = sorted(extract_numbers(body) - ctx_numbers)
            if missing:
                bad_numbers = tuple(missing)
                reasons.append(
                    "states " + ", ".join(missing) + ", which appears in no retrieved passage"
                )

        invalid: tuple[int, ...] = ()
        unsupported_cites: tuple[int, ...] = ()
        if cfg.check_citations and cites:
            invalid = tuple(c for c in cites if not 1 <= c <= len(ctx_toks))
            if invalid:
                reasons.append(
                    "cites passage " + ", ".join(f"[{c}]" for c in invalid)
                    + f" but only {len(ctx_toks)} passage(s) were retrieved"
                )
            valid = [c for c in cites if c not in invalid]
            if len(valid) > 1:
                # A claim citing several passages is asserting that they support
                # it *together*, which is exactly what SYSTEM_PROMPT rule 2 asks
                # the model to do ("You MAY combine facts stated across
                # passages"). Testing each cited passage on its own therefore
                # vetoed the behaviour we instruct: a sentence synthesised from
                # two passages fails both halves, because neither alone carries
                # 35% of the combined sentence's content words.
                #
                # Measured on reports/guardrails_e2e.json: this veto produced 6
                # of the 9 refusals on gold-answerable questions, three of them
                # at grounding scores of 0.92, 0.64 and 0.58 -- above every
                # threshold, refused on this check alone. That is the "I drafted
                # an answer but 1 statement(s) weren't supported" the demo was
                # showing on questions the corpus answers.
                #
                # The check is not weakened, only asked the right question:
                # cited passages must *collectively* contain the claim. A
                # single-citation claim is still tested against that one
                # passage, and the number and validity checks are untouched.
                union: set[str] = set()
                for c in valid:
                    union |= set(ctx_toks[c - 1])
                union_cov = len(claim_set & union) / len(claim_set)
                if union_cov < cfg.citation_support_threshold:
                    unsupported_cites = tuple(valid)
                    reasons.append(
                        "cites " + ", ".join(f"[{c}]" for c in valid)
                        + ", which together do not contain this statement"
                    )
            else:
                unsupported_cites = tuple(
                    c for c in valid
                    if per_ctx_cov[c - 1] < cfg.citation_support_threshold
                )
                if unsupported_cites:
                    reasons.append(
                        "cites " + ", ".join(f"[{c}]" for c in unsupported_cites)
                        + ", which does not contain this statement"
                    )

        hard_fail = bool(bad_numbers or invalid or unsupported_cites)
        supported = best_score >= cfg.claim_threshold and not hard_fail
        if not supported and not reasons:
            # Phrased as a verb phrase, like every other reason above, because
            # the refusal text in guardrails.policy joins these onto "it ...".
            # As a standalone clause this produced "it only 0% of its content
            # words appear in any retrieved passage", which is ungrammatical and
            # was being shown to users.
            reasons.append(
                f"appears in no retrieved passage "
                f"(only {best_cov:.0%} of its content words matched)"
            )

        return ClaimVerdict(
            text=claim,
            supported=supported,
            score=best_score,
            coverage=best_cov,
            lcs=best_lcs,
            best_context=best_i,
            best_chunk_id=ctx_ids[best_i] if best_i is not None and ctx_ids else None,
            citations=cites,
            invalid_citations=invalid,
            unsupported_citations=unsupported_cites,
            unsupported_numbers=bad_numbers,
            method="lexical",
            reason="" if supported else "; ".join(reasons),
        )

    # -- whole answer ---------------------------------------------------------

    def verify(self, answer: str, contexts: Sequence[Any]) -> GroundingVerdict:
        """Check a complete answer. Lexical layer only unless ``nli`` was given.

        Args:
            answer: The generated answer, citation markers included.
            contexts: The passages it was generated from, in the order the
                citation indices refer to (``[1]`` is ``contexts[0]``).

        Returns:
            A :class:`GroundingVerdict`. An empty answer is *not* grounded --
            silence with a citation UI around it is still a failure to answer.
            An answer with no context is likewise ungrounded, whatever it says.
        """
        t0 = time.perf_counter_ns()
        claims = self.split_claims(answer)
        if not claims:
            return GroundingVerdict(
                grounded=False, score=0.0,
                latency_ms=(time.perf_counter_ns() - t0) / 1e6,
            )

        prepared = self._prepare(contexts)
        verdicts = [self.verify_claim(c, contexts, prepared=prepared) for c in claims]
        used_nli = False
        if self.nli is not None:
            texts = [_context_text(c) for c in contexts]
            used_nli = self._rescue_with_nli(verdicts, texts)

        return self._finalise(verdicts, used_nli, t0)

    async def averify(self, answer: str, contexts: Sequence[Any]) -> GroundingVerdict:
        """:meth:`verify` with the model layer moved off the event loop.

        The lexical layer runs inline -- it is cheaper than a thread hop. Only
        the NLI rescue is offloaded, so an API handler can await this while the
        next request is being served, and a caller can start it on a partial
        answer concurrently with generation of the rest.

        Args: See :meth:`verify`.

        Returns:
            A :class:`GroundingVerdict`.
        """
        t0 = time.perf_counter_ns()
        claims = self.split_claims(answer)
        if not claims:
            return GroundingVerdict(
                grounded=False, score=0.0,
                latency_ms=(time.perf_counter_ns() - t0) / 1e6,
            )
        prepared = self._prepare(contexts)
        verdicts = [self.verify_claim(c, contexts, prepared=prepared) for c in claims]
        used_nli = False
        if self.nli is not None:
            texts = [_context_text(c) for c in contexts]
            used_nli = await asyncio.get_running_loop().run_in_executor(
                None, self._rescue_with_nli, verdicts, texts
            )
        return self._finalise(verdicts, used_nli, t0)

    # -- internals ------------------------------------------------------------

    def _rescue_with_nli(self, verdicts: list[ClaimVerdict], texts: Sequence[str]) -> bool:
        """Re-check lexically-failed claims with the entailment model, in place.

        A claim rejected for a fabricated number is *not* rescued: entailment
        models are weak at arithmetic and numeric contradiction, and an exact
        mismatch is already conclusive. Only overlap failures -- the paraphrase
        case NLI exists to fix -- are reconsidered.

        Returns:
            ``True`` if the model was consulted at least once.
        """
        assert self.nli is not None
        consulted = False
        for v in verdicts:
            if v.supported or v.unsupported_numbers or v.invalid_citations:
                continue
            consulted = True
            best = 0.0
            for premise in texts:
                best = max(best, float(self.nli.entails(premise, v.text)))
                if best >= self.config.nli_threshold:
                    break
            if best >= self.config.nli_threshold:
                v.supported = True
                v.method = "nli"
                v.score = max(v.score, best)
                v.reason = ""
        return consulted

    def _finalise(
        self, verdicts: list[ClaimVerdict], used_nli: bool, t0: int
    ) -> GroundingVerdict:
        """Aggregate claim verdicts into an answer-level decision."""
        cfg = self.config
        # A contentless sentence ("Yes.") is weighted 0 rather than 1. It asserts
        # nothing, so it should neither help nor hurt -- but at weight 1 it
        # carried a score of 1.0 into the mean and could lift an otherwise weak
        # answer over the bar. If an answer is *nothing but* such sentences the
        # total weight is 0, and an answer with no verifiable content is not a
        # grounded answer: score 0.0, refused.
        weights = [0 if v.method == "trivial" else max(1, len(_tokens(v.text))) for v in verdicts]
        total = sum(weights)
        score = (
            sum(w * v.score for w, v in zip(weights, verdicts)) / total if total else 0.0
        )
        unsupported = [v for v in verdicts if not v.supported]
        hard = any(v.unsupported_numbers or v.invalid_citations or v.unsupported_citations
                   for v in unsupported)
        grounded = (
            not hard
            and score >= cfg.answer_threshold
            and (not unsupported if cfg.require_all_claims else True)
        )
        return GroundingVerdict(
            grounded=grounded,
            per_claim=verdicts,
            unsupported_claims=[v.text for v in unsupported],
            score=score,
            latency_ms=(time.perf_counter_ns() - t0) / 1e6,
            used_nli=used_nli,
        )


def _parse_citations(claim: str) -> tuple[int, ...]:
    """Extract 1-based citation indices from ``[1]``, ``[2, 3]``, ``[1][4]``.

    Returns:
        Indices in order of first appearance, de-duplicated. Ordinary square
        brackets around non-numeric text are ignored by construction.
    """
    out: list[int] = []
    for m in _CITATION.finditer(claim):
        for part in re.split(r"[,;]", m.group(1)):
            part = part.strip()
            if part.isdigit():
                n = int(part)
                if n not in out:
                    out.append(n)
    return tuple(out)
