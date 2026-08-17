"""Pre-retrieval input guard: the cheapest possible "should we even try?".

This runs *before* embedding, retrieval and generation, on every request, so its
budget is microseconds. Everything here is pure Python and precompiled regex --
no numpy, no model, no I/O -- which is why it can sit unconditionally in front of
a 200 ms pipeline without being part of the latency story.

Three families of rejection, in increasing order of subtlety:

**Degenerate transcripts.** Streaming STT emits *something* for silence, a cough
or a hot mic: ``"um"``, ``"uh huh"``, ``"hello?"``, ``"mic check one two"``.
Retrieving for those wastes an index lookup and, far worse, an LLM call whose
output will be confidently irrelevant. Catching them here is the single highest
value/cost ratio in the whole guardrail stack.

**Prompt injection.** The transcript is untrusted text that ends up inside a
prompt, so it is an injection surface even though it arrived by voice. The layer
shipped here is heuristic: weighted regex over a normalised transcript, summed
into a score, compared against a documented threshold. The design point that
matters is the *false positive* stance -- see :data:`INJECTION_PATTERNS`.

**Unsafe content.** A compact, deliberately illustrative wordlist plus a few
harm-seeking patterns. This is not a content moderation system and does not
pretend to be one; see :data:`_STRONG_PROFANITY`.

A model-based injection detector (``protectai/deberta-v3-base-prompt-injection-v2``
exported to ONNX) is the production upgrade. It is *not* the default because it
costs ~11 ms at seq_len 64 on this box (measured, ``bench_guardrails.py``) --
5.5 % of the entire latency budget to catch obfuscations the heuristic misses.
The seam is :attr:`InputGuardConfig.model_detector`: supply any
``Callable[[str], float]`` and its score is fused with the heuristic. Nothing in
this module imports a model, so the import cost stays at zero.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterable, Mapping

__all__ = [
    "InputVerdict",
    "InputGuardConfig",
    "InputGuard",
    "InjectionPattern",
    "INJECTION_PATTERNS",
    "normalise",
]


# --- result -------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class InputVerdict:
    """Outcome of the pre-retrieval check.

    Attributes:
        allowed: ``True`` if the request should proceed to retrieval.
        reason: Human-readable, specific explanation. ``None`` when allowed.
            Written to be shown to the user verbatim -- "the transcript was only
            filler words" beats "invalid input".
        category: Machine-readable rejection class, one of ``empty``,
            ``too_short``, ``filler``, ``too_long``, ``self_harm``, ``unsafe``,
            ``prompt_injection``, ``profanity``. ``None`` when allowed.
        scores: Every signal computed, whether or not it fired. Exposed so the
            HUD and the eval harness can see *how close* a benign query came to
            being blocked, which is how the threshold gets tuned.
        latency_ms: Wall-clock cost of this check.
        matched: Names of the injection/unsafe patterns that fired. Empty when
            nothing matched.
    """

    allowed: bool
    reason: str | None = None
    category: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    matched: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "category": self.category,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "latency_ms": round(self.latency_ms, 4),
            "matched": list(self.matched),
        }


# --- text normalisation -------------------------------------------------------

#: Zero-width and bidirectional control characters. These exist in injection
#: payloads for exactly one reason: to break naive regex matching while leaving
#: the rendered text (and the tokeniser's view of it) unchanged.
_INVISIBLE: Final = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\u00ad]")
_WS: Final = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Canonicalise a transcript before pattern matching.

    NFKC folds the fullwidth/lookalike forms that survive a copy-paste through a
    chat client; invisible characters are dropped outright; whitespace is
    collapsed so ``\\n\\n###   system`` and ``### system`` match the same rule.

    Args:
        text: Raw transcript.

    Returns:
        Normalised text, whitespace-collapsed and stripped. Case is preserved --
        the patterns are case-insensitive, and preserving case keeps the string
        usable for the reason message.
    """
    t = unicodedata.normalize("NFKC", text)
    t = _INVISIBLE.sub("", t)
    return _WS.sub(" ", t).strip()


_LEET: Final = str.maketrans({"4": "a", "3": "e", "1": "i", "0": "o", "5": "s",
                              "7": "t", "@": "a", "$": "s", "!": "i"})
_RUN2: Final = re.compile(r"(.)\1+")
_TOKEN: Final = re.compile(r"[^\W_]+", re.UNICODE)
#: A run of single letters glued by punctuation or digits -- ``f.u.c.k``,
#: ``s-h-i-t``. Matching this *shape* first is what lets the wordlist check stay
#: token-exact, which is the whole defence against the Scunthorpe problem.
_SEPARATED: Final = re.compile(r"\b(?:[a-z](?:[^a-z0-9\s]|\d)){2,}[a-z]\b", re.I)
_NON_ALPHA: Final = re.compile(r"[^a-z]")


def _fold(token: str) -> str:
    """Undo the two obfuscations a wordlist can cheaply survive.

    Leetspeak substitution (``sh1t``) and character repetition (``shiiiit``).
    The *same* fold is applied to the wordlist at import time, so comparison is
    an exact set membership test on both sides and no substring matching -- and
    therefore no Scunthorpe problem -- is ever involved.
    """
    return _RUN2.sub(r"\1", token.lower().translate(_LEET))


# --- injection patterns -------------------------------------------------------


@dataclass(slots=True, frozen=True)
class InjectionPattern:
    """One weighted prompt-injection heuristic.

    Attributes:
        name: Stable identifier, surfaced in :attr:`InputVerdict.matched`.
        regex: Compiled, case-insensitive.
        weight: Contribution to the injection score. Scores sum and clamp at
            1.0, so a weight >= the threshold is a unilateral block.
        hard: ``True`` for patterns that are imperative *commands aimed at the
            model*. Hard matches are immune to the interrogative discount below;
            soft matches are topical mentions that a legitimate question could
            plausibly contain.
    """

    name: str
    regex: re.Pattern[str]
    weight: float
    hard: bool = True


def _p(name: str, pattern: str, weight: float, hard: bool = True) -> InjectionPattern:
    return InjectionPattern(name, re.compile(pattern, re.IGNORECASE), weight, hard)


#: The heuristic layer, ordered roughly by how damning each match is.
#:
#: **The false-positive stance.** A RAG corpus contains questions *about* LLMs,
#: so ``"what does the system prompt of an AI assistant do?"`` is a legitimate
#: query that must be answered, not blocked. The rule that separates it from an
#: attack is grammatical, not lexical: an attack is an *imperative directed at
#: the model* ("reveal your system prompt"), while a legitimate query is an
#: *interrogative about a topic* ("what does a system prompt do"). So the bare
#: phrase "system prompt" is a soft signal weighted at 0.35 -- deliberately
#: below the 0.6 threshold, so it can never block on its own -- and it is
#: further halved when the transcript is a well-formed question with no hard
#: match. Every hard pattern requires a command verb plus a second-person
#: possessive or an explicit override phrase.
INJECTION_PATTERNS: Final[tuple[InjectionPattern, ...]] = (
    _p(
        "instruction_override",
        r"\b(ignore|disregard|forget|override|discard)\b[^.?!]{0,30}"
        r"\b(previous|prior|above|earlier|all|any)\b[^.?!]{0,30}"
        r"\b(instruction|prompt|rule|direction|command|context)s?\b",
        1.0,
    ),
    _p(
        "prompt_exfiltration",
        r"\b(reveal|show|print|repeat|output|display|leak|dump|recite)\b[^.?!]{0,25}"
        r"\byour\b[^.?!]{0,20}"
        r"\b(system\s*prompt|initial\s+instructions?|instructions?|rules|guidelines|prompt)\b",
        1.0,
    ),
    _p(
        "verbatim_exfiltration",
        r"\b(repeat|print|output|echo|reproduce)\b[^.?!]{0,20}"
        r"\b(everything|all|the\s+text|the\s+words)\b[^.?!]{0,15}\b(above|before)\b",
        0.9,
    ),
    _p(
        "persona_override",
        r"\byou\s+are\s+now\b|\bfrom\s+now\s+on,?\s+you\b|\bpretend\s+(to\s+be|you\s+are)\b"
        r"|\bact\s+as\s+(if\s+you\s+are\s+)?(an?\s+)?(dan|jailbroken|unrestricted|unfiltered|evil)\b",
        0.9,
    ),
    _p(
        "named_jailbreak",
        r"\b(dan\s+mode|developer\s+mode|do\s+anything\s+now|sudo\s+mode|god\s+mode)\b",
        0.8,
    ),
    _p(
        "safety_override",
        r"\bbypass\s+(your\s+)?(safety|filters?|guardrails?|restrictions?)\b"
        r"|\b(with|using)\s+(no|without\s+any)\s+(restrictions?|filters?|censorship|limitations?)\b"
        r"|\bstop\s+being\s+(an?\s+)?(ai|assistant|helpful)\b",
        0.8,
    ),
    _p(
        "new_instructions",
        r"\b(new|updated|revised|real|actual)\s+(instructions?|system\s+prompt|rules?)\s*[:\-]",
        0.8,
    ),
    _p(
        "delimiter_injection",
        r"<\|[a-z_]{2,20}\|>|\[/?INST\]|\[/?SYS\]|</?(system|assistant|user)>"
        r"|(?:^|\s)###\s*(system|instruction)",
        0.9,
    ),
    _p(
        "role_label",
        r"(?:^|\s)(system|assistant)\s*:\s*(?=\S)",
        0.7,
    ),
    # --- soft signals: topical, not imperative --------------------------------
    _p("system_prompt_mention", r"\bsystem\s+prompt\b|\bprompt\s+injection\b", 0.35, hard=False),
    # "what is jailbreaking an iPhone" is a real MS MARCO query. The word alone
    # is a topic; only "act as jailbroken" (persona_override) is an instruction.
    _p("jailbreak_mention", r"\bjailbreak(s|en|ing|ed)?\b|\bjailbroken\b", 0.35, hard=False),
    _p(
        "encoded_payload",
        r"\b[A-Za-z0-9+/]{48,}={0,2}\b",
        0.5,
        hard=False,
    ),
)

#: A transcript that opens with a wh-word/auxiliary, or ends in a question mark,
#: is behaving like a question. Used only to *discount* soft signals.
_INTERROGATIVE: Final = re.compile(
    r"^\s*(what|why|how|when|where|who|whom|whose|which|is|are|was|were|do|does|did"
    r"|can|could|should|would|will|has|have|had|tell\s+me|explain|define|describe"
    r"|name|list|in\s+what|according\s+to)\b",
    re.IGNORECASE,
)


# --- unsafe content -----------------------------------------------------------

#: Harm-seeking *instructional* requests. The line drawn here is instructional
#: vs informational: "what is a pipe bomb" is an encyclopaedia question that
#: MS MARCO genuinely contains and this guard deliberately allows; "how do I
#: make a pipe bomb" is a request for capability and is refused.
_UNSAFE_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "weapon_synthesis",
        re.compile(
            r"\bhow\s+(to|do\s+i|can\s+i|would\s+i|d\s+i)\b[^.?!]{0,40}"
            r"\b(make|build|construct|synthesi[sz]e|manufacture|cook)\b[^.?!]{0,40}"
            r"\b(bomb|explosive|ied|napalm|nerve\s+agent|sarin|ricin|meth|methamphetamine"
            r"|fentanyl|silencer|untraceable\s+gun)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "malware_authoring",
        re.compile(
            r"\b(write|create|generate|code|build)\b[^.?!]{0,30}"
            r"\b(malware|ransomware|keylogger|rootkit|botnet|computer\s+virus)\b",
            re.IGNORECASE,
        ),
    ),
)

#: Routed to its own category so the API can answer with crisis resources rather
#: than a generic refusal. Conflating the two is a product failure, not a
#: nuance -- a person in crisis should never receive "I cannot help with that".
_SELF_HARM: Final = re.compile(
    r"\b(kill\s+myself|end\s+my\s+life|commit\s+suicide|suicide\s+methods?"
    r"|how\s+to\s+(kill|hurt)\s+myself|self[-\s]?harm|cut\s+myself)\b",
    re.IGNORECASE,
)

#: Illustrative, not exhaustive, and intentionally free of slurs -- publishing a
#: slur list inside a hackathon repo is not a feature. A real deployment swaps
#: in a maintained list (LDNOOBW) or a toxicity classifier
#: (``minuva/MiniLMv2-toxic-jigsaw``, ~4 ms at seq 64, measured) behind the same
#: :attr:`InputGuardConfig.model_detector`-style seam.
_STRONG_PROFANITY: Final[frozenset[str]] = frozenset(
    {"fuck", "fucking", "fucker", "motherfucker", "shit", "bullshit", "bitch",
     "bastard", "asshole", "arsehole", "cunt", "wanker", "prick", "twat", "dickhead"}
)
_MILD_PROFANITY: Final[frozenset[str]] = frozenset(
    {"damn", "damned", "hell", "crap", "bloody", "piss", "pissed", "bugger",
     "bollocks", "arse", "ass"}
)
#: Folded copies, so membership tests compare like with like (see :func:`_fold`).
_STRONG_FOLDED: Final[frozenset[str]] = frozenset(_fold(w) for w in _STRONG_PROFANITY)
_MILD_FOLDED: Final[frozenset[str]] = frozenset(_fold(w) for w in _MILD_PROFANITY)

#: Disfluencies, acknowledgements and mic-check noise. Contains no content word
#: and no wh-word, which is what makes the "every token is filler" rule safe:
#: a single real word anywhere in the transcript keeps it alive. Single-letter
#: words are deliberately absent -- they are handled by the length rule, so that
#: ``"a"`` is reported as too short rather than as filler.
_FILLER: Final[frozenset[str]] = frozenset(
    {"um", "umm", "uhm", "uh", "uhh", "er", "erm", "ah", "aah", "eh", "oh", "hm",
     "hmm", "mhm", "mm", "mmm", "huh", "hey", "hi", "hello", "yo", "ok", "okay",
     "yeah", "yep", "yes", "no", "nope", "nah", "so", "well", "like", "right",
     "anyway", "actually", "basically", "literally", "just", "you", "know",
     "mean", "the", "and", "test", "testing", "check", "mic", "one", "two",
     "three", "sorry", "thanks", "thank", "please", "wait", "hold", "on"}
)


# --- configuration ------------------------------------------------------------


@dataclass(slots=True)
class InputGuardConfig:
    """Every tunable in this stage, in one place, serialisable to JSON.

    Attributes:
        min_chars: Transcripts shorter than this (after normalisation) are
            rejected as ``too_short``. Three characters admits ``"ai"`` only via
            ``min_content_tokens``; it exists to kill ``"?"`` and ``"a"``.
        min_content_tokens: Number of non-filler tokens of length >= 2 required.
            One, not two: ``"photosynthesis"`` is a real query.
        max_chars: Upper bound on a *spoken* question. A 600-character transcript
            is not speech; it is a paste, and pastes are where injection payloads
            live. Also bounds regex work, so the stage stays O(1)-ish.
        injection_threshold: Summed pattern weight at which the request is
            refused. 0.6 admits any single soft signal and blocks any single
            hard one.
        soft_discount: Multiplier applied to the injection score when the
            transcript reads as a question and no hard pattern fired.
        profanity_threshold: Strong terms score 1.0, mild 0.25. At 0.7 a single
            strong term blocks and any amount of mild profanity does not, so
            ``"what the hell is a quasar"`` is answered.
        block_profanity: Product policy switch. Profanity in a *question* is not
            an attack, and some deployments would rather answer it; turn this
            off and the score is still reported, just not enforced.
        check_injection / check_unsafe: Stage switches for ablation.
        model_detector: Optional ``Callable[[str], float]`` returning P(injection)
            in ``[0, 1]``. When present its score is fused with the heuristic by
            ``max`` -- a union of detectors, since each catches what the other
            misses. Never called if ``check_injection`` is off.
    """

    min_chars: int = 3
    min_content_tokens: int = 1
    max_chars: int = 600
    injection_threshold: float = 0.6
    soft_discount: float = 0.5
    profanity_threshold: float = 0.7
    block_profanity: bool = True
    check_injection: bool = True
    check_unsafe: bool = True
    model_detector: Callable[[str], float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view. ``model_detector`` is a live object and is reported
        as a boolean presence flag rather than serialised."""
        return {
            "min_chars": self.min_chars,
            "min_content_tokens": self.min_content_tokens,
            "max_chars": self.max_chars,
            "injection_threshold": self.injection_threshold,
            "soft_discount": self.soft_discount,
            "profanity_threshold": self.profanity_threshold,
            "block_profanity": self.block_profanity,
            "check_injection": self.check_injection,
            "check_unsafe": self.check_unsafe,
            "model_detector": self.model_detector is not None,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        model_detector: Callable[[str], float] | None = None,
    ) -> "InputGuardConfig":
        """Rebuild from :meth:`to_dict`; the detector must be re-supplied."""
        fields = {
            "min_chars", "min_content_tokens", "max_chars", "injection_threshold",
            "soft_discount", "profanity_threshold", "block_profanity",
            "check_injection", "check_unsafe",
        }
        kw = {k: v for k, v in data.items() if k in fields}
        return cls(model_detector=model_detector, **kw)  # type: ignore[arg-type]


# --- the guard ----------------------------------------------------------------


class InputGuard:
    """Sub-millisecond pre-retrieval filter.

    Args:
        config: Tunables. Defaults are the shipped operating point.

    Example:
        >>> InputGuard().check("um uh").category
        'filler'
        >>> InputGuard().check("what is the boiling point of water").allowed
        True
    """

    def __init__(self, config: InputGuardConfig | None = None) -> None:
        self.config = config or InputGuardConfig()

    def __repr__(self) -> str:
        return f"InputGuard(injection_threshold={self.config.injection_threshold})"

    # -- individual signals ---------------------------------------------------

    def injection_score(self, text: str) -> tuple[float, list[str], bool]:
        """Score a normalised transcript for prompt injection.

        Args:
            text: Output of :func:`normalise`.

        Returns:
            ``(score, matched_names, any_hard)``. ``score`` is clamped to
            ``[0, 1]`` and already includes the interrogative discount and the
            optional model detector.
        """
        cfg = self.config
        total = 0.0
        matched: list[str] = []
        hard = False
        for pat in INJECTION_PATTERNS:
            if pat.regex.search(text):
                matched.append(pat.name)
                total += pat.weight
                hard = hard or pat.hard
        if matched and not hard and self._is_question(text):
            total *= cfg.soft_discount
        total = min(1.0, total)
        if cfg.model_detector is not None:
            model = float(cfg.model_detector(text))
            if model > total:
                total = min(1.0, model)
                matched.append("model_detector")
        return total, matched, hard

    @staticmethod
    def _is_question(text: str) -> bool:
        return bool(_INTERROGATIVE.match(text)) or text.rstrip().endswith("?")

    @staticmethod
    def profanity_score(tokens: Iterable[str], text: str) -> float:
        """Score profanity by exact membership on folded tokens.

        Args:
            tokens: Word tokens of the transcript (any case).
            text: The normalised transcript, used only to catch separator
                obfuscation (``f.u.c.k``) that destroys tokenisation.

        Returns:
            ``1.0`` for any strong term, ``0.25`` for a mild one, ``0.0``
            otherwise. Deliberately not additive: ten mild words are still mild.
        """
        score = 0.0
        for tok in tokens:
            folded = _fold(tok)
            if folded in _STRONG_FOLDED:
                return 1.0
            if folded in _MILD_FOLDED:
                score = max(score, 0.25)
        for m in _SEPARATED.finditer(text):
            if _fold(_NON_ALPHA.sub("", m.group(0).lower())) in _STRONG_FOLDED:
                return 1.0
        return score

    # -- the check ------------------------------------------------------------

    def check(self, text: str | None) -> InputVerdict:
        """Decide whether a transcript is worth retrieving for.

        Order matters and is deliberate: degenerate input is checked first
        because it is the common case and the cheapest, self-harm before generic
        unsafe content because it needs a different response, and injection
        before profanity because an injection payload containing profanity
        should be reported as an injection.

        Args:
            text: Raw transcript. ``None`` and whitespace are handled.

        Returns:
            An :class:`InputVerdict`. ``scores`` is always fully populated, even
            on the allow path, so the HUD can show margins.
        """
        t0 = time.perf_counter_ns()
        cfg = self.config
        raw = normalise(text or "")
        tokens = _TOKEN.findall(raw)
        content = [tok for tok in tokens if len(tok) >= 2 and tok.lower() not in _FILLER]
        scores: dict[str, float] = {
            "n_chars": float(len(raw)),
            "n_tokens": float(len(tokens)),
            "n_content_tokens": float(len(content)),
            "filler_ratio": 0.0 if not tokens else 1.0 - len(content) / len(tokens),
            "injection": 0.0,
            "profanity": 0.0,
            "unsafe": 0.0,
        }

        def done(
            allowed: bool,
            reason: str | None = None,
            category: str | None = None,
            matched: tuple[str, ...] = (),
        ) -> InputVerdict:
            return InputVerdict(
                allowed=allowed,
                reason=reason,
                category=category,
                scores=scores,
                latency_ms=(time.perf_counter_ns() - t0) / 1e6,
                matched=matched,
            )

        # 1. degenerate transcripts -------------------------------------------
        # Three distinct categories, kept distinct because they have different
        # fixes: nothing was heard, only disfluencies were heard, or something
        # real but unusably brief was heard.
        if not raw or not tokens:
            return done(False, "I didn't hear a question -- the transcript had no words.", "empty")
        if len(raw) > cfg.max_chars:
            return done(
                False,
                f"That's {len(raw)} characters; I only take spoken questions up to "
                f"{cfg.max_chars}. Ask the specific question you want answered.",
                "too_long",
            )
        lowered = [tok.lower() for tok in tokens]
        if any(tok in _FILLER for tok in lowered) and all(
            tok in _FILLER or len(tok) < 2 for tok in lowered
        ):
            return done(
                False,
                f"That sounded like filler rather than a question (“{raw}”). "
                "Ask me something about the indexed passages.",
                "filler",
            )
        if len(raw) < cfg.min_chars or len(content) < cfg.min_content_tokens:
            return done(
                False,
                f"“{raw}” is too short to retrieve on -- I need at least "
                f"{cfg.min_content_tokens} content word(s).",
                "too_short",
            )

        # 2. safety ------------------------------------------------------------
        if cfg.check_unsafe:
            if _SELF_HARM.search(raw):
                scores["unsafe"] = 1.0
                return done(
                    False,
                    "This isn't something a search index should answer. If you're "
                    "struggling, please contact a local crisis line -- in India, "
                    "Tele-MANAS on 14416.",
                    "self_harm",
                    ("self_harm",),
                )
            for name, pattern in _UNSAFE_PATTERNS:
                if pattern.search(raw):
                    scores["unsafe"] = 1.0
                    return done(
                        False,
                        "That asks for instructions to cause harm, which I won't "
                        "retrieve or generate. I'm happy to answer factual questions "
                        "about the topic instead.",
                        "unsafe",
                        (name,),
                    )

        # 3. prompt injection ---------------------------------------------------
        if cfg.check_injection:
            score, matched, _hard = self.injection_score(raw)
            scores["injection"] = score
            if score >= cfg.injection_threshold:
                return done(
                    False,
                    "That reads as an attempt to change my instructions rather than "
                    f"a question about the corpus (matched: {', '.join(matched)}). "
                    "I only answer from the indexed passages.",
                    "prompt_injection",
                    tuple(matched),
                )

        # 4. profanity ----------------------------------------------------------
        if cfg.check_unsafe:
            prof = self.profanity_score(tokens, raw)
            scores["profanity"] = prof
            if cfg.block_profanity and prof >= cfg.profanity_threshold:
                return done(
                    False,
                    "Let's keep it civil -- ask the question without the profanity "
                    "and I'll answer it.",
                    "profanity",
                    ("profanity",),
                )

        return done(True)
