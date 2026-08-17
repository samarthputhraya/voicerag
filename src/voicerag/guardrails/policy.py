"""One policy object composing the three guardrail stages.

The stages are independent by design -- input filtering, abstention and
grounding answer different questions at different points in the pipeline -- but
the API, the HUD and the ablation all want a *single* object that owns every
threshold and emits a *single* report. That object is :class:`GuardrailPolicy`.

Two things it deliberately does not do. It does not run retrieval or generation:
the caller owns the pipeline and calls each stage where it belongs, because only
the caller knows what can be overlapped with what. And it does not decide the
final response text on its own -- :meth:`GuardrailPolicy.refusal_message` returns
a string and the API decides whether to send it, which keeps this module free of
transport concerns and trivially testable.

**Refusals are specific.** "I cannot answer that" is worthless to a user and
worse to a judge: it is indistinguishable from a crash. Every refusal this
module produces names the actual cause with the actual number -- which signal
failed, what it measured, what the corpus does not contain. That is also what
makes the abstention behaviour demonstrable rather than assertable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .abstention import AbstentionGate, AbstentionVerdict
from .grounding import GroundingChecker, GroundingConfig, GroundingVerdict, NLIScorer
from .input_guard import InputGuard, InputGuardConfig, InputVerdict

__all__ = ["GuardrailReport", "GuardrailPolicy", "salient_terms"]

_WORD = re.compile(r"[a-z0-9']+")
#: Words that carry no topic. Includes the wh-words, because "what is X" is
#: about X, not about "what".
_QUERY_STOP = frozenset(
    """what which who whom whose when where why how is are was were do does did
    a an the of in on at to for from by with and or but be been being as that
    this these those it its i you me my your can could should would will shall
    tell show explain define describe list name about into over under please
    there here have has had many much""".split()
)


def salient_terms(query: str, limit: int = 6) -> list[str]:
    """The content words of a query, in order, de-duplicated.

    Used to write refusals that name the subject ("nothing on *tarsier
    gestation*") instead of echoing the whole transcript back, which reads like
    a parser error.

    Args:
        query: Raw user query or transcript.
        limit: Maximum terms returned.

    Returns:
        Lowercased content words. Empty if the query was entirely function
        words, in which case the caller should fall back to the raw query.
    """
    out: list[str] = []
    for w in _WORD.findall(query.lower()):
        if w in _QUERY_STOP or len(w) < 2:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= limit:
            break
    return out


@dataclass(slots=True)
class GuardrailReport:
    """The whole guardrail outcome, in the shape the frontend already expects.

    The first nine fields are a fixed contract with ``web/lib/api.ts``; renaming
    one silently blanks a panel in the HUD, so they are listed here in the same
    order as in the TypeScript interface. Everything after them is additive
    detail that clients may ignore.

    Attributes:
        input_allowed: Did the transcript survive the pre-retrieval guard.
        input_reason: Why not, if not.
        abstained: Did the system decline to deliver a generated answer, for
            *any* reason -- blocked input, insufficient retrieval, or an answer
            that failed grounding. This is what the UI badges.
        abstain_reason: The specific cause, quantitative where possible.
        abstain_confidence: ``P(the corpus cannot answer this)`` from the
            abstention gate. Reported even when the system answered, so the HUD
            can show how close the call was.
        abstain_signals: Every retrieval signal behind that number.
        grounded: Grounding outcome, or ``None`` when nothing was generated.
        grounding_score: Length-weighted mean claim support in ``[0, 1]``.
        unsupported_claims: Sentences the grounding layer could not support.
        input_category: Machine-readable rejection class.
        calibrated: Whether the abstention gate was using a fitted model.
        stage_latency_ms: Per-stage cost, for the latency receipt.
        latency_ms: Sum of the stages that ran.
    """

    input_allowed: bool = True
    input_reason: str | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    abstain_confidence: float = 0.0
    abstain_signals: dict[str, float] = field(default_factory=dict)
    grounded: bool | None = None
    grounding_score: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    input_category: str | None = None
    calibrated: bool = False
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON body for the API response. Keys match ``GuardrailReport`` in
        ``web/lib/api.ts`` exactly."""
        return {
            "input_allowed": self.input_allowed,
            "input_reason": self.input_reason,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "abstain_confidence": round(self.abstain_confidence, 4),
            "abstain_signals": {k: round(v, 5) for k, v in self.abstain_signals.items()},
            "grounded": self.grounded,
            "grounding_score": (
                None if self.grounding_score is None else round(self.grounding_score, 4)
            ),
            "unsupported_claims": list(self.unsupported_claims),
            "input_category": self.input_category,
            "calibrated": self.calibrated,
            "stage_latency_ms": {k: round(v, 3) for k, v in self.stage_latency_ms.items()},
            "latency_ms": round(self.latency_ms, 3),
        }


class GuardrailPolicy:
    """Composes the three stages and owns every threshold in one place.

    Args:
        input_guard: Pre-retrieval filter.
        abstention: Retrieval-signal gate.
        grounding: Post-generation verifier.
        answer_on_ungrounded: If ``False`` (the default) an answer that fails
            grounding is withheld and reported as an abstention. Turning it on
            shows the answer with the unsupported sentences flagged instead --
            the right choice for a research UI, the wrong one for a product.

    Example:
        >>> policy = GuardrailPolicy()
        >>> report = policy.report("um", input_verdict=policy.check_input("um"))
        >>> report.abstained, report.input_category
        (True, 'filler')
    """

    __slots__ = ("input_guard", "abstention", "grounding", "answer_on_ungrounded")

    def __init__(
        self,
        input_guard: InputGuard | None = None,
        abstention: AbstentionGate | None = None,
        grounding: GroundingChecker | None = None,
        *,
        answer_on_ungrounded: bool = False,
    ) -> None:
        self.input_guard = input_guard or InputGuard()
        self.abstention = abstention or AbstentionGate()
        self.grounding = grounding or GroundingChecker()
        self.answer_on_ungrounded = answer_on_ungrounded

    def __repr__(self) -> str:
        return (
            f"GuardrailPolicy({self.input_guard!r}, {self.abstention!r}, "
            f"{self.grounding!r})"
        )

    # -- stages ---------------------------------------------------------------

    def check_input(self, text: str | None) -> InputVerdict:
        """Run the pre-retrieval guard. See :meth:`InputGuard.check`."""
        return self.input_guard.check(text)

    def check_retrieval(self, hits: Sequence[Any] | Mapping[str, float]) -> AbstentionVerdict:
        """Run the abstention gate. See :meth:`AbstentionGate.judge`."""
        return self.abstention.judge(hits)

    def check_answer(self, answer: str, contexts: Sequence[Any]) -> GroundingVerdict:
        """Run the lexical grounding layer. See :meth:`GroundingChecker.verify`."""
        return self.grounding.verify(answer, contexts)

    async def acheck_answer(self, answer: str, contexts: Sequence[Any]) -> GroundingVerdict:
        """Async grounding, with the optional NLI layer off the event loop."""
        return await self.grounding.averify(answer, contexts)

    # -- composition ----------------------------------------------------------

    def report(
        self,
        query: str,
        *,
        input_verdict: InputVerdict | None = None,
        abstention_verdict: AbstentionVerdict | None = None,
        grounding_verdict: GroundingVerdict | None = None,
    ) -> GuardrailReport:
        """Fold whichever stages ran into one report.

        Stages are optional because the pipeline short-circuits: a blocked
        transcript never reaches retrieval, and an abstention never reaches
        generation, so the corresponding verdicts legitimately do not exist.
        Passing ``None`` records that a stage did not run rather than that it
        passed -- the difference matters when reading a trace.

        Args:
            query: The user's query, used to write a topical refusal.
            input_verdict: From :meth:`check_input`.
            abstention_verdict: From :meth:`check_retrieval`.
            grounding_verdict: From :meth:`check_answer`.

        Returns:
            A :class:`GuardrailReport` whose ``abstain_reason`` is set by the
            *earliest* stage that refused, since that is the cause a user needs
            to hear; later stages could not have run anyway.
        """
        report = GuardrailReport(calibrated=self.abstention.calibrated)
        stages: dict[str, float] = {}

        if input_verdict is not None:
            stages["input_guard"] = input_verdict.latency_ms
            report.input_allowed = input_verdict.allowed
            report.input_reason = input_verdict.reason
            report.input_category = input_verdict.category
            if not input_verdict.allowed:
                report.abstained = True
                report.abstain_reason = input_verdict.reason
                report.abstain_confidence = 1.0

        if abstention_verdict is not None:
            stages["abstention"] = abstention_verdict.latency_ms
            report.abstain_signals = dict(abstention_verdict.signals)
            if not report.abstained:
                report.abstain_confidence = abstention_verdict.confidence
                if abstention_verdict.should_abstain:
                    report.abstained = True
                    report.abstain_reason = self.refusal_message(
                        query, abstention=abstention_verdict
                    )

        if grounding_verdict is not None:
            stages["grounding"] = grounding_verdict.latency_ms
            report.grounded = grounding_verdict.grounded
            report.grounding_score = grounding_verdict.score
            report.unsupported_claims = list(grounding_verdict.unsupported_claims)
            if not grounding_verdict.grounded and not self.answer_on_ungrounded:
                if not report.abstained:
                    report.abstained = True
                    report.abstain_reason = self.refusal_message(
                        query, grounding=grounding_verdict
                    )

        report.stage_latency_ms = {k: round(v, 4) for k, v in stages.items()}
        report.latency_ms = sum(stages.values())
        return report

    def refusal_message(
        self,
        query: str,
        *,
        input_verdict: InputVerdict | None = None,
        abstention: AbstentionVerdict | None = None,
        grounding: GroundingVerdict | None = None,
    ) -> str:
        """Write the sentence the user actually reads when we decline.

        Exactly one of the three verdicts should be supplied -- the one that
        caused the refusal. Checked in pipeline order if several are given.

        Args:
            query: The user's query, mined for a topic to name.
            input_verdict: A blocked transcript. Its own reason is already
                specific, so it is returned unchanged.
            abstention: A gate refusal. Produces "nothing on <topic>" plus the
                measurement that justified it.
            grounding: An answer that failed verification. Names how many
                statements could not be supported and quotes the first.

        Returns:
            A user-facing sentence. Falls back to a generic-but-honest message
            only if no verdict is supplied.
        """
        if input_verdict is not None and not input_verdict.allowed:
            return input_verdict.reason or "That input can't be answered."

        if abstention is not None and abstention.should_abstain:
            terms = salient_terms(query)
            topic = " ".join(terms) if terms else (query.strip() or "that")
            detail = abstention.reason.removeprefix("Not answering: ").rstrip(".")
            return (
                f"The indexed passages have nothing on “{topic}”, so I'd be guessing: "
                f"{detail}."
            )

        if grounding is not None and not grounding.grounded:
            n = len(grounding.unsupported_claims)
            if n:
                first = grounding.unsupported_claims[0]
                why = next(
                    (c.reason for c in grounding.per_claim if c.text == first and c.reason),
                    "",
                )
                tail = f" — it {why}" if why else ""
                # A deadline-truncated answer can leave a one- or two-character
                # tail behind ("C"), which the sentence splitter then hands us as
                # a "claim". Quoting that back reads as a bug rather than as a
                # guardrail working, so name the count instead of the fragment.
                stub = len(first.strip()) < 12 or len(first.split()) < 3
                if stub:
                    return (
                        f"I drafted an answer but {n} statement(s) weren't supported by "
                        "the retrieved passages, so I won't state them."
                    )
                shown = first if len(first) <= 160 else first[:157].rstrip() + "…"
                return (
                    f"I drafted an answer but {n} statement(s) weren't supported by the "
                    f"retrieved passages, so I won't state them. The first was “{shown}”"
                    f"{tail}."
                )
            return (
                f"The drafted answer only reached {grounding.score:.0%} support against "
                "the retrieved passages, which is below the bar for stating it as fact."
            )

        return "I don't have grounds to answer that from the indexed passages."

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Every threshold in the stack, as JSON."""
        return {
            "version": 1,
            "answer_on_ungrounded": self.answer_on_ungrounded,
            "input": self.input_guard.config.to_dict(),
            "abstention": self.abstention.to_dict(),
            "grounding": self.grounding.config.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, Any],
        *,
        injection_model: Any = None,
        nli: NLIScorer | None = None,
    ) -> "GuardrailPolicy":
        """Rebuild a policy from :meth:`to_dict`.

        Args:
            d: Serialised policy.
            injection_model: Live object for the optional injection detector;
                only its presence is persisted, never the model itself.
            nli: Live NLI scorer, same reasoning.

        Returns:
            A configured :class:`GuardrailPolicy`.
        """
        return cls(
            InputGuard(
                InputGuardConfig.from_dict(d.get("input", {}), model_detector=injection_model)
            ),
            AbstentionGate.from_dict(d.get("abstention", {})),
            GroundingChecker(GroundingConfig.from_dict(d.get("grounding", {})), nli=nli),
            answer_on_ungrounded=bool(d.get("answer_on_ungrounded", False)),
        )

    def save(self, path: str | Path) -> Path:
        """Write the whole policy to one JSON file. Parent dirs are created."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        injection_model: Any = None,
        nli: NLIScorer | None = None,
    ) -> "GuardrailPolicy":
        """Read a policy written by :meth:`save`."""
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")),
            injection_model=injection_model,
            nli=nli,
        )

    def describe(self) -> dict[str, Any]:
        """Human-readable summary for the write-up and the ablation table."""
        return {
            "stages": ["input_guard", "abstention", "grounding"],
            "abstention_calibrated": self.abstention.calibrated,
            "abstention_model": (
                None if self.abstention.model is None
                else self.abstention.model.coefficients()
            ),
            "nli_enabled": self.grounding.nli is not None,
            "injection_model_enabled": self.input_guard.config.model_detector is not None,
            "config": self.to_dict(),
        }
