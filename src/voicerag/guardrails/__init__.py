"""Guardrails: knowing when not to answer, and checking what was answered.

Three stages at three points in the pipeline, ordered by cost:

===================  ================  =======================================
Stage                Runs              Cost
===================  ================  =======================================
:mod:`input_guard`   before retrieval  ~10 us, pure regex
:mod:`abstention`    after retrieval   ~30 us, numpy over signals fusion
                                       already computed
:mod:`grounding`     after generation  <1 ms lexical; optional NLI is opt-in
===================  ================  =======================================

:mod:`policy` composes all three, owns every threshold, serialises to JSON and
emits the single :class:`~voicerag.guardrails.policy.GuardrailReport` that the
API returns and the frontend HUD renders.

Importing this package pulls in numpy and the stdlib and nothing else -- no
model, no index, no network -- so it is safe on any code path.
"""

from .abstention import (
    FEATURE_NAMES,
    PRIOR_RULES,
    AbstentionGate,
    AbstentionVerdict,
    CalibratedModel,
    CalibrationResult,
    SignalRule,
    binary_metrics,
    retrieval_signals,
)
from .grounding import (
    ClaimVerdict,
    GroundingBackendUnavailable,
    GroundingChecker,
    GroundingConfig,
    GroundingVerdict,
    NLIScorer,
    extract_numbers,
    load_nli,
)
from .input_guard import (
    INJECTION_PATTERNS,
    InjectionPattern,
    InputGuard,
    InputGuardConfig,
    InputVerdict,
    normalise,
)
from .policy import GuardrailPolicy, GuardrailReport, salient_terms

__all__ = [
    # input
    "InputGuard",
    "InputGuardConfig",
    "InputVerdict",
    "InjectionPattern",
    "INJECTION_PATTERNS",
    "normalise",
    # abstention
    "AbstentionGate",
    "AbstentionVerdict",
    "CalibratedModel",
    "CalibrationResult",
    "SignalRule",
    "PRIOR_RULES",
    "FEATURE_NAMES",
    "binary_metrics",
    "retrieval_signals",
    # grounding
    "GroundingChecker",
    "GroundingConfig",
    "GroundingVerdict",
    "ClaimVerdict",
    "NLIScorer",
    "GroundingBackendUnavailable",
    "load_nli",
    "extract_numbers",
    # policy
    "GuardrailPolicy",
    "GuardrailReport",
    "salient_terms",
]
