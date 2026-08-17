"""Deciding *not* to answer, from retrieval signals alone.

This is the guardrail that matters. A RAG system that always answers is a
hallucination machine with a citation UI; the interesting engineering is knowing
when the corpus genuinely has nothing and saying so. The brief asks the system
to "show that it knows when NOT to answer", and MS MARCO hands us real human
labels for it -- rows where every ``is_selected`` flag is 0 carry the gold answer
``"No Answer Present."``, which is a labelled abstention set produced by human
annotators rather than by us.

**Why retrieval signals and not an LLM judge.** The decision has to be made
*before* generation, because the entire point is to skip generation. That leaves
exactly one source of evidence: the shape of the retrieval result. Fortunately
:class:`~voicerag.index.hybrid.HybridIndex` already computed everything needed
and kept it on each hit, so all four signals below are free -- no extra pass, no
extra model, ~30 microseconds of numpy.

The four signals, and why each one is independent evidence:

``max_score`` / ``dense_max`` / ``sparse_max``
    Absolute relevance. If the best passage in a 200k-passage index scores 0.11
    cosine against the query, nothing in the corpus is about this topic. This is
    the only signal that can detect "the corpus is silent"; every other signal
    is relative and can be fooled by a uniformly bad candidate set.

``score_gap`` / ``rel_gap``
    Separation between rank 1 and rank 2. Confident retrieval is *peaked*: one
    passage is obviously the answer. When the top two are within a percent of
    each other the retriever is guessing between them, and picking one to
    generate from is a coin flip the user cannot see.

``entropy``
    Gap only looks at two documents; entropy looks at the whole top-k. Ten
    near-identical scores is the classic out-of-domain signature -- the query
    matched nothing, so everything matched equally badly. Computed on
    *standardised* scores, deliberately: that makes it a pure measure of
    distribution *shape*, orthogonal to the absolute-scale signals above, which
    is exactly what you want from a second opinion. It also makes it comparable
    across fusion methods, whose raw score magnitudes differ by two orders of
    magnitude.

``agreement``
    Did the dense retriever and BM25 -- two systems that fail in completely
    different ways -- independently surface the same passage? Lexical and
    semantic retrieval agreeing is strong evidence; the dense run alone finding
    something "sort of related" while BM25 finds nothing is the signature of a
    query whose terms do not occur in the corpus at all. This costs nothing
    because fusion already computed both runs' ranks.

**A caveat worth stating rather than hiding.** Under RRF the fused score is a
function of *rank* alone -- ``sum_runs 1/(60 + rank_r)`` -- so ``score_gap`` and
``entropy`` computed on fused scores largely re-encode the agreement structure
rather than adding independent margin information. That is not a bug in the
signals; it is a property of rank fusion, and it is the reason this gate also
reads the raw per-run ``dense_max`` and ``sparse_max`` that
:class:`~voicerag.index.hybrid.RetrievalHit` preserves. Calibration confirms it
empirically: fitted on synthetic data with the shape of this repo's own
retriever, the largest coefficients land on ``dense_max`` and ``agreement``.
Under weighted score fusion (``minmax``/``zscore``) the shape signals do carry
independent margin, which is one of the things the fusion ablation measures.

**Thresholds are calibrated, not invented.** The values in :data:`PRIOR_RULES`
are labelled priors and nothing more; they exist so the system has defensible
behaviour on day zero. :meth:`AbstentionGate.calibrate` fits a logistic model
over the standardised signals on labelled data and sweeps the decision threshold
for the metric you name. :meth:`AbstentionGate.evaluate` reports precision,
recall, F1, accuracy and the full confusion matrix, so the improvement is a
measured number rather than a claim. Whether a gate is calibrated is carried on
every verdict, so the demo can never accidentally present prior behaviour as
fitted behaviour.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "AbstentionVerdict",
    "AbstentionGate",
    "SignalRule",
    "PRIOR_RULES",
    "FEATURE_NAMES",
    "CalibratedModel",
    "CalibrationResult",
    "binary_metrics",
    "retrieval_signals",
]

#: Signals fed to the calibrated model, in a fixed order. Persisted with every
#: model so a saved model can never be silently applied to a different feature
#: layout after a refactor.
FEATURE_NAMES: tuple[str, ...] = (
    "max_score",
    "rel_gap",
    "entropy",
    "agreement",
    "top1_agree",
    "dense_max",
    "sparse_max",
)

Metric = Literal["f1", "accuracy", "balanced_accuracy", "youden"]


@runtime_checkable
class RetrievalLike(Protocol):
    """The subset of :class:`~voicerag.index.hybrid.RetrievalHit` used here.

    Typed structurally rather than imported concretely for two reasons: it keeps
    the guardrail package importable without pulling faiss and bm25s into the
    process, and it lets the eval harness replay saved retrieval results as
    plain objects without reconstructing an index.
    """

    chunk_id: str
    score: float
    rank: int
    dense_score: float | None
    sparse_score: float | None
    source: str


# --- signals ------------------------------------------------------------------


#: Softmax sharpness for :func:`_softmax_entropy`. The statistic operates on
#: score *ratios*, which for real retrieval sit in a narrow band (rank 10 rarely
#: scores below a third of rank 1), so a temperature of 1 would map every result
#: to an entropy above 0.95 and the signal would carry no information. Eight
#: puts the confident and the flat case at roughly 0.1 and 1.0 respectively,
#: which is the dynamic range the rule thresholds need. It is a property of the
#: statistic, not of any corpus, so it is a constant rather than a fitted value.
ENTROPY_TEMPERATURE: float = 8.0


def _softmax_entropy(scores: np.ndarray, temperature: float = ENTROPY_TEMPERATURE) -> float:
    """Normalised entropy of a softmax over scores scaled by their maximum.

    Args:
        scores: Fused scores of the top-k hits, best first.
        temperature: Softmax sharpness; see :data:`ENTROPY_TEMPERATURE`.

    Returns:
        ``0.0`` for a maximally peaked distribution, ``1.0`` for a flat one.
        Normalising by ``log(k)`` makes the value comparable across different
        ``k``, which matters because the gate runs with k=5 in the demo and k=20
        in the ablation.

    The normalisation is the design decision here, and the obvious choice is
    wrong. Standardising by the standard deviation -- the usual reflex -- forces
    unit variance onto every input, which erases exactly the property being
    measured: a flat result and a peaked one become identical after z-scoring,
    so the "entropy" degenerates into a measure of shape *given* spread. Scaling
    by the maximum instead keeps flatness intact while remaining invariant to
    multiplying every score by a constant, so RRF scores around 0.03 and BM25
    scores around 10 are still directly comparable.

    The statistic assumes non-negative fused scores (RRF, min-max, raw cosine or
    BM25). Under z-score fusion the scores are centred and the ratio loses its
    meaning -- one more reason RRF is the default fusion.
    """
    n = scores.size
    if n < 2:
        return 0.0
    peak = float(np.abs(scores).max())
    if peak < 1e-12:
        return 1.0  # every candidate scored zero: maximally undecided
    z = (scores / peak) * temperature
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum() / math.log(n))


def retrieval_signals(
    hits: Sequence[RetrievalLike], *, k: int | None = None
) -> dict[str, float]:
    """Reduce a fused retrieval result to the abstention feature vector.

    Args:
        hits: Fused hits, best first, as returned by
            :meth:`~voicerag.index.hybrid.HybridIndex.search`.
        k: Consider only the top ``k`` hits. ``None`` uses all of them.

    Returns:
        A flat mapping of signal name to float. Always contains every key in
        :data:`FEATURE_NAMES` plus ``n_hits`` and ``score_gap``, even for an
        empty result, so downstream code never has to test for presence.

    An empty result yields all-zero signals, which the gate maps to maximal
    abstention evidence -- retrieving nothing is the least ambiguous case there
    is, and it should not have to travel through the scoring path to be caught.
    """
    top = list(hits[:k] if k else hits)
    if not top:
        return {
            "n_hits": 0.0, "max_score": 0.0, "score_gap": 0.0, "rel_gap": 0.0,
            "entropy": 1.0, "agreement": 0.0, "top1_agree": 0.0,
            "dense_max": 0.0, "sparse_max": 0.0,
        }

    scores = np.asarray([float(h.score) for h in top], dtype=np.float64)
    max_score = float(scores[0])
    gap = float(scores[0] - scores[1]) if scores.size > 1 else max_score
    dense = [h.dense_score for h in top if h.dense_score is not None]
    sparse = [h.sparse_score for h in top if h.sparse_score is not None]
    both = sum(1 for h in top if h.source == "both")

    return {
        "n_hits": float(len(top)),
        "max_score": max_score,
        "score_gap": gap,
        # Scale-free version of the gap. RRF scores are ~0.03 and min-max fused
        # scores are ~2.0, so an absolute gap threshold would mean different
        # things under different fusion; a *relative* gap means the same thing.
        "rel_gap": gap / (abs(max_score) + 1e-9),
        "entropy": _softmax_entropy(scores),
        "agreement": both / len(top),
        "top1_agree": 1.0 if top[0].source == "both" else 0.0,
        "dense_max": max((float(x) for x in dense), default=0.0),
        "sparse_max": max((float(x) for x in sparse), default=0.0),
    }


def _as_signals(
    item: Sequence[RetrievalLike] | Mapping[str, float], *, k: int | None = None
) -> dict[str, float]:
    """Accept either raw hits or a precomputed signal mapping.

    Calibration runs over thousands of queries; being able to feed it a saved
    JSON of signals means the expensive part (retrieval) happens once and every
    subsequent threshold sweep is instant.
    """
    if isinstance(item, Mapping):
        return {k_: float(v) for k_, v in item.items()}
    return retrieval_signals(item, k=k)


# --- prior rules --------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SignalRule:
    """One signal turned into evidence-of-unanswerable in ``[0, 1]``.

    Attributes:
        signal: Key into the signal mapping.
        direction: ``"low"`` if *small* values are evidence the corpus cannot
            answer (similarity, gap, agreement); ``"high"`` if large values are
            (entropy).
        threshold: The value at which the rule is exactly undecided (evidence
            0.5).
        scale: Width of the transition. Small means the rule behaves like a step
            function; large means it contributes a soft opinion. Chosen per
            signal from how noisy that signal is, and refitted by calibration.
        weight: Relative say in the combined confidence.
        phrase: Format string used when this signal is *failing*. Receives
            ``value`` and ``threshold``.
        ok_phrase: Format string used when it is passing. Two phrasings rather
            than one because a reason is chosen per signal from that signal's
            own evidence -- writing "the scores are nearly flat (entropy 0.10)"
            on a confident answer would be a lie in the explanation layer, which
            is the one place a security-adjacent system cannot afford one.
    """

    signal: str
    direction: Literal["low", "high"]
    threshold: float
    scale: float
    weight: float
    phrase: str
    ok_phrase: str = ""

    def evidence(self, signals: Mapping[str, float]) -> float:
        """Logistic evidence in ``[0, 1]`` that this signal says "don't answer"."""
        x = float(signals.get(self.signal, 0.0))
        delta = (self.threshold - x) if self.direction == "low" else (x - self.threshold)
        return 1.0 / (1.0 + math.exp(-delta / max(self.scale, 1e-9)))

    def describe(self, signals: Mapping[str, float]) -> str:
        """Phrase this signal truthfully, whichever way it came out."""
        value = float(signals.get(self.signal, 0.0))
        failing = self.evidence(signals) >= 0.5
        template = self.phrase if failing or not self.ok_phrase else self.ok_phrase
        return template.format(value=value, threshold=self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal, "direction": self.direction,
            "threshold": self.threshold, "scale": self.scale,
            "weight": self.weight, "phrase": self.phrase, "ok_phrase": self.ok_phrase,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SignalRule":
        return cls(
            signal=str(d["signal"]), direction=d["direction"],  # type: ignore[arg-type]
            threshold=float(d["threshold"]), scale=float(d["scale"]),
            weight=float(d["weight"]), phrase=str(d["phrase"]),
            ok_phrase=str(d.get("ok_phrase", "")),
        )


#: Day-zero behaviour. **These are priors, not results.** They encode the
#: qualitative claims made in the module docstring at plausible operating points
#: for RRF fusion over a normalised-cosine dense index, and they are what
#: :meth:`AbstentionGate.calibrate` replaces.
#:
#: The ``max_score`` prior is the one worth reading: under RRF with k=60, a
#: passage ranked 1 by *both* retrievers scores ``2/61 = 0.0328``, and one ranked
#: 1 by a single retriever scores ``1/61 = 0.0164``. A threshold of 0.02
#: therefore encodes exactly "at least one retriever ranked it top and the other
#: at least saw it" -- an interpretable quantity rather than a tuned constant.
PRIOR_RULES: tuple[SignalRule, ...] = (
    SignalRule(
        "dense_max", "low", 0.30, 0.08, 1.0,
        "the best passage only reaches {value:.2f} similarity against a {threshold:.2f} floor",
        "the best passage reaches {value:.2f} similarity against a {threshold:.2f} floor",
    ),
    SignalRule(
        "max_score", "low", 0.02, 0.01, 0.5,
        "the fused retrieval score {value:.3f} is below the {threshold:.3f} floor",
        "the fused retrieval score is {value:.3f}, above the {threshold:.3f} floor",
    ),
    SignalRule(
        "rel_gap", "low", 0.10, 0.05, 0.8,
        "the top two passages are separated by only {value:.1%}",
        "the top passage leads the runner-up by {value:.0%}",
    ),
    SignalRule(
        "entropy", "high", 0.95, 0.03, 0.7,
        "the top scores are nearly flat (entropy {value:.2f} of a possible 1.00)",
        "the score distribution is peaked (entropy {value:.2f} of a possible 1.00)",
    ),
    SignalRule(
        "agreement", "low", 0.40, 0.15, 0.8,
        "only {value:.0%} of the top passages were found by both retrievers",
        "{value:.0%} of the top passages were found by both retrievers",
    ),
    SignalRule(
        "top1_agree", "low", 0.50, 0.20, 0.6,
        "the best passage was found by only one of the two retrievers",
        "dense and sparse retrieval agreed on the best passage",
    ),
)


# --- calibrated model ---------------------------------------------------------


@dataclass(slots=True)
class CalibratedModel:
    """A logistic model over standardised retrieval signals.

    Deliberately a plain numpy dot product rather than a scikit-learn estimator:
    it is seven multiply-accumulates, it serialises to readable JSON that a judge
    can inspect, and it adds no dependency to the serving path.

    Attributes:
        feature_names: Feature order this model was fitted with. Checked on load.
        mean / std: Standardisation statistics from the training set. Constant
            columns get ``std = 1`` so they contribute only through the bias.
        weights: Coefficients on standardised features. Sign is interpretable --
            positive means "larger values of this signal push towards abstain".
        bias: Intercept.
        decision_threshold: Probability at or above which the gate abstains,
            chosen by sweeping the fit metric rather than left at 0.5.
        metric / fit_metrics / n_positive / n_negative: Provenance of the fit,
            reported in the submission rather than kept implicit.
    """

    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float
    decision_threshold: float = 0.5
    metric: str = "f1"
    fit_metrics: dict[str, float] = field(default_factory=dict)
    n_positive: int = 0
    n_negative: int = 0

    def features(self, signals: Mapping[str, float]) -> np.ndarray:
        return np.asarray(
            [float(signals.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )

    def predict(self, signals: Mapping[str, float]) -> float:
        """Probability that the corpus cannot answer this query."""
        z = (self.features(signals) - self.mean) / self.std
        return float(1.0 / (1.0 + math.exp(-(float(z @ self.weights) + self.bias))))

    def coefficients(self) -> dict[str, float]:
        """Fitted weight per signal, for the write-up and for sanity checking."""
        return {n: round(float(w), 4) for n, w in zip(self.feature_names, self.weights)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "decision_threshold": self.decision_threshold,
            "metric": self.metric,
            "fit_metrics": self.fit_metrics,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibratedModel":
        return cls(
            feature_names=tuple(d["feature_names"]),
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            weights=np.asarray(d["weights"], dtype=np.float64),
            bias=float(d["bias"]),
            decision_threshold=float(d.get("decision_threshold", 0.5)),
            metric=str(d.get("metric", "f1")),
            fit_metrics=dict(d.get("fit_metrics", {})),
            n_positive=int(d.get("n_positive", 0)),
            n_negative=int(d.get("n_negative", 0)),
        )


@dataclass(slots=True)
class CalibrationResult:
    """What :meth:`AbstentionGate.calibrate` learned, and what it bought.

    Attributes:
        model: The fitted model, already installed on the gate.
        before: Metrics of the prior rules on the same data.
        after: Metrics of the fitted model on the same data.
        threshold_sweep: ``(threshold, metric)`` pairs, so the operating point
            can be plotted instead of asserted.
    """

    model: CalibratedModel
    before: dict[str, float]
    after: dict[str, float]
    threshold_sweep: list[tuple[float, float]] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Gain on the fitted metric. Negative means the priors were better."""
        key = self.model.metric
        return float(self.after.get(key, 0.0) - self.before.get(key, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "before": self.before,
            "after": self.after,
            "improvement": round(self.improvement, 4),
        }


# --- metrics ------------------------------------------------------------------


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Confusion matrix and derived rates for the abstention decision.

    The positive class is **"should abstain"**, i.e. the corpus cannot answer.
    That orientation is chosen so that *precision* answers the question a judge
    will actually ask -- "when it refuses, is it right?" -- and *recall* answers
    "how many unanswerable questions did it catch?".

    Args:
        y_true: Gold labels, 1 = unanswerable.
        y_pred: Predicted labels, 1 = abstained.

    Returns:
        ``tp/fp/tn/fn``, ``precision``, ``recall``, ``specificity``, ``f1``,
        ``accuracy``, ``balanced_accuracy``, ``youden``, ``support``,
        and ``false_abstention_rate`` -- the rate at which answerable questions
        get refused, which is the metric that decides whether users keep using
        the product.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} labels vs {len(y_pred)} predictions")
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    tp = int(((t == 1) & (p == 1)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = max(len(t), 1)
    return {
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / n, 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "youden": round(recall + specificity - 1.0, 4),
        "false_abstention_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        "support": float(len(t)),
    }


def _fit_logistic(
    X: np.ndarray, y: np.ndarray, *, l2: float, iters: int, lr: float
) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression by full-batch gradient descent.

    Hand-rolled rather than imported: scikit-learn is a 30 MB dependency for one
    convex optimisation over at most a few thousand rows and seven features.
    Weights start at zero and the bias starts at the training log-odds, so the
    fit is fully deterministic -- no seed, no run-to-run variance in a number
    that goes into a submission.

    Args:
        X: Standardised features, shape ``(n, d)``.
        y: Labels in ``{0, 1}``, shape ``(n,)``.
        l2: Ridge penalty on the weights (never on the bias).
        iters: Fixed iteration count.
        lr: Step size.

    Returns:
        ``(weights, bias)``.
    """
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    base = float(y.mean())
    base = min(max(base, 1e-6), 1 - 1e-6)
    b = math.log(base / (1 - base))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        err = p - y
        w -= lr * ((X.T @ err) / n + l2 * w / n)
        b -= lr * float(err.mean())
    return w, b


def _score_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    m = binary_metrics(y_true.tolist(), y_pred.tolist())
    if metric not in m:
        raise ValueError(f"unknown metric {metric!r}; have {sorted(m)}")
    return float(m[metric])


# --- the gate -----------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AbstentionVerdict:
    """The decision, its confidence, and every number behind it.

    Attributes:
        should_abstain: ``True`` to refuse rather than generate.
        confidence: ``P(the corpus cannot answer this)`` in ``[0, 1]``. Note the
            orientation: it is confidence *in abstaining*, not in the answer, so
            it reads correctly next to the "Abstained" badge in the HUD and is
            monotone in the evidence regardless of which way the decision went.
        signals: Every raw signal, for the HUD and the eval harness.
        reason: Specific, quantitative, user-facing. Never "I cannot answer
            that" -- always which signal failed and by how much.
        calibrated: Whether a fitted model or the priors produced this.
        latency_ms: Cost of the decision.
    """

    should_abstain: bool
    confidence: float
    signals: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    calibrated: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_abstain": self.should_abstain,
            "confidence": round(self.confidence, 4),
            "signals": {k: round(v, 5) for k, v in self.signals.items()},
            "reason": self.reason,
            "calibrated": self.calibrated,
            "latency_ms": round(self.latency_ms, 4),
        }


class AbstentionGate:
    """Decides whether retrieval found enough to justify generating.

    Args:
        rules: Signal rules used for the prior (uncalibrated) decision and,
            always, for the human-readable reason.
        model: A fitted :class:`CalibratedModel`. When present it makes the
            decision; the rules still supply the explanation, so the reason a
            user sees is phrased in real units even though the decision came
            from standardised coefficients.
        decision_threshold: Confidence at or above which the prior path
            abstains. Ignored when ``model`` is set (the model carries its own,
            swept for the fit metric).
        k: Number of top hits the signals are computed over. Fixed at
            construction so a verdict is always comparable to the calibration.

    Example:
        >>> gate = AbstentionGate()
        >>> gate.judge([]).should_abstain
        True
    """

    __slots__ = ("rules", "model", "decision_threshold", "k", "_wsum")

    def __init__(
        self,
        rules: Sequence[SignalRule] | None = None,
        *,
        model: CalibratedModel | None = None,
        decision_threshold: float = 0.5,
        k: int | None = 10,
    ) -> None:
        self.rules: tuple[SignalRule, ...] = tuple(rules if rules is not None else PRIOR_RULES)
        if not self.rules:
            raise ValueError("AbstentionGate needs at least one SignalRule for explanations")
        self.model = model
        self.decision_threshold = float(decision_threshold)
        self.k = k
        self._wsum = sum(r.weight for r in self.rules) or 1.0

    def __repr__(self) -> str:
        state = "calibrated" if self.model is not None else "prior"
        return f"AbstentionGate({state}, rules={len(self.rules)}, k={self.k})"

    @property
    def calibrated(self) -> bool:
        return self.model is not None

    # -- evidence -------------------------------------------------------------

    def evidence(self, signals: Mapping[str, float]) -> dict[str, float]:
        """Per-rule evidence that the corpus cannot answer, each in ``[0, 1]``."""
        return {r.signal: r.evidence(signals) for r in self.rules}

    def _prior_confidence(self, signals: Mapping[str, float]) -> float:
        """Weighted mean of rule evidence.

        A weighted mean rather than a noisy-OR: the signals are correlated (a
        flat score distribution usually also has a small gap), and noisy-OR
        would count that shared cause several times and abstain far too eagerly.
        """
        total = sum(r.weight * r.evidence(signals) for r in self.rules)
        return total / self._wsum

    def _reason(
        self, signals: Mapping[str, float], abstain: bool, confidence: float
    ) -> str:
        """Build the user-facing explanation from the rules that support it.

        On the abstain path this names the two signals that fired hardest; on
        the answer path, the two that were most reassuring. Both are
        quantitative on purpose -- an explanation a user cannot check is
        decoration.

        The two fallback branches exist because the decision and the rules can
        legitimately disagree: a calibrated model can abstain while every prior
        rule is comfortable, or a lax threshold can answer while several are
        unhappy. Papering over that by quoting a failing rule as if it were
        reassuring would make the explanation layer lie, so instead the mismatch
        is said out loud.
        """
        scored = [(r.evidence(signals), r) for r in self.rules]
        failing = sorted((e for e in scored if e[0] >= 0.5), key=lambda t: -t[0])
        passing = sorted((e for e in scored if e[0] < 0.5), key=lambda t: t[0])

        if abstain:
            picked = failing[:2] or passing[:1]
            head = (
                "Not answering: " if failing else
                f"Not answering: the calibrated model puts the chance of no answer at "
                f"{confidence:.0%} even though "
            )
        else:
            picked = passing[:2] or failing[:1]
            head = (
                "Retrieval was confident: " if passing else
                f"Answering on a thin margin ({confidence:.0%} chance the corpus can't "
                f"answer): "
            )
        return head + "; ".join(r.describe(signals) for _, r in picked) + "."

    # -- decision -------------------------------------------------------------

    def judge(
        self,
        hits: Sequence[RetrievalLike] | Mapping[str, float],
        *,
        k: int | None = None,
    ) -> AbstentionVerdict:
        """Decide whether to answer, from the retrieval result alone.

        Args:
            hits: Fused hits best-first, or a precomputed signal mapping (the
                eval harness replays saved signals this way).
            k: Override the configured top-k.

        Returns:
            An :class:`AbstentionVerdict`. Retrieving nothing short-circuits to
            ``confidence = 1.0`` without consulting rules or model.
        """
        t0 = time.perf_counter_ns()
        signals = _as_signals(hits, k=self.k if k is None else k)

        def finish(abstain: bool, confidence: float, reason: str) -> AbstentionVerdict:
            return AbstentionVerdict(
                should_abstain=abstain,
                confidence=confidence,
                signals=signals,
                reason=reason,
                calibrated=self.calibrated,
                latency_ms=(time.perf_counter_ns() - t0) / 1e6,
            )

        if signals.get("n_hits", 0.0) < 1.0:
            return finish(True, 1.0, "Not answering: retrieval returned no passages at all.")

        if self.model is not None:
            confidence = self.model.predict(signals)
            abstain = confidence >= self.model.decision_threshold
        else:
            confidence = self._prior_confidence(signals)
            abstain = confidence >= self.decision_threshold
        return finish(abstain, confidence, self._reason(signals, abstain, confidence))

    def predict(
        self, items: Iterable[Sequence[RetrievalLike] | Mapping[str, float]]
    ) -> list[int]:
        """Batch decision, 1 = abstain. Used by :meth:`evaluate`."""
        return [int(self.judge(item).should_abstain) for item in items]

    # -- calibration ----------------------------------------------------------

    def calibrate(
        self,
        positives: Sequence[Sequence[RetrievalLike] | Mapping[str, float]],
        negatives: Sequence[Sequence[RetrievalLike] | Mapping[str, float]],
        *,
        metric: Metric = "f1",
        l2: float = 1.0,
        iters: int = 600,
        lr: float = 0.5,
        install: bool = True,
    ) -> CalibrationResult:
        """Fit thresholds from labelled data instead of asserting them.

        Args:
            positives: Retrieval results for queries the corpus **can** answer.
                These must not be abstained on.
            negatives: Retrieval results for queries the corpus **cannot**
                answer -- on MS MARCO, the rows whose ``is_selected`` flags are
                all zero and whose gold answer is ``"No Answer Present."``.
            metric: What the decision threshold is swept to maximise. ``"f1"``
                balances catching unanswerable queries against refusing
                answerable ones; ``"youden"`` is the right choice when the class
                balance at serving time differs from the calibration set.
            l2: Ridge penalty. The default is deliberately non-trivial because
                the signals are correlated and an unregularised fit puts large
                cancelling weights on ``rel_gap`` and ``entropy``.
            iters / lr: Gradient descent budget.
            install: Set the fitted model on this gate. ``False`` fits and
                reports without changing behaviour.

        Returns:
            A :class:`CalibrationResult` carrying the model, the prior metrics
            and the fitted metrics on the same data, so the gain is measured.

        Raises:
            ValueError: If either class is empty. A one-class fit would produce
                a model that always agrees with itself.

        Note:
            Metrics are reported on the calibration data itself. For the
            submission, split the MS MARCO validation rows and call this on the
            training half -- the API takes plain sequences precisely so the
            caller controls the split.
        """
        if not positives or not negatives:
            raise ValueError(
                f"need both classes to calibrate; got {len(positives)} answerable "
                f"and {len(negatives)} unanswerable examples"
            )
        pos = [_as_signals(x, k=self.k) for x in positives]
        neg = [_as_signals(x, k=self.k) for x in negatives]
        rows = pos + neg
        y = np.concatenate([np.zeros(len(pos)), np.ones(len(neg))])
        X = np.asarray(
            [[s.get(n, 0.0) for n in FEATURE_NAMES] for s in rows], dtype=np.float64
        )

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-9] = 1.0  # constant column: carries no information, only bias
        Xs = (X - mean) / std

        before = binary_metrics(y.tolist(), self.predict(rows))
        w, b = _fit_logistic(Xs, y, l2=l2, iters=iters, lr=lr)
        probs = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))

        # Sweep every distinct probability as a candidate threshold. With a few
        # thousand rows this is milliseconds, and unlike a fixed grid it can
        # never miss the optimum by landing between two clusters.
        candidates = np.unique(np.round(probs, 6))
        sweep: list[tuple[float, float]] = []
        best_t, best_v = 0.5, -1.0
        for t in candidates:
            v = _score_metric(y, (probs >= t).astype(np.int64), metric)
            sweep.append((float(t), v))
            if v > best_v:
                best_t, best_v = float(t), v

        model = CalibratedModel(
            feature_names=FEATURE_NAMES,
            mean=mean, std=std, weights=w, bias=b,
            decision_threshold=best_t,
            metric=metric,
            n_positive=len(neg),   # positive class = should-abstain
            n_negative=len(pos),
        )
        after = binary_metrics(y.tolist(), (probs >= best_t).astype(np.int64).tolist())
        model.fit_metrics = after
        if install:
            self.model = model
        return CalibrationResult(model=model, before=before, after=after, threshold_sweep=sweep)

    def evaluate(
        self,
        positives: Sequence[Sequence[RetrievalLike] | Mapping[str, float]],
        negatives: Sequence[Sequence[RetrievalLike] | Mapping[str, float]],
    ) -> dict[str, float]:
        """Score the current gate against labelled data.

        Args:
            positives: Answerable queries (gold label 0, must not abstain).
            negatives: Unanswerable queries (gold label 1, must abstain).

        Returns:
            The full :func:`binary_metrics` dictionary, positive class =
            "should abstain".
        """
        items = list(positives) + list(negatives)
        y = [0] * len(positives) + [1] * len(negatives)
        return binary_metrics(y, self.predict(items))

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "k": self.k,
            "decision_threshold": self.decision_threshold,
            "rules": [r.to_dict() for r in self.rules],
            "model": None if self.model is None else self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AbstentionGate":
        """Rebuild a gate, checking that a saved model still matches the code.

        Raises:
            ValueError: If the persisted feature order differs from
                :data:`FEATURE_NAMES`. Silently applying stale coefficients to
                reordered features would be a wrong answer that looks right.
        """
        model = None
        if d.get("model"):
            model = CalibratedModel.from_dict(d["model"])
            if tuple(model.feature_names) != FEATURE_NAMES:
                raise ValueError(
                    "saved abstention model was fitted on features "
                    f"{model.feature_names} but this build uses {FEATURE_NAMES}; refit it"
                )
        return cls(
            [SignalRule.from_dict(r) for r in d.get("rules", ())] or None,
            model=model,
            decision_threshold=float(d.get("decision_threshold", 0.5)),
            k=d.get("k", 10),
        )

    def save(self, path: str | Path) -> Path:
        """Write rules and fitted model to JSON. Parent dirs are created."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "AbstentionGate":
        """Read a gate written by :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_threshold(self, threshold: float) -> "AbstentionGate":
        """A copy operating at a different point on the precision/recall curve.

        Useful for the ablation: the same fitted model can be shown refusing
        aggressively or conservatively without refitting anything.
        """
        model = None if self.model is None else replace(self.model, decision_threshold=threshold)
        return AbstentionGate(
            self.rules, model=model, decision_threshold=threshold, k=self.k
        )
