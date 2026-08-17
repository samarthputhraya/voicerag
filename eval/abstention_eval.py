"""Abstention evaluation against MS MARCO's own "No Answer Present." labels.

Most RAG demos prove their guardrail with a handful of hand-written questions
about the moon. That proves nothing: the examples were chosen by the same person
who chose the thresholds. MS MARCO ships a real alternative -- queries where
human annotators read all ten Bing-retrieved candidates and marked *none* of
them relevant, so the gold answer is the literal string
``"No Answer Present."``. Those rows are a labelled, adversarially-mined
(they are real user queries that a real search engine failed) set of
*the corpus cannot answer this*.

This module scores the abstention gate on them:

* **Positive class = "should abstain."** Precision then answers the question a
  judge actually asks -- "when it refuses, is it right?" -- and recall answers
  "how many unanswerable queries did it catch?". This matches the orientation
  in :func:`voicerag.guardrails.abstention.binary_metrics`, so the two never
  disagree.
* **False abstention rate is reported as a first-class number.** Refusing an
  answerable question is the failure mode that makes users stop using the
  product, and it is invisible in accuracy when the classes are imbalanced.
* **Calibration is fitted on a train split and reported on a held-out test
  split.** :meth:`~voicerag.guardrails.abstention.AbstentionGate.calibrate`
  reports metrics on its own fitting data by design; quoting those as the
  system's abstention quality would be reporting training accuracy. The split
  is stratified and seeded.

A naming trap worth stating once: the guardrail's ``calibrate``/``evaluate``
take ``(positives, negatives)`` meaning ``(answerable, unanswerable)``, while
the positive *class* in the metrics is "should abstain". This module uses the
unambiguous names ``answerable`` / ``unanswerable`` throughout and does the
mapping in exactly one place, :func:`_split_by_label`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .dataset import Query
from .metrics import percentile

__all__ = [
    "AbstentionEvalResult",
    "AbstentionExample",
    "calibrate_and_evaluate",
    "collect_examples",
    "confusion_matrix",
    "evaluate_abstention",
    "split_examples",
    "sweep_to_markdown",
    "threshold_sweep",
    "to_markdown",
]

#: Given a query string, return fused retrieval hits (best first). The
#: benchmark supplies a closure over a built index; a test supplies a stub.
RetrieveFn = Callable[[str], Sequence[Any]]


@dataclass(slots=True)
class AbstentionExample:
    """One labelled query with the retrieval signals it produced.

    Attributes:
        query_id: MS MARCO query id.
        query: The query text that was retrieved with.
        unanswerable: Gold label. ``True`` when the annotators marked every
            candidate passage non-relevant.
        signals: Output of
            :func:`voicerag.guardrails.abstention.retrieval_signals`. Stored
            rather than the hits themselves because a threshold sweep replays
            these thousands of times and re-running retrieval per threshold
            would make the sweep the slowest part of the evaluation.
        query_type: MS MARCO type label, for per-type breakdowns.
    """

    query_id: int
    query: str
    unanswerable: bool
    signals: dict[str, float] = field(default_factory=dict)
    query_type: str = ""

    @property
    def label(self) -> int:
        """1 when the gate *should* abstain."""
        return int(self.unanswerable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "unanswerable": self.unanswerable,
            "query_type": self.query_type,
            "signals": {k: round(v, 5) for k, v in self.signals.items()},
        }


def collect_examples(
    queries: Sequence[Query],
    retrieve: RetrieveFn,
    *,
    k: int = 10,
    query_field: str = "eng_query",
    progress: Callable[[str], None] | None = None,
) -> list[AbstentionExample]:
    """Retrieve for each query once and reduce the hits to abstention signals.

    Args:
        queries: Mixed answerable and unanswerable queries. The label is read
            from :attr:`~eval.dataset.Query.answerable`, which comes from
            ``is_selected`` and the ``"No Answer Present."`` sentinel -- never
            from anything this module decides.
        retrieve: Callable mapping query text to fused hits.
        k: Top-k the signals are computed over. Must match the gate's ``k`` or
            the calibration is measuring a different feature space.
        query_field: ``"eng_query"`` or ``"indic_query"``.
        progress: Status callback, called every 50 queries.

    Returns:
        One :class:`AbstentionExample` per query, retrieval run exactly once.
    """
    from voicerag.guardrails.abstention import retrieval_signals

    out: list[AbstentionExample] = []
    for i, query in enumerate(queries):
        text = query.eng_query if query_field == "eng_query" else query.indic_query
        hits = retrieve(text)
        out.append(
            AbstentionExample(
                query_id=query.query_id,
                query=text,
                unanswerable=not query.answerable,
                signals=retrieval_signals(hits, k=k),
                query_type=query.query_type,
            )
        )
        if progress and i % 50 == 0:
            progress(f"signals for {i + 1}/{len(queries)} queries")
    return out


def _split_by_label(
    examples: Sequence[AbstentionExample],
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Return ``(answerable_signals, unanswerable_signals)``.

    The single place that translates this module's explicit labels into the
    guardrail API's ``(positives, negatives)`` argument order.
    """
    answerable = [e.signals for e in examples if not e.unanswerable]
    unanswerable = [e.signals for e in examples if e.unanswerable]
    return answerable, unanswerable


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    """Counts with the positive class = "abstained".

    Args:
        y_true: 1 = the corpus cannot answer.
        y_pred: 1 = the gate abstained.

    Returns:
        ``{"tp", "fp", "tn", "fn"}``.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"length mismatch: {len(y_true)} labels vs {len(y_pred)} predictions"
        )
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


@dataclass(slots=True)
class AbstentionEvalResult:
    """Scored abstention performance on one labelled set.

    Attributes:
        metrics: Full :func:`~voicerag.guardrails.abstention.binary_metrics`
            dictionary, positive class = "should abstain".
        confusion: Raw counts.
        n_answerable / n_unanswerable: Class support.
        threshold: Decision threshold in force.
        calibrated: Whether a fitted model made the decisions.
        sweep: Threshold sweep rows, if computed.
        by_type: Metrics split by MS MARCO query type.
        errors: The misclassified examples, worst first by confidence, so the
            report can show *what* it got wrong rather than only how often.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    confusion: dict[str, int] = field(default_factory=dict)
    n_answerable: int = 0
    n_unanswerable: int = 0
    threshold: float = 0.5
    calibrated: bool = False
    sweep: list[dict[str, float]] = field(default_factory=list)
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "confusion": self.confusion,
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "threshold": round(self.threshold, 4),
            "calibrated": self.calibrated,
            "sweep": self.sweep,
            "by_type": self.by_type,
            "errors": self.errors,
        }

    def write_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p


def evaluate_abstention(
    examples: Sequence[AbstentionExample],
    gate: Any | None = None,
    *,
    sweep: bool = True,
    max_errors: int = 10,
) -> AbstentionEvalResult:
    """Score a gate on labelled examples.

    Args:
        examples: Labelled examples from :func:`collect_examples`.
        gate: An :class:`~voicerag.guardrails.abstention.AbstentionGate`.
            ``None`` builds the default (uncalibrated, prior-rule) gate, which
            is the right baseline: it shows what the hand-reasoned thresholds
            achieve before any fitting.
        sweep: Also compute the threshold sweep curve.
        max_errors: How many misclassified examples to keep in the result.

    Returns:
        An :class:`AbstentionEvalResult`.

    Raises:
        ValueError: If ``examples`` is empty.
    """
    from voicerag.guardrails.abstention import AbstentionGate, binary_metrics

    if not examples:
        raise ValueError("abstention evaluation needs at least one example")
    g = gate if gate is not None else AbstentionGate()

    verdicts = [g.judge(e.signals) for e in examples]
    y_true = [e.label for e in examples]
    y_pred = [int(v.should_abstain) for v in verdicts]

    errors = sorted(
        (
            {
                "query_id": e.query_id,
                "query": e.query,
                "gold_unanswerable": e.unanswerable,
                "abstained": bool(p),
                "confidence": round(v.confidence, 4),
                "reason": v.reason,
                "signals": {k: round(x, 4) for k, x in e.signals.items()},
            }
            for e, v, p in zip(examples, verdicts, y_pred)
            if p != e.label
        ),
        # Most confident mistakes first: a wrong decision made with high
        # confidence is the one that reveals a broken feature, whereas a wrong
        # decision at 0.51 is just the threshold doing its job.
        key=lambda d: -abs(d["confidence"] - 0.5),
    )[:max_errors]

    by_type: dict[str, dict[str, float]] = {}
    types = {e.query_type for e in examples if e.query_type}
    for qtype in sorted(types):
        idx = [i for i, e in enumerate(examples) if e.query_type == qtype]
        by_type[qtype] = {
            "n": float(len(idx)),
            **binary_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx]),
        }

    threshold = (
        g.model.decision_threshold if getattr(g, "model", None) is not None
        else g.decision_threshold
    )
    return AbstentionEvalResult(
        metrics=binary_metrics(y_true, y_pred),
        confusion=confusion_matrix(y_true, y_pred),
        n_answerable=sum(1 for e in examples if not e.unanswerable),
        n_unanswerable=sum(1 for e in examples if e.unanswerable),
        threshold=float(threshold),
        calibrated=bool(getattr(g, "calibrated", False)),
        sweep=threshold_sweep(examples, g) if sweep else [],
        by_type=by_type,
        errors=errors,
    )


def threshold_sweep(
    examples: Sequence[AbstentionExample],
    gate: Any | None = None,
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    """Precision/recall/F1 as a function of the decision threshold.

    The curve, not the single operating point, is what shows whether the gate
    has any discriminative power: a gate whose F1 is flat across thresholds is
    not separating the classes, it is guessing at a lucky rate.

    Args:
        examples: Labelled examples.
        gate: Gate whose confidence scores are swept. ``None`` uses the default.
        thresholds: Explicit thresholds. ``None`` derives them from the observed
            confidence distribution -- the deciles of the actual scores plus a
            fixed 0.1 grid -- so the sweep cannot miss the operating region by
            landing between two clusters.

    Returns:
        One row per threshold with ``threshold``, ``precision``, ``recall``,
        ``f1``, ``accuracy``, ``false_abstention_rate``, sorted ascending.
    """
    from voicerag.guardrails.abstention import AbstentionGate, binary_metrics

    g = gate if gate is not None else AbstentionGate()
    confidences = [g.judge(e.signals).confidence for e in examples]
    y_true = [e.label for e in examples]

    if thresholds is None:
        grid = {round(x / 10, 2) for x in range(1, 10)}
        grid.update(
            round(percentile(confidences, p), 4) for p in range(5, 100, 5)
        )
        thresholds = sorted(grid)

    rows: list[dict[str, float]] = []
    for t in thresholds:
        y_pred = [int(c >= t) for c in confidences]
        m = binary_metrics(y_true, y_pred)
        rows.append(
            {
                "threshold": round(float(t), 4),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "accuracy": m["accuracy"],
                "false_abstention_rate": m["false_abstention_rate"],
            }
        )
    return rows


def split_examples(
    examples: Sequence[AbstentionExample],
    *,
    test_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[AbstentionExample], list[AbstentionExample]]:
    """Stratified train/test split.

    Stratified because the unanswerable class is the minority (roughly 1 row in
    8 on MS MARCO validation) and an unstratified split of a few hundred
    examples can easily leave the test half with almost none, producing an F1
    that swings on two examples.

    Args:
        examples: Labelled examples.
        test_fraction: Share held out.
        seed: RNG seed; the split is a pure function of it.

    Returns:
        ``(train, test)``.

    Raises:
        ValueError: If ``test_fraction`` is not strictly between 0 and 1.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    rng = random.Random(seed)
    train: list[AbstentionExample] = []
    test: list[AbstentionExample] = []
    for label in (0, 1):
        group = [e for e in examples if e.label == label]
        rng.shuffle(group)
        cut = int(round(len(group) * test_fraction))
        test.extend(group[:cut])
        train.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def calibrate_and_evaluate(
    examples: Sequence[AbstentionExample],
    *,
    test_fraction: float = 0.4,
    seed: int = 0,
    metric: str = "f1",
    k: int | None = 10,
) -> dict[str, Any]:
    """Fit thresholds on a train split, report on the held-out test split.

    Args:
        examples: Labelled examples covering both classes.
        test_fraction: Held-out share.
        seed: Split seed.
        metric: What the threshold sweep inside calibration maximises.
        k: Top-k the gate scores over; must match ``collect_examples``.

    Returns:
        A dict with ``prior`` (default thresholds, test split), ``calibrated``
        (fitted model, test split), ``train`` sizes and the fitted
        coefficients. Reporting the prior *and* the calibrated numbers on the
        same held-out data is the only way to show that calibration earned its
        complexity.

    Raises:
        ValueError: If either class is missing from the training split.
    """
    from voicerag.guardrails.abstention import AbstentionGate

    train, test = split_examples(examples, test_fraction=test_fraction, seed=seed)
    answerable, unanswerable = _split_by_label(train)
    if not answerable or not unanswerable:
        raise ValueError(
            f"calibration needs both classes in the training split; got "
            f"{len(answerable)} answerable and {len(unanswerable)} unanswerable"
        )

    prior_gate = AbstentionGate(k=k)
    prior = evaluate_abstention(test, prior_gate, sweep=False)

    fitted_gate = AbstentionGate(k=k)
    result = fitted_gate.calibrate(answerable, unanswerable, metric=metric)
    calibrated = evaluate_abstention(test, fitted_gate, sweep=True)

    return {
        "n_train": len(train),
        "n_test": len(test),
        "train_answerable": len(answerable),
        "train_unanswerable": len(unanswerable),
        "metric": metric,
        "prior": prior.to_dict(),
        "calibrated": calibrated.to_dict(),
        "coefficients": result.model.coefficients(),
        "fitted_threshold": round(result.model.decision_threshold, 4),
        "train_improvement": round(result.improvement, 4),
    }


# --- rendering ----------------------------------------------------------------


def to_markdown(
    result: AbstentionEvalResult,
    *,
    title: str = "Abstention (MS MARCO 'No Answer Present.' labels)",
    notes: Sequence[str] = (),
) -> str:
    """Render one evaluation as markdown, confusion matrix included."""
    m = result.metrics
    c = result.confusion
    out = [f"### {title}", ""]
    out.extend(f"{n}  " for n in notes)
    if notes:
        out.append("")
    out.append(
        f"{result.n_unanswerable} unanswerable and {result.n_answerable} answerable "
        f"queries; decision threshold {result.threshold:.3f} "
        f"({'calibrated' if result.calibrated else 'prior rules'}). "
        "Positive class = *should abstain*."
    )
    out.extend(
        [
            "",
            "| Precision | Recall | F1 | Accuracy | Balanced acc. | False-abstention rate |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {m.get('precision', 0):.3f} | {m.get('recall', 0):.3f} | "
            f"{m.get('f1', 0):.3f} | {m.get('accuracy', 0):.3f} | "
            f"{m.get('balanced_accuracy', 0):.3f} | {m.get('false_abstention_rate', 0):.3f} |",
            "",
            "**Confusion matrix**",
            "",
            "| | predicted answer | predicted abstain |",
            "|---|---:|---:|",
            f"| **gold answerable** | {c.get('tn', 0)} | {c.get('fp', 0)} |",
            f"| **gold unanswerable** | {c.get('fn', 0)} | {c.get('tp', 0)} |",
            "",
        ]
    )
    if result.errors:
        out.extend(["**Most confident mistakes**", ""])
        for e in result.errors[:5]:
            kind = "refused an answerable query" if not e["gold_unanswerable"] else "answered an unanswerable query"
            out.append(f"- `{e['query'][:70]}` -- {kind} at confidence {e['confidence']:.2f}")
        out.append("")
    return "\n".join(out)


def sweep_to_markdown(rows: Sequence[Mapping[str, float]], *, limit: int = 12) -> str:
    """Render a threshold sweep as a markdown table, thinned to ``limit`` rows."""
    if not rows:
        return "_No sweep rows._\n"
    step = max(1, len(rows) // limit)
    picked = list(rows)[::step]
    out = [
        "| Threshold | Precision | Recall | F1 | Accuracy | False-abstention |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in picked:
        out.append(
            f"| {r['threshold']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['accuracy']:.3f} | {r['false_abstention_rate']:.3f} |"
        )
    return "\n".join(out) + "\n"
