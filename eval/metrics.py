"""IR metrics and latency percentiles, implemented from scratch.

Written out rather than pulled from ``pytrec_eval`` or ``ranx`` for two
reasons. First, every number here is checked against a hand-computed value in
``tests/test_eval.py``, so the definition a judge reads is the definition that
produced the table -- there is no version of a third-party library in between.
Second, it keeps the dependency surface at numpy, which matters for a project
whose whole test suite has to run offline.

**Percentiles use the nearest-rank method**, ``index = ceil(p/100 * n)`` over
the ascending sample, with no interpolation. That choice is stated here because
it is load-bearing: P70 is an unusual percentile, the report quotes it, and the
frontend HUD computes it independently. Nearest-rank always returns an
*observed* measurement rather than a synthetic average of two, so the number in
the README is a run that actually happened, and the demo and the report cannot
disagree. P100 is therefore exactly the maximum.

Conventions used by every ranking metric:

* ``ranked`` is a list of ids, best first, already deduplicated by the caller
  when documents are retrieved via multiple chunks
  (:func:`dedupe_preserving_order` does that).
* ``relevant`` is a set of gold ids, or a mapping id -> graded gain.
* ``k`` is truncation depth. A shorter ``ranked`` list is *not* padded; metrics
  are computed over what was actually returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_PERCENTILES",
    "LatencySummary",
    "QueryResult",
    "RetrievalSummary",
    "aggregate",
    "dcg",
    "dedupe_preserving_order",
    "evaluate_run",
    "hit_rate",
    "mrr_at_k",
    "ndcg_at_k",
    "percentile",
    "percentiles",
    "precision_at_k",
    "recall_at_k",
]

#: Percentiles reported everywhere. P70 is in the list because the organisers
#: ask for it; P100 because a latency claim that hides its worst case is not a
#: claim.
DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 70.0, 90.0, 95.0, 99.0, 100.0)


# --- helpers ------------------------------------------------------------------


def dedupe_preserving_order(ids: Iterable[str]) -> list[str]:
    """Drop repeats, keeping the first (best-ranked) occurrence.

    Retrieval returns *chunks*, but relevance is judged on *passages*, and one
    passage can contribute several chunks. Collapsing to first occurrence turns
    a chunk ranking into a passage ranking without inventing positions: if the
    same passage occupies ranks 1 and 2, the passage at rank 3 becomes rank 2,
    which is exactly how a user experiences the list.

    Args:
        ids: Ids, best first.

    Returns:
        The same ids with later duplicates removed.
    """
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _gains(relevant: Mapping[str, float] | Iterable[str]) -> dict[str, float]:
    """Normalise a gold set or graded mapping to ``id -> gain``."""
    if isinstance(relevant, Mapping):
        return {str(k): float(v) for k, v in relevant.items()}
    return {str(k): 1.0 for k in relevant}


def _truncate(ranked: Sequence[str], k: int) -> Sequence[str]:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return ranked[:k]


# --- ranking metrics ----------------------------------------------------------


def recall_at_k(
    ranked: Sequence[str], relevant: Mapping[str, float] | Iterable[str], k: int
) -> float:
    """Fraction of gold documents retrieved in the top ``k``.

    Args:
        ranked: Retrieved ids, best first.
        relevant: Gold ids (or graded mapping; grades are ignored here, only
            membership matters).
        k: Cutoff.

    Returns:
        ``|retrieved@k ∩ gold| / |gold|``, or ``0.0`` when there is no gold --
        a query with no relevant document cannot be recalled, and returning
        ``0`` keeps it from silently inflating a macro-average by scoring 1.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    gold = set(_gains(relevant))
    if not gold:
        return 0.0
    found = sum(1 for d in _truncate(ranked, k) if d in gold)
    return found / len(gold)


def precision_at_k(
    ranked: Sequence[str], relevant: Mapping[str, float] | Iterable[str], k: int
) -> float:
    """Fraction of the top ``k`` that is relevant.

    The denominator is ``k`` itself, not ``min(k, len(ranked))``: a system that
    returns three documents when asked for ten has not achieved higher
    precision by returning less, and using the shorter denominator would reward
    exactly that.

    Args:
        ranked: Retrieved ids, best first.
        relevant: Gold ids or graded mapping.
        k: Cutoff.

    Returns:
        Precision in ``[0, 1]``.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    gold = set(_gains(relevant))
    found = sum(1 for d in _truncate(ranked, k) if d in gold)
    return found / float(k)


def mrr_at_k(
    ranked: Sequence[str], relevant: Mapping[str, float] | Iterable[str], k: int = 10
) -> float:
    """Reciprocal rank of the first relevant document, 0 if none in top ``k``.

    Args:
        ranked: Retrieved ids, best first.
        relevant: Gold ids or graded mapping.
        k: Cutoff.

    Returns:
        ``1 / rank`` of the first hit (1-based), else ``0.0``.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    gold = set(_gains(relevant))
    for rank, doc in enumerate(_truncate(ranked, k), start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def hit_rate(
    ranked: Sequence[str], relevant: Mapping[str, float] | Iterable[str], k: int = 10
) -> float:
    """``1.0`` if any gold document appears in the top ``k``, else ``0.0``.

    Equal to Recall@k only when there is exactly one gold document, which is
    the common MS MARCO case -- they are reported separately anyway because the
    dataset does contain multi-gold queries and the two then diverge.
    """
    gold = set(_gains(relevant))
    return 1.0 if any(d in gold for d in _truncate(ranked, k)) else 0.0


def dcg(gains: Sequence[float], k: int | None = None) -> float:
    """Discounted cumulative gain with the standard ``log2(rank + 1)`` discount.

    Args:
        gains: Per-position gains, in rank order (position 1 first).
        k: Optional truncation.

    Returns:
        ``sum(gain_i / log2(i + 1))`` for 1-based ``i``. Position 1 has
        discount ``log2(2) = 1``, i.e. no discount, which is the convention
        used by TREC and by every published nDCG this table will be compared to.
    """
    seq = gains if k is None else gains[:k]
    return sum(g / math.log2(i + 1) for i, g in enumerate(seq, start=1) if g)


def ndcg_at_k(
    ranked: Sequence[str],
    relevant: Mapping[str, float] | Iterable[str],
    k: int = 10,
) -> float:
    """Normalised DCG at ``k``, binary or graded.

    Args:
        ranked: Retrieved ids, best first.
        relevant: Gold ids (gain 1 each) or a mapping id -> graded gain.
        k: Cutoff.

    Returns:
        ``DCG@k / IDCG@k`` in ``[0, 1]``; ``0.0`` when no gold exists or all
        gains are zero.

    The ideal ranking is the gold gains sorted descending and truncated to
    ``k`` -- **not** to ``len(ranked)``. Normalising by the achievable-given-
    what-was-returned ideal is a common bug that makes a system which returns
    two documents look perfect; here, failing to return the other eight gold
    documents costs exactly what it should.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    gains = _gains(relevant)
    if not gains:
        return 0.0
    got = [gains.get(d, 0.0) for d in _truncate(ranked, k)]
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg <= 0.0:
        return 0.0
    return dcg(got) / idcg


# --- percentiles --------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile: ``sorted[ceil(p/100 * n) - 1]``.

    Args:
        values: Samples, any order. Not mutated.
        p: Percentile in ``[0, 100]``.

    Returns:
        An observed sample. ``p <= 0`` yields the minimum and ``p = 100`` the
        maximum. Never interpolates, so the reported P50 is a run that
        happened.

    Raises:
        ValueError: If ``values`` is empty or ``p`` is outside ``[0, 100]``.
    """
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {p}")
    n = len(values)
    if n == 0:
        raise ValueError("percentile of an empty sample is undefined")
    ordered = sorted(values)
    rank = math.ceil(p / 100.0 * n)
    return float(ordered[min(max(rank, 1), n) - 1])


def percentiles(
    values: Sequence[float], ps: Sequence[float] = DEFAULT_PERCENTILES
) -> dict[str, float]:
    """Several nearest-rank percentiles at once.

    Args:
        values: Samples.
        ps: Percentiles to compute.

    Returns:
        ``{"p50": ..., "p70": ...}``. Keys drop a trailing ``.0`` so ``p50``
        rather than ``p50.0``; fractional percentiles keep their decimal.
        Returns an empty dict for an empty sample rather than raising, because
        a benchmark stage that never ran (an abstained query never generates)
        legitimately has no samples and should report nothing rather than zero.
    """
    if not values:
        return {}
    out: dict[str, float] = {}
    for p in ps:
        key = f"p{int(p)}" if float(p).is_integer() else f"p{p}"
        out[key] = round(percentile(values, p), 3)
    return out


@dataclass(slots=True, frozen=True)
class LatencySummary:
    """Percentile summary of one timed stage.

    Attributes:
        name: Stage label, e.g. ``"retrieve"`` or ``"pipeline_total"``.
        n: Number of samples behind the percentiles.
        mean: Arithmetic mean, in milliseconds.
        pcts: Nearest-rank percentiles, keyed ``p50``, ``p70``, ...
    """

    name: str
    n: int
    mean: float
    pcts: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_samples(
        cls,
        name: str,
        samples: Sequence[float],
        ps: Sequence[float] = DEFAULT_PERCENTILES,
    ) -> "LatencySummary":
        """Summarise raw millisecond samples."""
        mean = sum(samples) / len(samples) if samples else 0.0
        return cls(name=name, n=len(samples), mean=round(mean, 3), pcts=percentiles(samples, ps))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "n": self.n, "mean_ms": self.mean, **self.pcts}


# --- run-level aggregation ----------------------------------------------------


@dataclass(slots=True, frozen=True)
class QueryResult:
    """Per-query retrieval outcome, before aggregation.

    Keeping the per-query rows (rather than only the mean) is what makes a
    per-``query_type`` breakdown and a paired comparison between two ablation
    rows possible after the fact, without re-running retrieval.
    """

    query_id: int
    ranked: tuple[str, ...]
    relevant: frozenset[str]
    latency_ms: float = 0.0
    query_type: str = ""

    def metrics(self, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
        """Every ranking metric for this one query."""
        out: dict[str, float] = {}
        for k in ks:
            out[f"recall@{k}"] = recall_at_k(self.ranked, self.relevant, k)
            out[f"precision@{k}"] = precision_at_k(self.ranked, self.relevant, k)
            out[f"hit@{k}"] = hit_rate(self.ranked, self.relevant, k)
        deepest = max(ks)
        out[f"mrr@{deepest}"] = mrr_at_k(self.ranked, self.relevant, deepest)
        out[f"ndcg@{deepest}"] = ndcg_at_k(self.ranked, self.relevant, deepest)
        return out


@dataclass(slots=True, frozen=True)
class RetrievalSummary:
    """Macro-averaged retrieval quality plus query-latency percentiles.

    Attributes:
        n_queries: Queries scored.
        metrics: Macro means, keyed ``recall@5``, ``ndcg@10``, ...
        latency: Percentile summary of per-query retrieval latency.
        by_type: Macro means split by MS MARCO ``query_type``. Empty when the
            caller did not supply types.
    """

    n_queries: int
    metrics: dict[str, float]
    latency: LatencySummary
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            **self.metrics,
            "latency": self.latency.to_dict(),
            "by_type": self.by_type,
        }


def aggregate(
    results: Sequence[QueryResult],
    *,
    ks: Sequence[int] = (1, 5, 10),
    by_type: bool = True,
) -> RetrievalSummary:
    """Macro-average per-query metrics.

    Macro (mean over queries) rather than micro (pooled counts) because every
    query is one user asking one thing; a query with six gold passages should
    not outvote five queries with one.

    Args:
        results: Per-query outcomes.
        ks: Cutoffs to report.
        by_type: Also break the means down by ``query_type``.

    Returns:
        A :class:`RetrievalSummary`. An empty input yields zeroed metrics with
        ``n_queries = 0`` rather than raising, so an ablation row for a
        strategy that produced no chunks still appears in the table as a
        visible zero instead of vanishing.
    """
    if not results:
        return RetrievalSummary(0, {}, LatencySummary.from_samples("retrieve", []), {})

    rows = [r.metrics(ks) for r in results]
    keys = list(rows[0])
    means = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}

    grouped: dict[str, dict[str, float]] = {}
    if by_type:
        buckets: dict[str, list[dict[str, float]]] = {}
        for res, row in zip(results, rows):
            if res.query_type:
                buckets.setdefault(res.query_type, []).append(row)
        for qtype, group in sorted(buckets.items()):
            grouped[qtype] = {
                "n": float(len(group)),
                **{k: round(sum(r[k] for r in group) / len(group), 4) for k in keys},
            }

    return RetrievalSummary(
        n_queries=len(results),
        metrics=means,
        latency=LatencySummary.from_samples(
            "retrieve", [r.latency_ms for r in results]
        ),
        by_type=grouped,
    )


def evaluate_run(
    run: Mapping[int, Sequence[str]],
    qrels: Mapping[int, Iterable[str]],
    *,
    ks: Sequence[int] = (1, 5, 10),
    latencies: Mapping[int, float] | None = None,
    query_types: Mapping[int, str] | None = None,
) -> RetrievalSummary:
    """Score a TREC-style run against qrels.

    Args:
        run: ``query_id -> ranked ids``, best first.
        qrels: ``query_id -> gold ids``.
        ks: Cutoffs.
        latencies: Optional ``query_id -> milliseconds``.
        query_types: Optional ``query_id -> MS MARCO query type``.

    Returns:
        A :class:`RetrievalSummary` over the queries present in **qrels**.
        Queries in ``qrels`` but missing from ``run`` are scored as complete
        misses rather than skipped -- a retriever that crashes on hard queries
        must not be rewarded for their absence.
    """
    results = [
        QueryResult(
            query_id=qid,
            ranked=tuple(dedupe_preserving_order(run.get(qid, ()))),
            relevant=frozenset(gold),
            latency_ms=float((latencies or {}).get(qid, 0.0)),
            query_type=(query_types or {}).get(qid, ""),
        )
        for qid, gold in qrels.items()
    ]
    return aggregate(results, ks=ks)
