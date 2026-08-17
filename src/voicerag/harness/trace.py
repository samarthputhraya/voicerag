"""Span tracing.

Every latency number this project publishes comes from here. The benchmark, the
API response and the on-screen HUD all read the same spans, so there is exactly
one source of truth for "how long did that take" and no opportunity for the
report and the demo to disagree.

Timing uses `perf_counter_ns` (monotonic, nanosecond resolution). Wall-clock
timestamps are recorded separately and only for display -- never for arithmetic.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["Span", "Trace", "current_trace", "traced"]


@dataclass(slots=True)
class Span:
    """One timed unit of work."""

    name: str
    start_ns: int
    end_ns: int | None = None
    parent: str | None = None
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    attrs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ms(self) -> float:
        if self.end_ns is None:
            return 0.0
        return (self.end_ns - self.start_ns) / 1e6

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent": self.parent,
            "ms": round(self.ms, 3),
            "attrs": self.attrs,
            "error": self.error,
        }


class Trace:
    """A collection of spans for a single request.

    Spans may overlap. That is intentional and load-bearing: guardrails and
    speculative retrieval run concurrently with other stages, so a naive sum of
    span durations would overstate wall-clock. `total_ms` measures the envelope,
    and `critical_path_ms` measures the union of busy intervals -- report both,
    because the gap between them is exactly the value the concurrency buys.
    """

    __slots__ = ("trace_id", "spans", "_t0", "_stack", "meta")

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.spans: list[Span] = []
        self._t0 = time.perf_counter_ns()
        self._stack: list[str] = []
        self.meta: dict[str, Any] = {}

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        s = Span(
            name=name,
            start_ns=time.perf_counter_ns(),
            parent=self._stack[-1] if self._stack else None,
            attrs=dict(attrs),
        )
        self.spans.append(s)
        self._stack.append(s.span_id)
        try:
            yield s
        except BaseException as exc:
            s.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            s.end_ns = time.perf_counter_ns()
            self._stack.pop()

    def mark(self, name: str, **attrs: Any) -> None:
        """Record a zero-duration event (e.g. `vad.speech_end`)."""
        now = time.perf_counter_ns()
        self.spans.append(Span(name=name, start_ns=now, end_ns=now, attrs=dict(attrs)))

    @property
    def total_ms(self) -> float:
        """Wall-clock envelope from trace creation to the last span end."""
        if not self.spans:
            return (time.perf_counter_ns() - self._t0) / 1e6
        last = max((s.end_ns or s.start_ns) for s in self.spans)
        return (last - self._t0) / 1e6

    @property
    def critical_path_ms(self) -> float:
        """Union of busy intervals across top-level spans.

        Overlapping work is counted once, so this is the honest "time actually
        spent computing" figure rather than the inflated sum of all spans.
        """
        tops = [s for s in self.spans if s.parent is None and s.end_ns is not None]
        if not tops:
            return 0.0
        iv = sorted((s.start_ns, s.end_ns) for s in tops)  # type: ignore[misc]
        total, cur_s, cur_e = 0, *iv[0]
        for s, e in iv[1:]:
            if s > cur_e:
                total += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        total += cur_e - cur_s
        return total / 1e6

    def breakdown(self) -> dict[str, float]:
        """Milliseconds per span name, summed across repeats."""
        out: dict[str, float] = {}
        for s in self.spans:
            if s.end_ns is not None:
                out[s.name] = round(out.get(s.name, 0.0) + s.ms, 3)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "total_ms": round(self.total_ms, 3),
            "critical_path_ms": round(self.critical_path_ms, 3),
            "breakdown": self.breakdown(),
            "spans": [s.to_dict() for s in self.spans],
            "meta": self.meta,
        }


_CURRENT: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "voicerag_trace", default=None
)


def current_trace() -> Trace | None:
    return _CURRENT.get()


@contextmanager
def traced(trace: Trace | None = None) -> Iterator[Trace]:
    """Bind a trace to the current context (and any tasks spawned from it)."""
    t = trace or Trace()
    token = _CURRENT.set(t)
    try:
        yield t
    finally:
        _CURRENT.reset(token)
