"""Timeouts, retries, circuit breaking and provider fallback.

The brief asks for "a real harness -- retries, structured I/O, error recovery",
and specifically not a raw prompt-and-hope call. The rules encoded here:

1. **Every remote call has a deadline.** A hung socket must never become a hung
   request; in a 200 ms pipeline a 30 s default timeout is indistinguishable
   from an outage.
2. **Retries are budgeted, not counted.** Under a latency SLO the question is
   never "how many attempts" but "how much wall-clock is left". A retry that
   cannot finish before the deadline is not attempted at all.
3. **Only retry what is retryable.** A 400 will fail identically the second
   time; retrying it just burns the budget that a real recovery needed.
4. **Break the circuit before the user feels it.** A provider that has failed
   repeatedly is skipped outright, so the fallback path pays no penalty for the
   dead primary.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence, TypeVar

from .trace import current_trace

T = TypeVar("T")

__all__ = [
    "TransientError",
    "PermanentError",
    "BudgetExhausted",
    "AllProvidersFailed",
    "RetryPolicy",
    "CircuitBreaker",
    "CircuitOpen",
    "Deadline",
    "call_with_retry",
    "first_healthy",
]


# --- error taxonomy -----------------------------------------------------------
# Callers classify their own failures. Guessing from exception type is how
# harnesses end up retrying 401s forever.

class TransientError(Exception):
    """Worth retrying: timeout, 429, 5xx, connection reset."""


class PermanentError(Exception):
    """Not worth retrying: 4xx (bar 429), malformed request, auth failure."""


class BudgetExhausted(TransientError):
    """The latency budget ran out before the call could complete."""


class CircuitOpen(TransientError):
    """The provider is currently marked unhealthy and was skipped."""


class AllProvidersFailed(Exception):
    """Every provider in the fallback chain failed."""

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        detail = "; ".join(f"{k}: {v}" for k, v in errors.items())
        super().__init__(f"all providers failed -- {detail}")


# --- deadline -----------------------------------------------------------------

class Deadline:
    """A monotonic wall-clock budget shared across every stage of a request."""

    __slots__ = ("_end", "budget_ms")

    def __init__(self, budget_ms: float) -> None:
        self.budget_ms = budget_ms
        self._end = time.perf_counter_ns() + int(budget_ms * 1e6)

    @property
    def remaining_ms(self) -> float:
        return max(0.0, (self._end - time.perf_counter_ns()) / 1e6)

    @property
    def expired(self) -> bool:
        return self.remaining_ms <= 0.0

    def slice_ms(self, want_ms: float) -> float:
        """The largest timeout not exceeding both `want_ms` and what is left."""
        return min(want_ms, self.remaining_ms)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Deadline(remaining={self.remaining_ms:.1f}ms/{self.budget_ms:.0f}ms)"


# --- retry --------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter.

    Full jitter (sleep uniformly in ``[0, backoff]`` rather than exactly
    ``backoff``) is deliberate: with a fixed backoff, every client that failed
    during an outage retries in lockstep and re-creates the thundering herd that
    caused it.

    Args:
        attempts: Maximum total attempts, including the first.
        base_ms: Backoff for the first retry.
        max_ms: Ceiling for any single backoff.
        timeout_ms: Per-attempt timeout.
        jitter: Apply full jitter. Disable only in tests.
    """

    attempts: int = 3
    base_ms: float = 20.0
    max_ms: float = 200.0
    timeout_ms: float = 1_000.0
    jitter: bool = True

    def backoff_ms(self, attempt: int) -> float:
        raw = min(self.max_ms, self.base_ms * (2 ** attempt))
        return random.uniform(0.0, raw) if self.jitter else raw


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    deadline: Deadline | None = None,
    name: str = "call",
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Invoke `fn` with timeout, bounded retries and deadline awareness.

    Raises:
        PermanentError: propagated immediately, never retried.
        BudgetExhausted: the deadline expired before a viable attempt.
        TransientError: the last transient failure, once attempts run out.
    """
    pol = policy or RetryPolicy()
    trace = current_trace()
    last: BaseException | None = None

    for attempt in range(pol.attempts):
        if deadline is not None and deadline.expired:
            raise BudgetExhausted(f"{name}: budget exhausted before attempt {attempt + 1}")

        timeout_ms = pol.timeout_ms
        if deadline is not None:
            timeout_ms = deadline.slice_ms(timeout_ms)
            if timeout_ms <= 1.0:
                raise BudgetExhausted(f"{name}: <1ms of budget left")

        try:
            if trace is not None:
                with trace.span(f"{name}.attempt", attempt=attempt + 1):
                    return await asyncio.wait_for(fn(), timeout=timeout_ms / 1000.0)
            return await asyncio.wait_for(fn(), timeout=timeout_ms / 1000.0)

        except PermanentError:
            raise
        except (asyncio.TimeoutError, TransientError, ConnectionError, OSError) as exc:
            last = exc
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            if attempt == pol.attempts - 1:
                break
            sleep_ms = pol.backoff_ms(attempt)
            if deadline is not None:
                # Never sleep past the deadline, and never sleep away an attempt
                # we could still have made.
                sleep_ms = min(sleep_ms, max(0.0, deadline.remaining_ms - 2.0))
            if sleep_ms > 0:
                await asyncio.sleep(sleep_ms / 1000.0)

    raise TransientError(f"{name}: failed after {pol.attempts} attempts") from last


# --- circuit breaker ----------------------------------------------------------

@dataclass(slots=True)
class CircuitBreaker:
    """Three-state breaker: closed -> open -> half-open -> closed.

    After `threshold` consecutive failures the circuit opens and calls are
    rejected immediately for `reset_ms`. It then admits a single probe; success
    closes it, failure re-opens it. This keeps a dead provider from adding its
    full timeout to every request.

    Args:
        name: Provider identifier, used in errors and metrics.
        threshold: Consecutive failures that trip the breaker.
        reset_ms: How long to stay open before probing.
    """

    name: str
    threshold: int = 4
    reset_ms: float = 5_000.0
    _failures: int = field(default=0, init=False)
    _opened_ns: int | None = field(default=None, init=False)
    _half_open: bool = field(default=False, init=False)

    @property
    def state(self) -> str:
        if self._opened_ns is None:
            return "closed"
        elapsed = (time.perf_counter_ns() - self._opened_ns) / 1e6
        return "half_open" if elapsed >= self.reset_ms else "open"

    def allows(self) -> bool:
        st = self.state
        if st == "closed":
            return True
        if st == "half_open":
            self._half_open = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_ns = None
        self._half_open = False

    def record_failure(self) -> None:
        if self._half_open:
            # The probe failed: re-open for another full interval.
            self._opened_ns = time.perf_counter_ns()
            self._half_open = False
            return
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_ns = time.perf_counter_ns()

    def snapshot(self) -> dict[str, Any]:
        return {"provider": self.name, "state": self.state, "failures": self._failures}


# --- provider fallback --------------------------------------------------------

async def first_healthy(
    providers: Sequence[tuple[str, Callable[[], Awaitable[T]]]],
    *,
    breakers: dict[str, CircuitBreaker] | None = None,
    policy: RetryPolicy | None = None,
    deadline: Deadline | None = None,
    stage: str = "stage",
) -> tuple[str, T]:
    """Try providers in order, skipping open circuits. Return (name, result).

    Ordering is the caller's contract: put the fastest provider first and the
    most reliable one last, so the common path is quick and the tail still
    answers.

    Raises:
        AllProvidersFailed: with a per-provider error map for the trace.
    """
    breakers = breakers if breakers is not None else {}
    errors: dict[str, str] = {}
    trace = current_trace()

    for name, fn in providers:
        br = breakers.setdefault(name, CircuitBreaker(name=name))
        if not br.allows():
            errors[name] = "circuit open"
            if trace is not None:
                trace.mark(f"{stage}.skipped", provider=name, reason="circuit_open")
            continue
        try:
            result = await call_with_retry(
                fn, policy=policy, deadline=deadline, name=f"{stage}.{name}"
            )
        except BaseException as exc:  # noqa: BLE001 - recorded, then fall through
            br.record_failure()
            errors[name] = f"{type(exc).__name__}: {exc}"
            if trace is not None:
                trace.mark(f"{stage}.failed", provider=name, error=errors[name])
            if isinstance(exc, BudgetExhausted):
                break  # no budget left for anyone else
            continue
        br.record_success()
        if trace is not None:
            trace.mark(f"{stage}.selected", provider=name)
        return name, result

    raise AllProvidersFailed(errors)
