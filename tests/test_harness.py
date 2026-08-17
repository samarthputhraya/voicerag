"""Tests for tracing and resilience.

The resilience tests deliberately assert on *behaviour under failure*, because
that is the only time this code runs and the only time bugs in it are expensive.
"""

from __future__ import annotations

import asyncio

import pytest

from voicerag.harness.resilience import (
    AllProvidersFailed,
    BudgetExhausted,
    CircuitBreaker,
    Deadline,
    PermanentError,
    RetryPolicy,
    TransientError,
    call_with_retry,
    first_healthy,
)
from voicerag.harness.trace import Trace, current_trace, traced

FAST = RetryPolicy(attempts=3, base_ms=1, max_ms=2, timeout_ms=500, jitter=False)


# --- tracing ------------------------------------------------------------------

def test_spans_record_duration_and_nesting():
    t = Trace()
    with t.span("outer"):
        with t.span("inner"):
            pass
    outer = next(s for s in t.spans if s.name == "outer")
    inner = next(s for s in t.spans if s.name == "inner")
    assert inner.parent == outer.span_id
    assert outer.parent is None
    assert outer.ms >= inner.ms >= 0


def test_span_records_error_and_reraises():
    t = Trace()
    with pytest.raises(ValueError):
        with t.span("boom"):
            raise ValueError("nope")
    s = next(s for s in t.spans if s.name == "boom")
    assert s.error is not None and "ValueError" in s.error
    assert s.end_ns is not None, "span must close even on exception"


def test_marks_are_zero_duration_events():
    t = Trace()
    t.mark("vad.speech_end", source="client")
    s = t.spans[0]
    assert s.ms == 0.0
    assert s.attrs["source"] == "client"


@pytest.mark.asyncio
async def test_critical_path_counts_overlapping_work_once():
    """Two 60ms stages run concurrently: envelope ~60ms, naive sum ~120ms."""
    t = Trace()

    async def stage(name):
        with t.span(name):
            await asyncio.sleep(0.06)

    await asyncio.gather(stage("a"), stage("b"))

    naive_sum = sum(t.breakdown().values())
    assert naive_sum > 100, "both spans should each record ~60ms"
    assert t.critical_path_ms < 100, "overlap must be counted once"


def test_breakdown_sums_repeated_span_names():
    t = Trace()
    for _ in range(3):
        with t.span("retrieve"):
            pass
    assert "retrieve" in t.breakdown()
    assert len([s for s in t.spans if s.name == "retrieve"]) == 3


def test_traced_binds_and_restores_context():
    assert current_trace() is None
    with traced() as t:
        assert current_trace() is t
    assert current_trace() is None


# --- deadline -----------------------------------------------------------------

def test_deadline_slice_never_exceeds_remaining():
    d = Deadline(budget_ms=50)
    assert d.slice_ms(1_000) <= 50
    assert d.slice_ms(10) == pytest.approx(10, abs=1)


@pytest.mark.asyncio
async def test_deadline_expires():
    d = Deadline(budget_ms=10)
    await asyncio.sleep(0.03)
    assert d.expired
    assert d.remaining_ms == 0.0


# --- retry --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_succeeds_without_retry():
    calls = 0

    async def ok():
        nonlocal calls
        calls += 1
        return "fine"

    assert await call_with_retry(ok, policy=FAST) == "fine"
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_transient_then_succeeds():
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TransientError("503")
        return "recovered"

    assert await call_with_retry(flaky, policy=FAST) == "recovered"
    assert calls == 3


@pytest.mark.asyncio
async def test_permanent_error_is_never_retried():
    calls = 0

    async def bad():
        nonlocal calls
        calls += 1
        raise PermanentError("401 unauthorized")

    with pytest.raises(PermanentError):
        await call_with_retry(bad, policy=FAST)
    assert calls == 1, "a 401 must not be retried"


@pytest.mark.asyncio
async def test_gives_up_after_attempts_exhausted():
    calls = 0

    async def always():
        nonlocal calls
        calls += 1
        raise TransientError("down")

    with pytest.raises(TransientError):
        await call_with_retry(always, policy=FAST)
    assert calls == 3


@pytest.mark.asyncio
async def test_per_attempt_timeout_is_enforced():
    async def slow():
        await asyncio.sleep(5)

    pol = RetryPolicy(attempts=2, base_ms=1, max_ms=1, timeout_ms=20, jitter=False)
    with pytest.raises(TransientError):
        await call_with_retry(slow, policy=pol)


@pytest.mark.asyncio
async def test_retry_stops_when_deadline_exhausted():
    """The whole point: never retry into a budget that cannot pay for it."""
    calls = 0

    async def slow_fail():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        raise TransientError("down")

    d = Deadline(budget_ms=60)
    with pytest.raises((BudgetExhausted, TransientError)):
        await call_with_retry(
            slow_fail,
            policy=RetryPolicy(attempts=10, base_ms=1, max_ms=1, timeout_ms=500, jitter=False),
            deadline=d,
        )
    assert calls <= 3, f"budget should cap attempts, made {calls}"


@pytest.mark.asyncio
async def test_on_retry_callback_observes_each_failure():
    seen: list[int] = []

    async def flaky():
        if len(seen) < 2:
            raise TransientError("x")
        return 1

    await call_with_retry(flaky, policy=FAST, on_retry=lambda a, _e: seen.append(a))
    assert seen == [1, 2]


# --- circuit breaker ----------------------------------------------------------

def test_breaker_opens_after_threshold():
    br = CircuitBreaker(name="groq", threshold=3, reset_ms=10_000)
    assert br.state == "closed"
    for _ in range(3):
        br.record_failure()
    assert br.state == "open"
    assert not br.allows()


def test_breaker_success_resets_failure_count():
    br = CircuitBreaker(name="p", threshold=3)
    br.record_failure()
    br.record_failure()
    br.record_success()
    br.record_failure()
    assert br.state == "closed", "success must clear the streak"


@pytest.mark.asyncio
async def test_breaker_half_opens_then_closes_on_success():
    br = CircuitBreaker(name="p", threshold=1, reset_ms=20)
    br.record_failure()
    assert br.state == "open"
    await asyncio.sleep(0.03)
    assert br.state == "half_open"
    assert br.allows()
    br.record_success()
    assert br.state == "closed"


@pytest.mark.asyncio
async def test_failed_probe_reopens_the_breaker():
    br = CircuitBreaker(name="p", threshold=1, reset_ms=20)
    br.record_failure()
    await asyncio.sleep(0.03)
    assert br.allows()          # admits the probe
    br.record_failure()         # probe fails
    assert br.state == "open", "a failed probe must re-open, not accumulate"


# --- provider fallback --------------------------------------------------------

@pytest.mark.asyncio
async def test_falls_back_to_second_provider():
    async def primary():
        raise TransientError("groq down")

    async def secondary():
        return "answer from gemini"

    name, res = await first_healthy(
        [("groq", primary), ("gemini", secondary)], policy=FAST
    )
    assert name == "gemini"
    assert res == "answer from gemini"


@pytest.mark.asyncio
async def test_prefers_the_first_healthy_provider():
    async def primary():
        return "groq"

    async def secondary():  # pragma: no cover - must not run
        raise AssertionError("fallback ran despite healthy primary")

    name, _ = await first_healthy([("groq", primary), ("gemini", secondary)], policy=FAST)
    assert name == "groq"


@pytest.mark.asyncio
async def test_open_circuit_is_skipped_without_calling_provider():
    called = False

    async def dead():
        nonlocal called
        called = True
        raise TransientError("down")

    async def alive():
        return "ok"

    br = CircuitBreaker(name="dead", threshold=1, reset_ms=10_000)
    br.record_failure()
    breakers = {"dead": br}

    name, _ = await first_healthy(
        [("dead", dead), ("alive", alive)], breakers=breakers, policy=FAST
    )
    assert name == "alive"
    assert not called, "an open circuit must not issue a network call"


@pytest.mark.asyncio
async def test_all_providers_failed_reports_every_error():
    async def a():
        raise TransientError("a down")

    async def b():
        raise TransientError("b down")

    with pytest.raises(AllProvidersFailed) as ei:
        await first_healthy([("a", a), ("b", b)], policy=FAST)
    assert set(ei.value.errors) == {"a", "b"}


@pytest.mark.asyncio
async def test_fallback_records_provider_selection_in_the_trace():
    async def bad():
        raise TransientError("x")

    async def good():
        return 1

    with traced() as t:
        await first_healthy([("bad", bad), ("good", good)], policy=FAST, stage="generate")

    names = {s.name for s in t.spans}
    assert "generate.failed" in names
    assert "generate.selected" in names
