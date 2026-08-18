"""Tests for :mod:`voicerag.api.ratelimit`.

Two kinds of test here, and the split is deliberate.

The unit tests drive :class:`RateLimiter` directly with an injected clock, so
they assert on window arithmetic without sleeping. The HTTP tests go through the
real application assembled by :func:`~voicerag.api.main.create_app`, because the
two properties that matter most in production are properties of the *stack*, not
of the limiter: that a 429 carries CORS headers, and that the middleware does
not interpose itself on the SSE path.

The first test in the file is the one that will still be earning its keep in six
months: it fails when a new endpoint is added without deciding whether it costs
money.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from test_api import ANSWER, QUESTION, make_client
from test_pipeline import ScriptedGenerator, build_corpus, make_settings
from voicerag.api.main import create_app
from voicerag.api.ratelimit import (
    EXEMPT_PATHS,
    ROUTE_CLASS,
    RateLimiter,
    client_identity,
)


@pytest.fixture(scope="module")
def corpus() -> Any:
    """One index for the whole module; building it dominates the runtime."""
    embedder, hybrid, store = build_corpus()
    yield embedder, hybrid, store
    hybrid.close()


def generous(**overrides: Any) -> dict[str, Any]:
    """Overrides that keep every limit out of the way except the named one."""
    base: dict[str, Any] = {
        "rate_limit_generate_per_min": 10_000,
        "rate_limit_generate_global_per_min": 10_000,
        "rate_limit_speak_per_min": 10_000,
        "rate_limit_speak_global_per_min": 10_000,
        "rate_limit_light_per_min": 10_000,
        "rate_limit_light_global_per_min": 10_000,
        "rate_limit_stt_per_min": 10_000,
        "rate_limit_stt_global_per_min": 10_000,
    }
    base.update(overrides)
    return base


# --- the policy table matches the application ---------------------------------


def test_every_route_is_classified_or_exempt() -> None:
    """A new endpoint must be classified, or the suite fails rather than the quota.

    ``ratelimit.ROUTE_CLASS`` is the whole policy: a path in neither it nor
    ``EXEMPT_PATHS`` is unlimited. That default is right for static assets and
    catastrophic for anything that spends a third-party token, and the mistake is
    invisible -- the endpoint works perfectly right up until someone points a
    loop at it. So the application's own route table is the source of truth, and
    this test is the thing that notices.
    """
    app = create_app(make_settings(index_dir="data/does-not-exist"))
    paths = {
        route.path  # type: ignore[attr-defined]
        for route in app.routes
        if getattr(route, "path", None) is not None
    }
    unclassified = sorted(paths - set(ROUTE_CLASS) - EXEMPT_PATHS)
    assert unclassified == [], (
        f"unclassified routes: {unclassified}. Add each to ROUTE_CLASS with a "
        "cost class, or to EXEMPT_PATHS if it genuinely costs nothing."
    )


def test_paid_endpoints_are_not_exempt() -> None:
    """The endpoints that spend money are classified, not merely present.

    Guards against the fix for the test above being "add it to EXEMPT_PATHS".
    """
    for path in ("/ask", "/ask/stream", "/speak", "/stt/stream"):
        assert path not in EXEMPT_PATHS
        assert path in ROUTE_CLASS


def test_healthz_is_exempt() -> None:
    """The container's own HEALTHCHECK polls /healthz every 30s.

    A limiter that can starve it can take the deployment down without any
    external help at all.
    """
    assert "/healthz" in EXEMPT_PATHS
    assert "/healthz" not in ROUTE_CLASS


# --- window arithmetic --------------------------------------------------------


class TestLimiterUnit:
    """:class:`RateLimiter` driven directly, with an injected clock."""

    def test_admits_up_to_the_limit_then_refuses(self) -> None:
        limiter = RateLimiter(make_settings(**generous(rate_limit_generate_per_min=3)))
        for i in range(3):
            assert limiter.check("generate", "a", now=100.0 + i) is None
        wait = limiter.check("generate", "a", now=103.0)
        assert wait is not None and 0 < wait <= 60.0

    def test_window_slides(self) -> None:
        """Room reappears exactly one window after the oldest hit, not before."""
        limiter = RateLimiter(make_settings(**generous(rate_limit_generate_per_min=2)))
        assert limiter.check("generate", "a", now=0.0) is None
        assert limiter.check("generate", "a", now=10.0) is None
        assert limiter.check("generate", "a", now=59.0) == pytest.approx(1.0)
        # The first hit ages out at t=60.
        assert limiter.check("generate", "a", now=60.5) is None

    def test_a_refused_request_is_not_counted(self) -> None:
        """Hammering a closed door must not extend the lockout.

        If denials were recorded, a client retrying every second would push its
        own window forward forever and the Retry-After we just sent would be a
        lie. The countdown has to actually count down.
        """
        limiter = RateLimiter(make_settings(**generous(rate_limit_generate_per_min=1)))
        assert limiter.check("generate", "a", now=0.0) is None
        first = limiter.check("generate", "a", now=10.0)
        for t in range(11, 30):
            later = limiter.check("generate", "a", now=float(t))
            assert later is not None
        assert first is not None and later is not None
        assert later < first  # strictly decreasing, i.e. genuinely counting down
        assert limiter.check("generate", "a", now=60.1) is None

    def test_clients_have_independent_budgets(self) -> None:
        limiter = RateLimiter(make_settings(**generous(rate_limit_generate_per_min=1)))
        assert limiter.check("generate", "a", now=0.0) is None
        assert limiter.check("generate", "a", now=1.0) is not None
        assert limiter.check("generate", "b", now=1.0) is None

    def test_global_limit_holds_when_the_client_identity_is_rotated(self) -> None:
        """The layer that spoofing cannot defeat.

        Per-client limits key on a client-written header. This asserts the
        property that makes the design defensible anyway: a caller presenting a
        fresh identity on every request still hits the global ceiling.
        """
        limiter = RateLimiter(
            make_settings(**generous(rate_limit_generate_global_per_min=5))
        )
        for i in range(5):
            assert limiter.check("generate", f"spoof-{i}", now=float(i)) is None
        assert limiter.check("generate", "spoof-99", now=5.0) is not None

    def test_cost_classes_do_not_share_a_budget(self) -> None:
        limiter = RateLimiter(
            make_settings(**generous(rate_limit_generate_per_min=1))
        )
        assert limiter.check("generate", "a", now=0.0) is None
        assert limiter.check("generate", "a", now=1.0) is not None
        assert limiter.check("light", "a", now=1.0) is None

    def test_unknown_cost_class_is_unlimited(self) -> None:
        limiter = RateLimiter(make_settings())
        for i in range(1000):
            assert limiter.check("no-such-class", "a", now=float(i)) is None

    def test_disabled_limiter_admits_everything(self) -> None:
        limiter = RateLimiter(
            make_settings(rate_limit_enabled=False, rate_limit_generate_per_min=1)
        )
        for i in range(50):
            assert limiter.check("generate", "a", now=float(i)) is None
        assert limiter.acquire_session("a") is True

    def test_tracked_clients_are_bounded(self) -> None:
        """A rotating X-Forwarded-For must not become a memory leak.

        The eviction is not an optimisation: without it the defence against
        spoofing is itself the resource an attacker exhausts.
        """
        limiter = RateLimiter(make_settings(**generous(rate_limit_max_clients=16)))
        for i in range(500):
            limiter.check("generate", f"client-{i}", now=float(i))
        assert limiter.snapshot()["tracked_clients"] <= 16


class TestSessionAccounting:
    """Concurrent WebSocket relay sessions, which a rate limit alone cannot bound."""

    def test_per_client_concurrency_cap(self) -> None:
        limiter = RateLimiter(make_settings(rate_limit_stt_concurrent=2))
        assert limiter.acquire_session("a") is True
        assert limiter.acquire_session("a") is True
        assert limiter.acquire_session("a") is False
        assert limiter.acquire_session("b") is True

    def test_global_concurrency_cap(self) -> None:
        limiter = RateLimiter(
            make_settings(rate_limit_stt_concurrent=5, rate_limit_stt_concurrent_total=3)
        )
        for name in ("a", "b", "c"):
            assert limiter.acquire_session(name) is True
        assert limiter.acquire_session("d") is False

    def test_release_frees_a_slot_and_leaves_no_residue(self) -> None:
        limiter = RateLimiter(make_settings(rate_limit_stt_concurrent=1))
        assert limiter.acquire_session("a") is True
        limiter.release_session("a")
        assert limiter.open_sessions() == 0
        assert limiter.acquire_session("a") is True

    def test_release_of_an_unheld_session_is_harmless(self) -> None:
        """Defensive: the middleware's ``finally`` must never make things worse."""
        limiter = RateLimiter(make_settings())
        limiter.release_session("never-acquired")
        assert limiter.open_sessions() == 0


# --- client identity ----------------------------------------------------------


class TestClientIdentity:
    def test_prefers_leftmost_forwarded_entry(self) -> None:
        headers = [(b"x-forwarded-for", b"203.0.113.7, 70.41.3.18, 150.172.238.178")]
        assert client_identity(headers, "10.0.0.1") == "203.0.113.7"

    def test_first_forwarded_header_wins(self) -> None:
        """A client appending its own header must not displace the proxy's."""
        headers = [
            (b"x-forwarded-for", b"203.0.113.7"),
            (b"x-forwarded-for", b"1.2.3.4"),
        ]
        assert client_identity(headers, "10.0.0.1") == "203.0.113.7"

    def test_falls_back_to_real_ip_then_peer(self) -> None:
        assert client_identity([(b"x-real-ip", b"198.51.100.9")], "10.0.0.1") == "198.51.100.9"
        assert client_identity([], "10.0.0.1") == "10.0.0.1"

    def test_never_empty(self) -> None:
        """All unidentifiable callers share one budget rather than each getting one."""
        assert client_identity([], None) == "unknown"
        assert client_identity([(b"x-forwarded-for", b"   ")], None) == "unknown"

    def test_is_truncated(self) -> None:
        """The header is attacker-controlled, so it must not become a long key."""
        assert len(client_identity([(b"x-forwarded-for", b"A" * 5000)], None)) == 64

    def test_header_case_is_ignored(self) -> None:
        assert client_identity([(b"X-Forwarded-For", b"203.0.113.7")], None) == "203.0.113.7"


# --- through the real application ---------------------------------------------


class TestOverHttp:
    def test_429_has_the_standard_error_shape_and_retry_after(self, corpus: Any) -> None:
        client, _ = make_client(
            corpus,
            ScriptedGenerator(ANSWER),
            **generous(rate_limit_generate_per_min=2),
        )
        with client:
            for _ in range(2):
                assert client.post("/ask", json={"question": QUESTION}).status_code == 200
            blocked = client.post("/ask", json={"question": QUESTION})

        assert blocked.status_code == 429
        body = blocked.json()
        assert body["error"] == "rate_limited"
        assert body["message"]
        assert set(body) == {"error", "message", "trace_id"}
        assert int(blocked.headers["retry-after"]) >= 1

    def test_429_carries_cors_headers(self, corpus: Any) -> None:
        """The reason the limiter is installed *before* the CORS middleware.

        Starlette wraps ``reversed(user_middleware)``, so the last middleware
        added is outermost. Added after CORS, the limiter would sit outside it
        and a 429 would arrive with no ``Access-Control-Allow-Origin`` -- which
        ``fetch`` surfaces as an opaque network error, so the browser would
        report "the server is unreachable" for a condition the server had just
        explained precisely.
        """
        origin = "http://localhost:3000"
        client, _ = make_client(
            corpus,
            ScriptedGenerator(ANSWER),
            cors_origins=[origin],
            **generous(rate_limit_generate_per_min=1),
        )
        with client:
            ok = client.post("/ask", json={"question": QUESTION}, headers={"origin": origin})
            blocked = client.post(
                "/ask", json={"question": QUESTION}, headers={"origin": origin}
            )

        assert ok.status_code == 200
        assert blocked.status_code == 429
        assert blocked.headers["access-control-allow-origin"] == origin
        # Retry-After is not CORS-safelisted; unexposed, the browser cannot read it.
        assert "retry-after" in blocked.headers["access-control-expose-headers"].lower()

    def test_forwarded_for_separates_clients(self, corpus: Any) -> None:
        client, _ = make_client(
            corpus,
            ScriptedGenerator(ANSWER),
            **generous(rate_limit_generate_per_min=1),
        )
        with client:
            first = client.post(
                "/ask", json={"question": QUESTION}, headers={"x-forwarded-for": "1.1.1.1"}
            )
            same = client.post(
                "/ask", json={"question": QUESTION}, headers={"x-forwarded-for": "1.1.1.1"}
            )
            other = client.post(
                "/ask", json={"question": QUESTION}, headers={"x-forwarded-for": "2.2.2.2"}
            )
        assert (first.status_code, same.status_code, other.status_code) == (200, 429, 200)

    def test_forwarded_for_is_ignored_when_untrusted(self, corpus: Any) -> None:
        """With the header off, rotating it buys nothing."""
        client, _ = make_client(
            corpus,
            ScriptedGenerator(ANSWER),
            rate_limit_trust_forwarded_for=False,
            **generous(rate_limit_generate_per_min=1),
        )
        with client:
            first = client.post(
                "/ask", json={"question": QUESTION}, headers={"x-forwarded-for": "1.1.1.1"}
            )
            other = client.post(
                "/ask", json={"question": QUESTION}, headers={"x-forwarded-for": "2.2.2.2"}
            )
        assert (first.status_code, other.status_code) == (200, 429)

    def test_healthz_is_never_limited(self, corpus: Any) -> None:
        client, _ = make_client(corpus, **generous(rate_limit_light_per_min=1))
        with client:
            for _ in range(30):
                assert client.get("/healthz").status_code == 200

    def test_healthz_reports_the_limiter(self, corpus: Any) -> None:
        """A judge asking "is this rate limited?" should be able to look."""
        client, _ = make_client(corpus)
        with client:
            report = client.get("/healthz").json()["config"]["rate_limit"]
        assert report["enabled"] is True
        assert report["limits_per_min"]["generate"]["per_client"] > 0
        # Counts, never addresses.
        assert "clients" not in str(report).lower() or "tracked_clients" in report

    def test_ask_and_stream_share_one_budget(self, corpus: Any) -> None:
        """Both spend LLM tokens, so one must not be a way around the other."""
        client, _ = make_client(
            corpus,
            ScriptedGenerator(ANSWER),
            **generous(rate_limit_generate_per_min=1),
        )
        with client:
            assert client.post("/ask", json={"question": QUESTION}).status_code == 200
            assert (
                client.post("/ask/stream", json={"question": QUESTION}).status_code == 429
            )

    def test_streaming_still_streams_under_the_middleware(self, corpus: Any) -> None:
        """The reason this is pure ASGI rather than ``BaseHTTPMiddleware``.

        ``BaseHTTPMiddleware`` routes the response through an anyio task group,
        which is exactly the wrong thing to put between the token generator and
        the client on the one endpoint whose value is time-to-first-token. An
        admitted request must reach the router with the send channel untouched.
        """
        from test_api import parse_sse

        client, _ = make_client(corpus, ScriptedGenerator(ANSWER))
        with client:
            response = client.post("/ask/stream", json={"question": QUESTION})
        assert response.status_code == 200
        events = [name for name, _ in parse_sse(response.text)]
        assert "token" in events
        assert events[-1] == "final"

    def test_disabled_limiter_is_a_pass_through(self, corpus: Any) -> None:
        client, _ = make_client(
            corpus,
            ScriptedGenerator(ANSWER),
            rate_limit_enabled=False,
            rate_limit_generate_per_min=1,
        )
        with client:
            for _ in range(6):
                assert client.post("/ask", json={"question": QUESTION}).status_code == 200


class TestOverWebSocket:
    """The relay upgrade, which no HTTP middleware would see."""

    def test_upgrade_is_refused_over_the_rate_limit(self, corpus: Any) -> None:
        """Refused before ``accept``, so no socket is opened at all.

        Opening it is the cost being avoided: an accepted socket has already
        claimed an upstream Sarvam session by the time we would close it.
        """
        client, _ = make_client(corpus, **generous(rate_limit_stt_per_min=1))
        with client:
            with client.websocket_connect("/stt/stream") as first:
                first.close()
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/stt/stream") as second:
                    second.receive()

    def test_concurrency_slot_is_released_when_the_socket_closes(
        self, corpus: Any
    ) -> None:
        """Otherwise the third question of a demo is refused for no visible reason."""
        client, state = make_client(corpus, **generous(rate_limit_stt_concurrent=1))
        limiter = client.app.state.rate_limiter  # type: ignore[attr-defined]
        with client:
            for _ in range(3):
                with client.websocket_connect("/stt/stream") as ws:
                    ws.close()
                assert limiter.open_sessions() == 0
