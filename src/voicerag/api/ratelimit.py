"""Rate limiting for a link that is public, credentialed and on a free tier.

Why this exists
---------------

The deployed demo is a URL with no authentication in front of live Sarvam and
Groq credentials, and Groq's free tier caps at 8,000 tokens per minute -- around
four or five RAG answers. A single loop pointed at ``POST /ask`` spends the
minute's quota faster than a judge can type a question. The visible failure is
not a 429 from us; it is ``Every generation provider failed`` on the judge's
screen, which reads as a broken submission rather than as someone else's abuse.

What this promises, and what it does not
----------------------------------------

Two layers, protecting against different things. The distinction matters enough
to state plainly rather than imply:

**Per-client limits are advisory.** Behind a reverse proxy the socket peer is
the proxy, so the only available identity is ``X-Forwarded-For`` -- a header
written by the client. It stops a naive scraper, a runaway retry loop and an
accidentally-shared benchmark script. A determined attacker rotates the header
and walks through it, and no amount of tuning changes that.

**The global limits are not spoofable**, because they count requests rather than
identities. That layer is what actually keeps the quota alive, and it is why the
per-client numbers can stay generous enough that two judges behind one NAT never
see a 429.

Design notes
------------

**Pure ASGI, not** ``starlette.middleware.base.BaseHTTPMiddleware``. Three
reasons, all load-bearing here. ``BaseHTTPMiddleware`` wraps the downstream app
in an anyio task group, putting a queue between ``/ask/stream`` and the client
on the one path whose whole purpose is to emit tokens the instant they exist. It
does not see ``websocket`` scopes at all, and ``WS /stt/stream`` is the endpoint
that opens a metered upstream session per connection. And a plain function that
either forwards the scope untouched or sends one response is a small thing to
reason about on the request path being benchmarked.

**In-process counters, no new dependency.** ``requirements.txt`` is pinned and
deliberately lean, and the container runs ``--workers 1`` (see the Dockerfile
CMD), so a process-local counter is not an approximation of the truth -- it *is*
the truth. ``slowapi`` would add a dependency and a Redis-shaped abstraction in
order to count integers in one process.

**State lives on the application instance**, not in a module global. Every test
builds a fresh app via :func:`~voicerag.api.main.create_app`, so limits cannot
leak from one test into the next, and two apps in one interpreter do not share a
budget.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Sequence

from starlette.responses import JSONResponse

from ..config import Settings
from .schemas import ErrorResponse

__all__ = [
    "RateLimiter",
    "RateLimitMiddleware",
    "Rule",
    "ROUTE_CLASS",
    "EXEMPT_PATHS",
    "client_identity",
    "WINDOW_S",
]

log = logging.getLogger("voicerag.api.ratelimit")

#: Window length, seconds. A minute is the unit Groq's own quota is denominated
#: in, so matching it keeps the two numbers comparable when reading a log.
WINDOW_S = 60.0

#: Longest client identity retained. ``X-Forwarded-For`` is attacker-controlled
#: and unbounded; truncating caps the memory one request can make us allocate.
#: An IPv6 address with a zone and a port fits comfortably.
MAX_IDENTITY_CHARS = 64


@dataclass(frozen=True)
class Rule:
    """The two limits that apply to one cost class.

    Attributes:
        per_client: Requests per minute from one client identity. Advisory --
            see the module docstring on why ``X-Forwarded-For`` cannot be
            trusted.
        per_server: Requests per minute across every client. Not spoofable;
            this is the layer that protects the third-party quota.
    """

    per_client: int
    per_server: int


#: Which cost class each route belongs to.
#:
#: Exact paths, not prefixes: ``/ask/stream`` starts with ``/ask``, and a prefix
#: match would make the table's behaviour depend on iteration order.
#:
#: **This table is the entire policy.** A path in neither this map nor
#: :data:`EXEMPT_PATHS` is unlimited -- correct for static assets, wrong for
#: anything that spends money. ``tests/test_ratelimit.py`` asserts that every
#: route the application actually registers appears in one of the two, so adding
#: a paid endpoint without classifying it fails the suite rather than the quota.
ROUTE_CLASS: dict[str, str] = {
    "/ask": "generate",
    "/ask/stream": "generate",
    "/speak": "speak",
    "/speculate": "light",
    "/examples": "light",
    "/stats": "light",
    "/stt/token": "light",
    "/stt/stream": "stt",
}

#: Deliberately unlimited.
#:
#: ``/healthz`` is polled by the container's own ``HEALTHCHECK`` every 30 s, and
#: a limiter that can starve it can take the deployment down by itself. ``/`` is
#: either the endpoint map or the frontend's ``index.html``; neither costs
#: anything beyond a dict lookup or a file read.
EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/", "/api", "/healthz", "/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
)


def client_identity(headers: Iterable[Sequence[bytes]], peer: str | None) -> str:
    """Best available identity for one caller.

    Prefers the leftmost ``X-Forwarded-For`` entry, then ``X-Real-IP``, then the
    socket peer. Inside a container behind a platform proxy the peer is the
    gateway address and is identical for every visitor, so without the forwarded
    header the per-client layer degenerates into a second global one -- which is
    safe, merely useless.

    The value is truncated, because the header is attacker-controlled and a long
    one would otherwise become a long dictionary key.

    Args:
        headers: Raw ASGI header pairs, ``(name, value)`` as bytes.
        peer: ``scope["client"][0]``, when the server provided one.

    Returns:
        A short string, never empty. ``"unknown"`` when nothing is available, so
        all such callers share one budget rather than each receiving their own.
    """
    forwarded = b""
    real_ip = b""
    for pair in headers:
        name = bytes(pair[0]).lower()
        # First occurrence wins: a client appending a second X-Forwarded-For
        # must not be able to displace the one the proxy wrote.
        if name == b"x-forwarded-for" and not forwarded:
            forwarded = bytes(pair[1])
        elif name == b"x-real-ip" and not real_ip:
            real_ip = bytes(pair[1])

    candidate = forwarded.decode("latin-1").split(",")[0].strip()
    if not candidate:
        candidate = real_ip.decode("latin-1").strip()
    if not candidate:
        candidate = (peer or "").strip()
    if not candidate:
        return "unknown"
    return candidate[:MAX_IDENTITY_CHARS]


class _Window:
    """A sliding window of hit timestamps.

    Exact, rather than a token bucket's approximation: at these volumes the
    deque holds tens of floats, and exactness makes ``Retry-After`` a real
    number instead of an estimate.
    """

    __slots__ = ("_hits", "_limit")

    def __init__(self, limit: int) -> None:
        self._hits: deque[float] = deque()
        self._limit = limit

    def retry_after(self, now: float) -> float | None:
        """Seconds until this window has room, or ``None`` if it has room now."""
        cutoff = now - WINDOW_S
        while self._hits and self._hits[0] <= cutoff:
            self._hits.popleft()
        if len(self._hits) < self._limit:
            return None
        return WINDOW_S - (now - self._hits[0])

    def record(self, now: float) -> None:
        """Count one admitted request."""
        self._hits.append(now)

    def __len__(self) -> int:
        return len(self._hits)


class RateLimiter:
    """Per-client and global sliding windows, plus WebSocket session accounting.

    Not thread-safe, and does not need to be: every caller runs on the one
    asyncio event loop, and no ``await`` occurs between reading a window and
    writing it.

    Args:
        cfg: Settings supplying the limits.
    """

    __slots__ = (
        "_rules",
        "_clients",
        "_servers",
        "_sessions",
        "_max_clients",
        "_enabled",
        "_stt_concurrent_client",
        "_stt_concurrent_total",
        "_trust_forwarded",
    )

    def __init__(self, cfg: Settings) -> None:
        self._enabled = cfg.rate_limit_enabled
        self._trust_forwarded = cfg.rate_limit_trust_forwarded_for
        self._max_clients = cfg.rate_limit_max_clients
        self._stt_concurrent_client = cfg.rate_limit_stt_concurrent
        self._stt_concurrent_total = cfg.rate_limit_stt_concurrent_total
        self._rules: dict[str, Rule] = {
            "generate": Rule(
                cfg.rate_limit_generate_per_min, cfg.rate_limit_generate_global_per_min
            ),
            "speak": Rule(cfg.rate_limit_speak_per_min, cfg.rate_limit_speak_global_per_min),
            "light": Rule(cfg.rate_limit_light_per_min, cfg.rate_limit_light_global_per_min),
            "stt": Rule(cfg.rate_limit_stt_per_min, cfg.rate_limit_stt_global_per_min),
        }
        #: ``(class, identity) -> window``, as an LRU. The eviction is what stops
        #: a client rotating ``X-Forwarded-For`` from turning this defence into a
        #: memory leak.
        self._clients: OrderedDict[tuple[str, str], _Window] = OrderedDict()
        self._servers: dict[str, _Window] = {}
        self._sessions: dict[str, int] = {}

    @property
    def enabled(self) -> bool:
        """Whether any limit is applied at all."""
        return self._enabled

    @property
    def trust_forwarded_for(self) -> bool:
        """Whether ``X-Forwarded-For`` is consulted when identifying a client."""
        return self._trust_forwarded

    def check(self, cost_class: str, identity: str, *, now: float | None = None) -> float | None:
        """Admit one request, or report how long the caller must wait.

        The global window is tested first, and both windows are recorded only on
        admission. A denied request is not counted, so a client hammering a
        closed door cannot extend its own lockout indefinitely -- which would
        make the ``Retry-After`` just sent a lie.

        Args:
            cost_class: A key of the rule table; an unknown class is unlimited.
            identity: From :func:`client_identity`.
            now: Monotonic seconds. Injected by tests; real callers omit it.

        Returns:
            ``None`` when admitted, else seconds to wait.
        """
        if not self._enabled:
            return None
        rule = self._rules.get(cost_class)
        if rule is None:
            return None
        moment = time.monotonic() if now is None else now

        server = self._servers.get(cost_class)
        if server is None:
            server = self._servers[cost_class] = _Window(rule.per_server)
        wait = server.retry_after(moment)
        if wait is not None:
            log.warning(
                "rate limited",
                extra={
                    "fields": {
                        "scope": "global",
                        "cost_class": cost_class,
                        "limit_per_min": rule.per_server,
                        "retry_after_s": round(wait, 3),
                    }
                },
            )
            return wait

        key = (cost_class, identity)
        client = self._clients.get(key)
        if client is None:
            client = _Window(rule.per_client)
            self._clients[key] = client
            while len(self._clients) > self._max_clients:
                self._clients.popitem(last=False)
        else:
            self._clients.move_to_end(key)
        wait = client.retry_after(moment)
        if wait is not None:
            log.warning(
                "rate limited",
                extra={
                    "fields": {
                        "scope": "client",
                        "cost_class": cost_class,
                        "limit_per_min": rule.per_client,
                        "retry_after_s": round(wait, 3),
                    }
                },
            )
            return wait

        server.record(moment)
        client.record(moment)
        return None

    def acquire_session(self, identity: str) -> bool:
        """Reserve one concurrent WebSocket relay session.

        Separate from :meth:`check`, which bounds the *rate* of new sessions.
        This bounds how many are open at once -- the quantity Sarvam meters, and
        one a rate limit alone cannot control: a client can open ten sockets in a
        second, stay under any per-minute limit, and hold ten upstream sessions
        for the full :data:`~voicerag.api.stt_relay._MAX_SESSION_S`.

        Returns:
            ``True`` when the caller may proceed. On ``False`` nothing was
            reserved, and :meth:`release_session` must not be called.
        """
        if not self._enabled:
            return True
        total = sum(self._sessions.values())
        if total >= self._stt_concurrent_total:
            log.warning(
                "stt relay refused",
                extra={
                    "fields": {
                        "scope": "global",
                        "reason": "concurrent_sessions",
                        "open": total,
                        "limit": self._stt_concurrent_total,
                    }
                },
            )
            return False
        held = self._sessions.get(identity, 0)
        if held >= self._stt_concurrent_client:
            log.warning(
                "stt relay refused",
                extra={
                    "fields": {
                        "scope": "client",
                        "reason": "concurrent_sessions",
                        "open": held,
                        "limit": self._stt_concurrent_client,
                    }
                },
            )
            return False
        self._sessions[identity] = held + 1
        return True

    def release_session(self, identity: str) -> None:
        """Release a session reserved by :meth:`acquire_session`."""
        if not self._enabled:
            return
        held = self._sessions.get(identity, 0) - 1
        if held > 0:
            self._sessions[identity] = held
        else:
            # Drop the key rather than leave a zero behind, so a client cycling
            # identities does not accumulate entries.
            self._sessions.pop(identity, None)

    def open_sessions(self, identity: str | None = None) -> int:
        """Open relay sessions, in total or for one identity."""
        if identity is None:
            return sum(self._sessions.values())
        return self._sessions.get(identity, 0)

    def snapshot(self) -> dict[str, Any]:
        """Current occupancy, for a health endpoint or a test.

        Counts only, never identities: a rate-limit report is not a place to
        publish a list of visitors' addresses.
        """
        return {
            "enabled": self._enabled,
            "tracked_clients": len(self._clients),
            "open_stt_sessions": sum(self._sessions.values()),
            "limits_per_min": {
                name: {"per_client": rule.per_client, "per_server": rule.per_server}
                for name, rule in self._rules.items()
            },
        }


def too_many(retry_after: float) -> JSONResponse:
    """The 429 body, in the one error shape the rest of the API uses.

    ``Retry-After`` is rounded *up*: a client that trusts a rounded-down value
    retries straight into a second denial, which reads as a limit that is stuck
    rather than one that is counting down.
    """
    seconds = max(1, int(retry_after) + 1)
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error="rate_limited",
            message=(
                "Too many requests. This is a public demo sharing one free-tier "
                f"quota; try again in {seconds}s."
            ),
        ).model_dump(),
        headers={"Retry-After": str(seconds)},
    )


class RateLimitMiddleware:
    """Applies :class:`RateLimiter` to ``http`` and ``websocket`` scopes.

    **Must be installed before the CORS middleware.** Starlette's
    ``add_middleware`` inserts at position 0, and the stack is built by wrapping
    ``reversed(user_middleware)``, so the middleware added *last* ends up
    outermost. Added after CORS, this one would sit outside it and a 429 would
    reach the browser with no ``Access-Control-Allow-Origin`` header -- which
    ``fetch`` surfaces as an opaque network failure, not as a rate limit. The
    frontend would then report "the server is unreachable" for a condition it
    could have explained precisely.

    Args:
        app: The downstream ASGI application.
        limiter: Shared limiter; also reachable as ``app.state.rate_limiter``.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], limiter: RateLimiter) -> None:
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        kind = scope.get("type")
        if kind not in ("http", "websocket") or not self.limiter.enabled:
            await self.app(scope, receive, send)
            return

        cost_class = ROUTE_CLASS.get(scope.get("path", ""))
        if cost_class is None:
            await self.app(scope, receive, send)
            return

        # CORS preflight does not reach here in the browser case -- CORSMiddleware
        # is outside this one and answers OPTIONS itself -- but a bare OPTIONS
        # costs nothing and should not consume a client's budget either way.
        if kind == "http" and scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        peer = client[0] if client else None
        if self.limiter.trust_forwarded_for:
            identity = client_identity(scope.get("headers", []), peer)
        else:
            identity = (peer or "unknown")[:MAX_IDENTITY_CHARS]

        if kind == "websocket":
            await self._websocket(scope, receive, send, cost_class, identity)
            return

        wait = self.limiter.check(cost_class, identity)
        if wait is not None:
            await too_many(wait)(scope, receive, send)
            return
        await self.app(scope, receive, send)

    async def _websocket(
        self, scope: dict[str, Any], receive: Any, send: Any, cost_class: str, identity: str
    ) -> None:
        """Gate a relay upgrade, and hold the session slot for its lifetime.

        A refusal is expressed as ``websocket.close`` sent in reply to
        ``websocket.connect`` and *before* any ``websocket.accept``, which is the
        ASGI way to decline a handshake: the server answers the upgrade with an
        HTTP error instead of opening a socket and immediately closing it.
        Opening it is precisely the cost being avoided.
        """
        admitted = self.limiter.check(cost_class, identity) is None and self.limiter.acquire_session(
            identity
        )
        if admitted:
            try:
                await self.app(scope, receive, send)
            finally:
                self.limiter.release_session(identity)
            return

        message = await receive()
        if message.get("type") != "websocket.connect":  # pragma: no cover - protocol violation
            return
        # 1013 "Try Again Later" is the registered close code for exactly this,
        # and the client will not see it: a pre-accept close has no close frame
        # to carry it, so uvicorn answers the handshake with HTTP 403 and drops
        # the code (websockets_sansio_impl rejects with FORBIDDEN without
        # reading it). It is stated for the reader and for any server that does
        # surface it; the browser-visible outcome is a failed upgrade either way,
        # which is the correct outcome and the one the tests assert.
        await send({"type": "websocket.close", "code": 1013})
