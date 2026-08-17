"""Groq generator: OpenAI-compatible SSE streaming over a warm connection pool.

Groq is the primary generator because decode throughput is the binding
constraint on this pipeline. The 200 ms bar is measured to the last output
token, so tokens-per-second is not a nice-to-have -- it is most of the budget.

**Why the connection pool is created once and never torn down.** A cold HTTPS
connection costs a DNS lookup, a TCP handshake and a TLS handshake: roughly
100 ms on a good link, which is half the entire budget spent before a single
byte of prompt is sent. Reusing one warm, keep-alive pooled connection reduces
that to zero on every request after the first. This is why :class:`GroqGenerator`
owns a long-lived :class:`httpx.AsyncClient` instead of opening a client per
call, why :meth:`warm` exists to pay the handshake at process start rather than
on the user's first question, and why ``keepalive_expiry`` is set well above any
plausible idle gap between demo questions. A per-request client would be simpler
code and a guaranteed failure of the requirement.

HTTP/2 is requested when the ``h2`` package is available. httpx negotiates it
via ALPN and transparently falls back to HTTP/1.1, so this is a free upgrade
where the endpoint supports it: one multiplexed connection means a speculative
request and a real request never queue behind each other head-of-line.

Default model is ``openai/gpt-oss-20b`` with ``reasoning_effort="low"``. The
reasoning models will happily spend hundreds of tokens thinking before they emit
an answer, and thinking tokens are billed to our latency budget exactly like
answer tokens; ``"low"`` is what keeps a reasoning model inside a realtime
budget.
"""

from __future__ import annotations

import json
from importlib.util import find_spec
from typing import Any, AsyncIterator, Mapping, MutableMapping

from ..harness.resilience import PermanentError, TransientError
from .base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, Generator

__all__ = ["GroqGenerator", "GROQ_BASE_URL", "DEFAULT_GROQ_MODEL"]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

#: Models that accept the ``reasoning_effort`` parameter. Sending it to a model
#: that does not understand it is a 400, so it is gated on the model id rather
#: than sent unconditionally.
_REASONING_MODEL_MARKERS = ("gpt-oss", "reasoning", "qwen3", "deepseek-r1")


class GroqGenerator(Generator):
    """Streaming client for Groq's OpenAI-compatible chat completions API.

    Args:
        api_key: Bearer token.
        model: Model id. Defaults to :data:`DEFAULT_GROQ_MODEL`.
        base_url: API root, without a trailing slash.
        max_tokens: Hard output ceiling; see
            :data:`~voicerag.generate.base.DEFAULT_MAX_TOKENS`.
        temperature: Sampling temperature, 0.0 for reproducible evaluation.
        reasoning_effort: ``"low"``, ``"medium"`` or ``"high"``. Sent only for
            models that accept it. ``None`` omits it entirely.
        max_tokens_field: Which spelling of the output cap to send --
            ``"max_tokens"`` or ``"max_completion_tokens"``. Configurable
            because the two are accepted inconsistently across OpenAI-compatible
            gateways and model families, and this is a one-line fix rather than
            a code change when the live API disagrees.
        include_usage: Ask for a final usage chunk. Costs one extra SSE frame
            *after* the last token and buys exact token counts for the latency
            report, which turns "about 60 tokens" into a measured number.
        connect_timeout_s: TCP+TLS timeout. Deliberately short: a connection
            that has not been established in 2 s will not meet the budget, and
            the request should fail over rather than wait.
        timeout_s: Total request timeout.
        client: Injected ``httpx.AsyncClient``. Supplying one is how tests run
            offline and how a deployment shares one warm pool across generators.
        transport: Injected ``httpx.AsyncBaseTransport`` used when this class
            builds its own client -- the seam the SSE tests use.
        extra_body: Additional top-level payload fields, merged last.

    Raises:
        ValueError: If ``api_key`` is empty or ``max_tokens_field`` is unknown.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_GROQ_MODEL,
        base_url: str = GROQ_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        reasoning_effort: str | None = "low",
        max_tokens_field: str = "max_tokens",
        include_usage: bool = True,
        connect_timeout_s: float = 2.0,
        timeout_s: float = 15.0,
        client: Any | None = None,
        transport: Any | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature)
        if not api_key:
            raise ValueError("GroqGenerator requires an api_key")
        if max_tokens_field not in ("max_tokens", "max_completion_tokens"):
            raise ValueError(
                "max_tokens_field must be 'max_tokens' or 'max_completion_tokens', "
                f"got {max_tokens_field!r}"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._reasoning_effort = reasoning_effort
        self._max_tokens_field = max_tokens_field
        self._include_usage = include_usage
        self._connect_timeout_s = connect_timeout_s
        self._timeout_s = timeout_s
        self._client = client
        self._transport = transport
        self._extra_body = dict(extra_body or {})

    # ------------------------------------------------------------ transport

    @property
    def supports_reasoning_effort(self) -> bool:
        """Whether ``reasoning_effort`` is sent for the configured model."""
        low = self.model.lower()
        return any(marker in low for marker in _REASONING_MODEL_MARKERS)

    def _ensure_client(self) -> Any:
        """Return the shared client, building it on first use.

        Built lazily and cached on the instance so the pool outlives individual
        requests. This is the object whose warmth the whole latency argument
        rests on -- see the module docstring.

        Raises:
            PermanentError: If :mod:`httpx` is not installed.
        """
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - only without the dependency
            raise PermanentError(
                "GroqGenerator needs the 'httpx' package: pip install httpx"
            ) from exc

        kwargs: dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": httpx.Timeout(self._timeout_s, connect=self._connect_timeout_s),
            "headers": {
                "authorization": f"Bearer {self._api_key}",
                "content-type": "application/json",
                # Explicit: some proxies default to gzip, and a compressed SSE
                # stream can be buffered by the compressor until it has enough
                # bytes to emit -- which converts token-by-token streaming into
                # one late block and destroys TTFT.
                "accept": "text/event-stream",
                "accept-encoding": "identity",
            },
            "limits": httpx.Limits(
                max_keepalive_connections=8,
                max_connections=16,
                # Far longer than any gap between demo questions, so the
                # handshake is paid once per process rather than once per pause.
                keepalive_expiry=600.0,
            ),
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        elif find_spec("h2") is not None:
            # httpx needs `h2` present to offer HTTP/2 at all; asking for it
            # without the package is an immediate ImportError. ALPN then decides
            # per connection, with HTTP/1.1 as the automatic fallback.
            kwargs["http2"] = True
        self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def warm(self) -> bool:
        """Pre-establish the TLS connection so the first query does not pay it.

        Call at startup. Best-effort by design: a warm-up that fails must never
        stop the service from starting, because the first real request will
        simply pay the handshake it would have paid anyway.

        Returns:
            True if the endpoint answered.
        """
        try:
            resp = await self._ensure_client().get(
                "/models", headers={"accept": "application/json"}
            )
            return int(resp.status_code) < 500
        except Exception:  # noqa: BLE001 - warm-up is advisory, never fatal
            return False

    async def aclose(self) -> None:
        """Close the connection pool."""
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    # ------------------------------------------------------------ payload

    def build_payload(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build the exact JSON body sent to ``/chat/completions``.

        Public and pure so the wire format can be asserted in a test without a
        transport -- against an API this environment cannot reach, that
        assertion is the specification.

        Args:
            system: System prompt; goes first so it is the cacheable prefix.
            user: User turn with context and question.
            max_tokens: Per-call override of the output ceiling.
            **overrides: Additional top-level fields, merged last.

        Returns:
            The request body.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "top_p": 1.0,
            self._max_tokens_field: int(max_tokens or self.max_tokens),
            "stream": True,
        }
        if self._include_usage:
            payload["stream_options"] = {"include_usage": True}
        if self._reasoning_effort and self.supports_reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        payload.update(self._extra_body)
        payload.update(overrides)
        return payload

    # ------------------------------------------------------------ streaming

    async def _stream_deltas(
        self,
        system: str,
        user: str,
        *,
        meta: MutableMapping[str, Any],
        max_tokens: int | None = None,
        **overrides: Any,
    ) -> AsyncIterator[str]:
        """Stream deltas from the chat completions endpoint.

        Raises:
            PermanentError: 4xx other than 429 -- a bad key, a bad model, a
                malformed body. Retrying spends budget to fail identically.
            TransientError: 429, 5xx, timeouts and connection errors.
        """
        client = self._ensure_client()
        payload = self.build_payload(system, user, max_tokens=max_tokens, **overrides)
        try:
            async with client.stream(
                "POST", f"{self._base_url}/chat/completions", json=payload
            ) as resp:
                status = int(resp.status_code)
                if status >= 400:
                    body = await _read_error_body(resp)
                    _raise_for_status(status, body)
                async for line in resp.aiter_lines():
                    for delta in _parse_sse_line(line, meta):
                        yield delta
        except (PermanentError, TransientError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalised to the taxonomy
            raise TransientError(f"groq stream failed: {exc}") from exc


async def _read_error_body(resp: Any) -> str:
    """Read an error response body without letting the read mask the status."""
    try:
        raw = await resp.aread()
    except Exception:  # noqa: BLE001 - the status code is the useful part
        return ""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", "replace")[:400]
    return str(raw)[:400]


def _raise_for_status(status: int, body: str) -> None:
    """Map an HTTP status onto the harness error taxonomy."""
    if status == 429 or status >= 500:
        raise TransientError(f"groq HTTP {status}: {body}")
    raise PermanentError(f"groq HTTP {status}: {body}")


def _parse_sse_line(line: str, meta: MutableMapping[str, Any]) -> list[str]:
    """Extract text deltas from one SSE line, recording metadata as it appears.

    Returns a list because a chunk may legitimately carry no text: keepalive
    comments, the ``[DONE]`` sentinel, the role-only opening chunk, and the
    usage-only final chunk all parse to zero deltas. Yielding ``""`` for those
    instead would be indistinguishable from a real empty token and would corrupt
    the TTFT measurement, which is defined on the first *non-empty* delta.
    """
    if not line or line.startswith(":"):
        return []
    if not line.startswith("data:"):
        return []
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return []
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        # A malformed frame is not worth failing a nearly complete answer over;
        # the stream carries on and the guardrails judge the result.
        return []
    if not isinstance(chunk, dict):
        return []

    if isinstance(chunk.get("model"), str):
        meta.setdefault("model", chunk["model"])
    usage = chunk.get("usage") or (chunk.get("x_groq") or {}).get("usage")
    if isinstance(usage, Mapping):
        meta["usage"] = dict(usage)

    out: list[str] = []
    for choice in chunk.get("choices") or ():
        if not isinstance(choice, Mapping):
            continue
        if choice.get("finish_reason"):
            meta["finish_reason"] = choice["finish_reason"]
        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            content = delta.get("content")
            if isinstance(content, str) and content:
                out.append(content)
        # Non-streaming shape, seen when a gateway coalesces a short answer.
        message = choice.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, str) and content:
                out.append(content)
    return out
