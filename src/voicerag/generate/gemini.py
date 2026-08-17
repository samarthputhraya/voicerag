"""Gemini Flash-Lite generator: the fallback when Groq is unavailable.

A fallback is only worth having if it fails independently of the primary.
Gemini is a different company, a different network path and a different capacity
pool, so a Groq incident does not take it with it -- which is the entire point.
It is second rather than first because its decode throughput is lower, and under
a full-completion budget throughput is what decides whether the answer lands.

Two settings do the heavy lifting:

* ``thinkingBudget: 0`` disables Flash-Lite's thinking phase. Thinking tokens are
  generated before the first answer token and are billed to our latency budget
  exactly like answer tokens, so leaving it on can add hundreds of milliseconds
  to a request whose entire budget is 200 ms.
* ``maxOutputTokens`` mirrors the Groq ceiling, so failing over changes which
  vendor answers but not how long the answer is allowed to be. A fallback that
  quietly produces a three-sentence answer would turn a provider outage into a
  latency failure.

Wire shape implemented here: ``POST {base}/models/{model}:streamGenerateContent
?alt=sse`` with the key in the ``x-goog-api-key`` header rather than the query
string, so it cannot leak into logs, proxies or a screen recording of the demo.
"""

from __future__ import annotations

import json
from importlib.util import find_spec
from typing import Any, AsyncIterator, Mapping, MutableMapping

from ..harness.resilience import PermanentError, TransientError
from .base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, Generator

__all__ = ["GeminiGenerator", "GEMINI_BASE_URL", "DEFAULT_GEMINI_MODEL"]

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


class GeminiGenerator(Generator):
    """Streaming client for the Gemini ``streamGenerateContent`` endpoint.

    Args:
        api_key: Sent as ``x-goog-api-key``.
        model: Model id. Defaults to :data:`DEFAULT_GEMINI_MODEL`.
        base_url: API root, without a trailing slash.
        max_tokens: Output ceiling, mapped to ``maxOutputTokens``.
        temperature: Sampling temperature.
        thinking_budget: Thinking-token allowance. ``0`` disables thinking;
            ``None`` omits the field and lets the model decide, which is the
            right choice only if latency stops mattering.
        connect_timeout_s: TCP+TLS timeout.
        timeout_s: Total request timeout.
        client: Injected ``httpx.AsyncClient``; the offline-test seam and the
            way a deployment shares a warm pool.
        transport: Injected ``httpx.AsyncBaseTransport`` for a self-built client.
        extra_config: Extra ``generationConfig`` fields, merged last.

    Raises:
        ValueError: If ``api_key`` is empty.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        base_url: str = GEMINI_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        thinking_budget: int | None = 0,
        connect_timeout_s: float = 2.0,
        timeout_s: float = 15.0,
        client: Any | None = None,
        transport: Any | None = None,
        extra_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature)
        if not api_key:
            raise ValueError("GeminiGenerator requires an api_key")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._thinking_budget = thinking_budget
        self._connect_timeout_s = connect_timeout_s
        self._timeout_s = timeout_s
        self._client = client
        self._transport = transport
        self._extra_config = dict(extra_config or {})

    # ------------------------------------------------------------ transport

    def _ensure_client(self) -> Any:
        """Return the shared client, building it on first use.

        Cached on the instance for the same reason as
        :class:`~voicerag.generate.groq.GroqGenerator`: the TLS handshake must
        be paid once per process, not once per question.

        Raises:
            PermanentError: If :mod:`httpx` is not installed.
        """
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - only without the dependency
            raise PermanentError(
                "GeminiGenerator needs the 'httpx' package: pip install httpx"
            ) from exc

        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(self._timeout_s, connect=self._connect_timeout_s),
            "headers": {
                "x-goog-api-key": self._api_key,
                "content-type": "application/json",
                "accept": "text/event-stream",
                # See GroqGenerator: a compressed SSE stream can be buffered by
                # the compressor and arrive as one late block.
                "accept-encoding": "identity",
            },
            "limits": httpx.Limits(
                max_keepalive_connections=4,
                max_connections=8,
                keepalive_expiry=600.0,
            ),
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        elif find_spec("h2") is not None:
            # See GroqGenerator: probe for the package, let ALPN choose.
            kwargs["http2"] = True
        self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def warm(self) -> bool:
        """Pre-establish the TLS connection. Best-effort; never raises."""
        try:
            resp = await self._ensure_client().get(
                f"{self._base_url}/models", headers={"accept": "application/json"}
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

    def url(self) -> str:
        """Full streaming endpoint URL, including ``alt=sse``.

        Without ``alt=sse`` this endpoint streams a JSON *array* that arrives in
        arbitrarily chunked fragments, which cannot be parsed incrementally
        without a streaming JSON parser -- and therefore cannot produce an
        honest TTFT.
        """
        return f"{self._base_url}/models/{self.model}:streamGenerateContent?alt=sse"

    def build_payload(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build the exact JSON body sent to ``streamGenerateContent``.

        Args:
            system: System prompt, sent as ``systemInstruction`` so it stays a
                stable prefix rather than being folded into the user turn.
            user: User turn with context and question.
            max_tokens: Per-call override of the output ceiling.
            **overrides: Additional top-level fields, merged last.

        Returns:
            The request body.
        """
        config: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": int(max_tokens or self.max_tokens),
            "candidateCount": 1,
        }
        if self._thinking_budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": int(self._thinking_budget)}
        config.update(self._extra_config)

        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }
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
        """Stream deltas from ``streamGenerateContent``.

        Raises:
            PermanentError: 4xx other than 429.
            TransientError: 429, 5xx, timeouts and connection errors.
        """
        client = self._ensure_client()
        payload = self.build_payload(system, user, max_tokens=max_tokens, **overrides)
        try:
            async with client.stream("POST", self.url(), json=payload) as resp:
                status = int(resp.status_code)
                if status >= 400:
                    body = await _read_error_body(resp)
                    if status == 429 or status >= 500:
                        raise TransientError(f"gemini HTTP {status}: {body}")
                    raise PermanentError(f"gemini HTTP {status}: {body}")
                async for line in resp.aiter_lines():
                    for delta in _parse_sse_line(line, meta):
                        yield delta
        except (PermanentError, TransientError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalised to the taxonomy
            raise TransientError(f"gemini stream failed: {exc}") from exc


async def _read_error_body(resp: Any) -> str:
    """Read an error body without letting the read mask the status code."""
    try:
        raw = await resp.aread()
    except Exception:  # noqa: BLE001 - the status code is the useful part
        return ""
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", "replace")[:400]
    return str(raw)[:400]


def _parse_sse_line(line: str, meta: MutableMapping[str, Any]) -> list[str]:
    """Extract text deltas from one Gemini SSE line.

    Gemini splits a single logical delta across ``parts``, so one frame can
    carry several text fragments; they are returned in order and concatenating
    them reproduces the answer exactly.
    """
    if not line or line.startswith(":") or not line.startswith("data:"):
        return []
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return []
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return []
    if not isinstance(chunk, dict):
        return []

    if isinstance(chunk.get("modelVersion"), str):
        meta.setdefault("model", chunk["modelVersion"])
    usage = chunk.get("usageMetadata")
    if isinstance(usage, Mapping):
        meta["usage"] = dict(usage)

    out: list[str] = []
    for cand in chunk.get("candidates") or ():
        if not isinstance(cand, Mapping):
            continue
        reason = cand.get("finishReason")
        if isinstance(reason, str) and reason:
            # Normalised to the OpenAI vocabulary so callers -- and the
            # `truncated` property on GenerationResult -- do not need to know
            # which vendor answered.
            meta["finish_reason"] = _FINISH_REASONS.get(reason, reason.lower())
        content = cand.get("content")
        if not isinstance(content, Mapping):
            continue
        for part in content.get("parts") or ():
            if not isinstance(part, Mapping):
                continue
            # `thought` parts are the model's reasoning, not its answer. They
            # must never reach the user, and counting one as the first token
            # would report a TTFT for text nobody sees.
            if part.get("thought"):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                out.append(text)
    return out


#: Gemini's stop reasons mapped onto the OpenAI names the rest of the code uses.
_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
}
