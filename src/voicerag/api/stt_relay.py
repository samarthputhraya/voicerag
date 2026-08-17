"""Server-side relay between the browser and Sarvam's realtime STT socket.

Why this exists
---------------

The original design had the browser dial ``wss://api.sarvam.ai`` directly,
carrying a short-lived credential minted by ``POST /stt/token``. That is the
right shape -- it is how OpenAI Realtime and ElevenLabs are meant to be used --
but it rests on a premise Sarvam does not satisfy:

* Sarvam's realtime endpoint accepts **no ``token`` query parameter**. The
  documented mechanisms are an ``api-subscription-key`` header on the upgrade
  request, or an ``api-subscription-key.<key>`` WebSocket subprotocol.
* Sarvam publishes **no ephemeral-token endpoint**. There is nothing to mint.

So a browser can only authenticate by carrying the permanent account key itself,
in devtools, with no expiry and no scope. :mod:`voicerag.api.stt_token` already
said this in as many words: capability mode "presumes a relay". This is that
relay -- the thing that makes the security boundary real rather than aspirational.

The cost is honest and worth stating: one extra network hop per audio frame,
between the browser and this process. When the API runs on the same machine as
the browser -- the demo case -- that hop is loopback and costs nothing
measurable. When the API is deployed far from the user it is a real tax, and the
alternative is publishing the account key. We pay the tax.

Wire protocol
-------------

Deliberately **identical** to Sarvam's own, in both directions, so the browser
client is the same client either way and this module stays a pump rather than a
translator. Client sends ``{"event": "audio_input", "audio": "<base64>"}`` and
friends; server sends back ``transcript.partial`` / ``transcript.final`` /
``vad.*`` / ``error`` / ``session.*`` exactly as received.

The one addition is a synthetic ``{"event": "relay.error", ...}`` frame, emitted
when the relay itself cannot reach Sarvam. It is namespaced so a client can tell
"the vendor rejected this" from "your own server could not connect".
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import urlencode

from fastapi import WebSocket, WebSocketDisconnect

from ..config import Settings
from ..stt.sarvam import REALTIME_URL

__all__ = ["relay_sarvam"]

log = logging.getLogger("voicerag.api.stt_relay")

#: Query parameters a client may override per session. Everything else is
#: server-chosen. Whitelisted rather than passed through so a client cannot
#: redirect the socket or silently switch to a model we have not budgeted for.
_CLIENT_PARAMS = frozenset(
    {
        "language_code",
        "stream_type",
        "endpointing",
        "mode",
        "threshold",
        "silence_duration_ms",
        "min_speech_duration_ms",
        "prefix_padding_ms",
        "return_timestamps",
    }
)

#: Audio geometry is fixed by what the browser's AudioWorklet produces. Letting
#: a client declare a sample rate the audio does not have yields a transcript
#: that is confidently wrong, which is worse than an error.
_FIXED_PARAMS = {"encoding": "linear16", "sample_rate": "16000"}


def _upstream_url(cfg: Settings, client_params: dict[str, str]) -> str:
    """Build the Sarvam socket URL for one session."""
    params: dict[str, str] = {
        "model": cfg.sarvam_model,
        "language_code": cfg.stt_language,
        "stream_type": cfg.stt_stream_type,
        "endpointing": "vad",
        **_FIXED_PARAMS,
    }
    for key, value in client_params.items():
        if key in _CLIENT_PARAMS and value:
            params[key] = value
    return f"{REALTIME_URL}?{urlencode(params)}"


async def relay_sarvam(ws: WebSocket, cfg: Settings) -> None:
    """Accept a browser socket and pump it to Sarvam until either side closes.

    Args:
        ws: The browser-facing WebSocket, not yet accepted.
        cfg: Settings supplying the account key and model defaults.
    """
    await ws.accept()

    if cfg.sarvam_api_key is None:
        await ws.send_json(
            {
                "event": "relay.error",
                "code": "stt_unconfigured",
                "is_fatal": True,
                "message": "SARVAM_API_KEY is not set on the server",
            }
        )
        await ws.close(code=1011)
        return

    url = _upstream_url(cfg, dict(ws.query_params))
    key = cfg.sarvam_api_key.get_secret_value()

    try:
        from websockets.asyncio.client import connect
    except Exception as exc:  # pragma: no cover - dependency is declared
        log.error("websockets client unavailable: %s", exc)
        await ws.send_json(
            {
                "event": "relay.error",
                "code": "relay_unavailable",
                "is_fatal": True,
                "message": "server is missing its websocket client",
            }
        )
        await ws.close(code=1011)
        return

    try:
        upstream_cm = connect(
            url,
            additional_headers={"api-subscription-key": key},
            open_timeout=10.0,
            max_size=None,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the client below
        log.warning("sarvam connect failed: %s", exc)
        await ws.send_json(
            {
                "event": "relay.error",
                "code": "upstream_unreachable",
                "is_fatal": True,
                "message": str(exc),
            }
        )
        await ws.close(code=1011)
        return

    try:
        async with upstream_cm as upstream:
            log.info("stt relay open")
            await _pump(ws, upstream)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to a client
        log.warning("stt relay failed: %s: %s", type(exc).__name__, exc)
        try:
            await ws.send_json(
                {
                    "event": "relay.error",
                    "code": "upstream_failed",
                    "is_fatal": True,
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        except Exception:  # noqa: BLE001 - client already gone
            pass
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001 - already closed
            pass


async def _pump(ws: WebSocket, upstream: Any) -> None:
    """Run both directions concurrently until either finishes.

    The first task to finish cancels the other. Both directions are equally
    authoritative about the session being over: the browser hanging up and Sarvam
    sending ``session.end`` are both terminal, and waiting for the second one
    after the first would leak a socket per session.
    """

    async def browser_to_sarvam() -> None:
        while True:
            message = await ws.receive()
            kind = message.get("type")
            if kind == "websocket.disconnect":
                return
            text = message.get("text")
            if text is not None:
                await upstream.send(text)
                continue
            data = message.get("bytes")
            if data:
                # A client that sends raw PCM instead of the JSON envelope is
                # accommodated rather than rejected: base64-framing it here is
                # two lines and removes a whole class of client-side bug.
                import base64
                import json

                await upstream.send(
                    json.dumps(
                        {"event": "audio_input", "audio": base64.b64encode(data).decode()},
                        separators=(",", ":"),
                    )
                )

    async def sarvam_to_browser() -> None:
        async for raw in upstream:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            await ws.send_text(raw)

    tasks = [
        asyncio.create_task(browser_to_sarvam(), name="stt-relay-up"),
        asyncio.create_task(sarvam_to_browser(), name="stt-relay-down"),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        # Surface a genuine failure; a normal close is not one.
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                raise exc
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
