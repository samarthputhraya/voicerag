"""ElevenLabs realtime speech-to-text: the STT fallback provider.

Second, not first, on purpose: Sarvam is hosted in India next to the users and
is tuned for Indic speech, which is what the MS MARCO-XI demo speaks. ElevenLabs
Scribe is the provider that answers when Sarvam does not -- a different company,
a different network path and a different failure domain, which is the only kind
of fallback that is actually worth having.

Because it implements the same :class:`~voicerag.stt.base.SttProvider` contract
as :class:`~voicerag.stt.sarvam.SarvamRealtimeStt`, it drops straight into
:func:`~voicerag.harness.resilience.first_healthy` with no adapter.

Wire protocol implemented here:

* ``wss://api.elevenlabs.io/v1/speech-to-text/realtime`` with
  ``model_id=scribe_v2_realtime``.
* Header ``xi-api-key: <key>``.
* Client -> server:
  ``{"message_type": "input_audio_chunk", "audio_base_64": "<base64 PCM>"}``.
* Server -> client: ``{"message_type": "partial_transcript" |
  "final_transcript" | "committed_transcript", "text": "..."}``.

Scribe distinguishes *final* (this utterance is decided) from *committed* (the
server will never revise it, even given later context). Both are surfaced as
``final`` events: downstream, a repeat final with identical text costs nothing
because the speculative driver keys its cache on normalised text and simply hits
it, whereas dropping the committed variant would lose the vendor's most accurate
output.
"""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.parse import urlencode

from ..harness.resilience import RetryPolicy
from ..harness.trace import current_trace
from .base import (
    ConnectFactory,
    TranscriptEvent,
    WebSocketSttProvider,
    b64,
    first_text,
)

__all__ = ["ElevenLabsRealtimeStt", "REALTIME_URL", "REALTIME_MODEL"]

REALTIME_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
REALTIME_MODEL = "scribe_v2_realtime"

#: Candidate paths for transcript text, most likely first. ElevenLabs documents
#: ``text`` at the top level; the alternates cost nothing and cover an envelope.
_TEXT_KEYS = ("text", "transcript", "data.text")


class ElevenLabsRealtimeStt(WebSocketSttProvider):
    """Streaming client for ElevenLabs Scribe realtime.

    Args:
        api_key: Sent as ``xi-api-key``.
        model_id: Realtime model. Only ``scribe_v2_realtime`` is a realtime
            model today; kept configurable so a newer one needs no code change.
        language_code: Optional ISO language hint. Omitted from the query string
            when ``None``, which is how Scribe is told to auto-detect.
        sample_rate: PCM sample rate of the frames passed to :meth:`stream`.
        encoding: PCM encoding label sent to the server.
        extra_params: Escape hatch for documented parameters this class does not
            model. Merged into the query string last, so it can also override.
        retry: Reconnection budget.
        connect: Injected connect factory; see
            :class:`~voicerag.stt.base.WebSocketSttProvider`.
        open_timeout_s: Handshake timeout.
        finalize_timeout_s: Grace period for the final transcript after audio
            ends.

    Raises:
        ValueError: If ``api_key`` is empty.
    """

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        *,
        model_id: str = REALTIME_MODEL,
        language_code: str | None = None,
        sample_rate: int = 16_000,
        encoding: str = "pcm_s16le_16",
        extra_params: Mapping[str, Any] | None = None,
        retry: RetryPolicy | None = None,
        connect: ConnectFactory | None = None,
        open_timeout_s: float = 5.0,
        finalize_timeout_s: float = 3.0,
    ) -> None:
        super().__init__(
            retry=retry,
            connect=connect,
            open_timeout_s=open_timeout_s,
            finalize_timeout_s=finalize_timeout_s,
        )
        if not api_key:
            raise ValueError("ElevenLabsRealtimeStt requires an api_key")
        self._api_key = api_key
        self._params: dict[str, Any] = {
            "model_id": model_id,
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        if language_code:
            self._params["language_code"] = language_code
        if extra_params:
            self._params.update(extra_params)

    # ------------------------------------------------------------ hooks

    def _url(self) -> str:
        return f"{REALTIME_URL}?{urlencode(self._params)}"

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._api_key}

    def _encode_frame(self, audio: bytes) -> str:
        """Serialise one PCM frame.

        Emits exactly
        ``{"message_type":"input_audio_chunk","audio_base_64":"<base64>"}``.
        Note the field is ``audio_base_64`` -- not ``audio``, not
        ``audio_base64`` -- which is precisely the sort of detail a test has to
        pin when the API cannot be reached from CI.
        """
        return json.dumps(
            {"message_type": "input_audio_chunk", "audio_base_64": b64(audio)},
            separators=(",", ":"),
        )

    def _parse(self, payload: Mapping[str, Any], t_ms: float) -> list[TranscriptEvent]:
        """Map one Scribe server message to transcript events."""
        kind = payload.get("message_type") or payload.get("type") or ""
        trace = current_trace()

        if kind == "partial_transcript":
            return [
                TranscriptEvent(
                    kind="partial",
                    text=first_text(payload, _TEXT_KEYS),
                    is_final=False,
                    t_ms=t_ms,
                    raw=payload,
                )
            ]
        if kind in ("final_transcript", "committed_transcript"):
            return [
                TranscriptEvent(
                    kind="final",
                    text=first_text(payload, _TEXT_KEYS),
                    is_final=True,
                    t_ms=t_ms,
                    raw=payload,
                )
            ]
        if kind == "error":
            # Scribe has no documented `is_fatal` flag, so every error is
            # treated as fatal for this socket: a provider that errors is
            # exactly when the fallback chain should move on, and this client is
            # already the last link in it.
            return [
                TranscriptEvent(
                    kind="error",
                    t_ms=t_ms,
                    raw={**payload, "is_fatal": bool(payload.get("is_fatal", True))},
                )
            ]
        if kind in ("session_end", "session.end", "close"):
            return [TranscriptEvent(kind="session_end", t_ms=t_ms, raw=payload)]
        if trace is not None and kind:
            trace.mark("stt.elevenlabs.unknown_event", event=str(kind))
        return []
