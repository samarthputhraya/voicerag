"""Sarvam speech-to-text: realtime WebSocket and batch REST clients.

Sarvam is the primary STT provider for this system for one architectural reason:
it is hosted in India, and so are the users. In the deployed topology the
*browser* holds this socket open directly to ``api.sarvam.ai`` and only the
resulting transcript crosses the ocean to our US-West API. Speech never
round-trips intercontinentally, and STT network time never lands in the server's
200 ms budget. This module is therefore used in two places: server-side for
batch and integration testing, and as the executable specification of the wire
protocol the browser client implements.

Wire protocol implemented here (documented values only, no invented fields):

* ``wss://api.sarvam.ai/speech-to-text-realtime/ws`` with the session configured
  entirely through query parameters.
* Header ``api-subscription-key: <key>``.
* Client -> server: ``{"event": "audio_input", "audio": "<base64 PCM>"}``.
* Server -> client: ``session.begin``, ``vad.speech_start``, ``vad.speech_end``,
  ``transcript.partial``, ``transcript.final``, ``error`` and ``session.end``.

Where the public documentation names an event but not the exact field carrying
its payload, this module reads an ordered list of candidate field names via
:func:`~voicerag.stt.base.first_text` rather than guessing one. Those candidate
lists are asserted in the tests, so the guess is explicit, reviewable and a
one-line change once the live API confirms the true spelling.
"""

from __future__ import annotations

import contextlib
import json
import struct
from typing import Any, Literal, Mapping
from urllib.parse import urlencode

from ..harness.resilience import PermanentError, RetryPolicy, TransientError
from ..harness.trace import current_trace
from .base import (
    AudioTooLongError,
    ConnectFactory,
    SttFatalError,
    SttProvider,
    TranscriptEvent,
    WebSocketSttProvider,
    b64,
    first_text,
)

__all__ = [
    "SarvamRealtimeStt",
    "SarvamBatchStt",
    "REALTIME_URL",
    "BATCH_URL",
    "REALTIME_MODEL",
    "BATCH_MAX_SECONDS",
    "audio_duration_s",
]

REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
BATCH_URL = "https://api.sarvam.ai/speech-to-text"

#: The only value the realtime endpoint accepts. Hard-coded as a default rather
#: than a free-form string so a typo fails at import review, not in the demo.
REALTIME_MODEL = "saaras:v3-realtime"

#: Sarvam's batch endpoint rejects audio at or beyond 30 s. Enforced client-side
#: so the failure is a clear local exception instead of an opaque HTTP 400 spent
#: after a full upload.
BATCH_MAX_SECONDS = 30.0

StreamType = Literal["fast", "balanced", "simulated"]
Endpointing = Literal["vad", "manual"]
Encoding = Literal["linear16", "linear32", "mulaw", "alaw"]

_VALID_STREAM_TYPES = ("fast", "balanced", "simulated")
_VALID_ENDPOINTING = ("vad", "manual")
_VALID_ENCODINGS = ("linear16", "linear32", "mulaw", "alaw")
_VALID_SAMPLE_RATES = (16_000, 8_000)

#: Candidate paths for transcript text, most likely first. See the module
#: docstring: preference, not invention.
_TEXT_KEYS = ("data.transcript", "data.text", "transcript", "text")

#: Bytes per sample for each documented encoding, used by the duration guard.
_ENCODING_WIDTH = {"linear16": 2, "linear32": 4, "mulaw": 1, "alaw": 1}


class SarvamRealtimeStt(WebSocketSttProvider):
    """Streaming client for Sarvam's realtime speech-to-text socket.

    Args:
        api_key: Sent as ``api-subscription-key``. Required; an empty key is
            rejected here rather than producing a 401 three seconds into a demo.
        language_code: BCP-47 tag (``"hi-IN"``, ``"en-IN"``) or ``"auto"``.
            Defaults to ``"auto"`` because the MS MARCO-XI demo is deliberately
            cross-lingual and the speaker's language is not known in advance.
        stream_type: ``"fast"`` trades a little accuracy for roughly 150 ms
            time-to-first-partial, which is the whole point of speculative
            retrieval downstream; ``"balanced"`` is the accuracy-first setting.
        endpointing: ``"vad"`` lets the server close the utterance on silence.
        encoding: PCM encoding of the frames passed to :meth:`stream`.
        sample_rate: 16000 or 8000 only; anything else is rejected by the API.
        mode: Optional vendor-specific mode string, sent only when provided.
        threshold: VAD speech probability threshold. ``None`` uses the server
            default (0.3).
        silence_duration_ms: Silence before the server ends an utterance.
            ``None`` uses the server default (500 ms).
        min_speech_duration_ms: Minimum speech before an utterance starts.
            ``None`` uses the server default (250 ms).
        return_timestamps: Request word timestamps. They arrive in
            :attr:`TranscriptEvent.raw` and are never parsed by the pipeline.
        prompt: Optional biasing prompt (domain vocabulary, named entities).
        retry: Reconnection budget.
        connect: Injected connect factory; see
            :class:`~voicerag.stt.base.WebSocketSttProvider`.
        open_timeout_s: Handshake timeout.
        finalize_timeout_s: Grace period for the final transcript after audio
            ends.

    Raises:
        ValueError: For any parameter outside the API's documented enum. These
            are validated in ``__init__``, before a socket is opened, because a
            rejected handshake costs a full RTT to discover.

    Note:
        Optional tuning parameters default to ``None`` and are then *omitted*
        from the query string. Sending a documented default explicitly gains
        nothing and adds one more field name that could be wrong.
    """

    name = "sarvam"

    def __init__(
        self,
        api_key: str,
        *,
        language_code: str = "auto",
        stream_type: StreamType = "fast",
        endpointing: Endpointing = "vad",
        encoding: Encoding = "linear16",
        sample_rate: int = 16_000,
        mode: str | None = None,
        threshold: float | None = None,
        silence_duration_ms: int | None = None,
        min_speech_duration_ms: int | None = None,
        return_timestamps: bool | None = None,
        prompt: str | None = None,
        model: str = REALTIME_MODEL,
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
            raise ValueError("SarvamRealtimeStt requires an api_key")
        if stream_type not in _VALID_STREAM_TYPES:
            raise ValueError(
                f"stream_type must be one of {_VALID_STREAM_TYPES}, got {stream_type!r}"
            )
        if endpointing not in _VALID_ENDPOINTING:
            raise ValueError(
                f"endpointing must be one of {_VALID_ENDPOINTING}, got {endpointing!r}"
            )
        if encoding not in _VALID_ENCODINGS:
            raise ValueError(
                f"encoding must be one of {_VALID_ENCODINGS}, got {encoding!r}"
            )
        if sample_rate not in _VALID_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {_VALID_SAMPLE_RATES}, got {sample_rate}"
            )

        self._api_key = api_key
        self._model = model
        self._params: dict[str, Any] = {
            "model": model,
            "language_code": language_code,
            "stream_type": stream_type,
            "endpointing": endpointing,
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        optional = {
            "mode": mode,
            "threshold": threshold,
            "silence_duration_ms": silence_duration_ms,
            "min_speech_duration_ms": min_speech_duration_ms,
            "return_timestamps": return_timestamps,
            "prompt": prompt,
        }
        for key, value in optional.items():
            if value is None:
                continue
            # Query strings have no booleans; the wire form is lowercase JSON
            # style, which is what every HTTP framework parses back to a bool.
            self._params[key] = "true" if value is True else "false" if value is False else value

    # ------------------------------------------------------------ hooks

    def _url(self) -> str:
        return f"{REALTIME_URL}?{urlencode(self._params)}"

    def _headers(self) -> dict[str, str]:
        return {"api-subscription-key": self._api_key}

    def _encode_frame(self, audio: bytes) -> str:
        """Serialise one PCM frame.

        Emits exactly ``{"event": "audio_input", "audio": "<base64>"}`` with no
        whitespace and in that key order. Separators are pinned because the
        frame is asserted byte-for-byte in the tests -- that assertion is the
        only defence this repo has against a protocol drift it cannot detect by
        calling the live API.
        """
        return json.dumps(
            {"event": "audio_input", "audio": b64(audio)}, separators=(",", ":")
        )

    def _parse(self, payload: Mapping[str, Any], t_ms: float) -> list[TranscriptEvent]:
        """Map one Sarvam server message to transcript events.

        The discriminator is read from ``type`` and then ``event``: Sarvam's
        client frames use ``event`` and its docs name the server events, so
        accepting either spelling costs one ``or`` and removes a whole class of
        silent failure.
        """
        kind = payload.get("type") or payload.get("event") or ""
        trace = current_trace()

        if kind == "transcript.partial":
            return [
                TranscriptEvent(
                    kind="partial",
                    text=first_text(payload, _TEXT_KEYS),
                    is_final=False,
                    t_ms=t_ms,
                    raw=payload,
                )
            ]
        if kind == "transcript.final":
            return [
                TranscriptEvent(
                    kind="final",
                    text=first_text(payload, _TEXT_KEYS),
                    is_final=True,
                    t_ms=t_ms,
                    raw=payload,
                )
            ]
        if kind == "vad.speech_start":
            return [TranscriptEvent(kind="speech_start", t_ms=t_ms, raw=payload)]
        if kind == "vad.speech_end":
            return [TranscriptEvent(kind="speech_end", t_ms=t_ms, raw=payload)]
        if kind == "session.end":
            return [TranscriptEvent(kind="session_end", t_ms=t_ms, raw=payload)]
        if kind == "error":
            return [
                TranscriptEvent(kind="error", t_ms=t_ms, raw=_normalise_error(payload))
            ]
        if kind == "session.begin":
            # Purely informational: the socket is already open by definition.
            # Recorded so the trace shows when the vendor accepted the session,
            # which is the difference between "connected" and "usable".
            if trace is not None:
                trace.mark("stt.sarvam.session_begin", t_ms=round(t_ms, 2))
            return []
        # Unknown event: ignore rather than fail. A vendor adding an event type
        # must not take down a running session.
        if trace is not None and kind:
            trace.mark("stt.sarvam.unknown_event", event=str(kind))
        return []


def _normalise_error(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten Sarvam's ``error`` payload to ``code`` / ``is_fatal`` / ``message``.

    The three fields are documented but their nesting is not, so this lifts them
    from either the top level or a ``data`` envelope. ``is_fatal`` defaults to
    False: an error we cannot classify should degrade the session, not kill it,
    and a genuinely dead socket will announce itself soon enough.
    """
    data = payload.get("data")
    nested: Mapping[str, Any] = data if isinstance(data, Mapping) else {}
    merged = dict(payload)
    for key in ("code", "is_fatal", "message"):
        value = payload.get(key, nested.get(key))
        if value is not None:
            merged[key] = value
    merged["is_fatal"] = bool(merged.get("is_fatal", False))
    return merged


# --- batch --------------------------------------------------------------------

def audio_duration_s(
    audio: bytes,
    *,
    sample_rate: int = 16_000,
    sample_width: int = 2,
    channels: int = 1,
) -> float:
    """Duration of an audio buffer in seconds.

    Understands RIFF/WAVE headers and falls back to treating the buffer as raw
    PCM with the supplied geometry. Getting this wrong in the *permissive*
    direction would defeat the 30 s guard, so an unparseable WAV header falls
    back to the raw-PCM computation over the whole buffer -- which over-counts
    by the header size and therefore errs toward rejecting, not accepting.

    Args:
        audio: WAV file bytes or raw interleaved PCM.
        sample_rate: Sample rate assumed for raw PCM.
        sample_width: Bytes per sample per channel assumed for raw PCM.
        channels: Channel count assumed for raw PCM.

    Returns:
        Duration in seconds; 0.0 for an empty buffer.
    """
    if not audio:
        return 0.0
    parsed = _parse_wav_geometry(audio)
    if parsed is not None:
        n_bytes, rate, width, chans = parsed
    else:
        n_bytes, rate, width, chans = len(audio), sample_rate, sample_width, channels
    denom = max(1, rate * width * chans)
    return n_bytes / denom


def _parse_wav_geometry(audio: bytes) -> tuple[int, int, int, int] | None:
    """Extract ``(data_bytes, sample_rate, sample_width, channels)`` from a WAV.

    Returns None when the buffer is not a RIFF/WAVE file, which is the common
    case for browser-captured PCM.
    """
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return None
    pos = 12
    channels = sample_rate = width = 0
    while pos + 8 <= len(audio):
        chunk_id = audio[pos : pos + 4]
        (size,) = struct.unpack_from("<I", audio, pos + 4)
        body = pos + 8
        if chunk_id == b"fmt " and body + 16 <= len(audio):
            _, channels, sample_rate, _, _, bits = struct.unpack_from(
                "<HHIIHH", audio, body
            )
            width = max(1, bits // 8)
        elif chunk_id == b"data":
            if not (channels and sample_rate and width):
                return None
            return (min(size, len(audio) - body), sample_rate, width, channels)
        pos = body + size + (size & 1)  # RIFF chunks are word-aligned
    return None


class SarvamBatchStt(SttProvider):
    """One-shot REST client for Sarvam's batch speech-to-text endpoint.

    Used for pre-recorded audio, for the offline evaluation harness and as the
    non-streaming fallback when a realtime socket cannot be opened at all. It
    cannot be used for the live demo path: it returns text only after the whole
    utterance has been uploaded and decoded, which is exactly the latency the
    realtime socket exists to remove.

    Args:
        api_key: Sent as ``api-subscription-key``.
        model: ``"saaras:v3"`` or ``"saaras:v4"``.
        language_code: Optional BCP-47 hint sent as a form field.
        client: Injected ``httpx.AsyncClient``. Supplying one is how the tests
            run offline and how a deployment shares a warm connection pool.
        base_url: Override for the endpoint, for staging or a local mock.
        timeout_s: Request timeout.
        sample_rate: Geometry assumed when the payload is raw PCM rather than a
            WAV file, used only by the duration guard.
        sample_width: Bytes per sample assumed for raw PCM.
        channels: Channels assumed for raw PCM.
    """

    name = "sarvam-batch"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "saaras:v3",
        language_code: str | None = None,
        client: Any | None = None,
        base_url: str = BATCH_URL,
        timeout_s: float = 15.0,
        sample_rate: int = 16_000,
        sample_width: int = 2,
        channels: int = 1,
    ) -> None:
        if not api_key:
            raise ValueError("SarvamBatchStt requires an api_key")
        self._api_key = api_key
        self._model = model
        self._language_code = language_code
        self._client = client
        self._base_url = base_url
        self._timeout_s = timeout_s
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self._channels = channels

    def stream(self, audio_chunks: Any) -> Any:
        """Not supported: this endpoint has no streaming mode.

        Raises:
            SttFatalError: Always. Failing loudly beats silently buffering the
                whole utterance and pretending to be a realtime provider, which
                would quietly blow the latency budget.
        """
        raise SttFatalError(
            "SarvamBatchStt is a batch endpoint; use SarvamRealtimeStt for streaming"
        )

    async def transcribe(self, audio: bytes) -> str:
        """Upload one utterance and return its transcript.

        Args:
            audio: WAV file bytes or raw PCM, strictly under
                :data:`BATCH_MAX_SECONDS`.

        Returns:
            The transcript, or ``""`` if the response carried no text.

        Raises:
            AudioTooLongError: Audio is at or over the 30 s limit.
            SttFatalError: Auth or request rejected (4xx other than 429).
            TransientError: Timeout, 429 or 5xx -- retryable by the harness.
        """
        duration = audio_duration_s(
            audio,
            sample_rate=self._sample_rate,
            sample_width=self._sample_width,
            channels=self._channels,
        )
        if duration >= BATCH_MAX_SECONDS:
            raise AudioTooLongError(
                f"sarvam batch STT accepts audio under {BATCH_MAX_SECONDS:g}s, "
                f"got {duration:.2f}s -- split the audio or use SarvamRealtimeStt",
                code="audio_too_long",
            )

        client = self._client or _default_httpx_client(self._timeout_s)
        data = {"model": self._model}
        if self._language_code:
            data["language_code"] = self._language_code

        trace = current_trace()
        with contextlib.ExitStack() as stack:
            if trace is not None:
                stack.enter_context(trace.span("stt.sarvam_batch", model=self._model))
            try:
                resp = await client.post(
                    self._base_url,
                    headers={"api-subscription-key": self._api_key},
                    files={"file": ("audio.wav", audio, "audio/wav")},
                    data=data,
                )
            except Exception as exc:  # noqa: BLE001 - mapped to the harness taxonomy
                raise TransientError(
                    f"sarvam batch STT request failed: {exc}"
                ) from exc

        _raise_for_status(resp)
        payload = resp.json()
        if not isinstance(payload, Mapping):
            return ""
        return first_text(payload, ("transcript", "text", "data.transcript")).strip()


def _raise_for_status(resp: Any) -> None:
    """Translate an HTTP status into the harness error taxonomy.

    429 and 5xx are transient (worth a retry inside the deadline); every other
    4xx is permanent, because a malformed request or a bad key fails identically
    on the second attempt and retrying it only spends budget.
    """
    status = int(getattr(resp, "status_code", 0))
    if status < 400:
        return
    body = getattr(resp, "text", "")[:200]
    if status == 429 or status >= 500:
        raise TransientError(f"sarvam STT HTTP {status}: {body}")
    raise SttFatalError(f"sarvam STT HTTP {status}: {body}", code=str(status))


def _default_httpx_client(timeout_s: float) -> Any:
    """Build a default async HTTP client.

    Lazy import: the batch path is optional, and this package must import
    cleanly in a deployment that only ever uses the realtime socket.

    Raises:
        PermanentError: If :mod:`httpx` is not installed.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - only without the dependency
        raise PermanentError(
            "batch STT needs the 'httpx' package: pip install httpx"
        ) from exc
    return httpx.AsyncClient(timeout=timeout_s)
