"""Speech-to-text interface and the shared realtime-WebSocket machinery.

Every STT backend -- Sarvam, ElevenLabs, or a fake in a test -- speaks one
contract::

    async for event in provider.stream(audio_chunks): ...
    text = await provider.transcribe(pcm_bytes)

That uniformity is what lets :func:`voicerag.harness.resilience.first_healthy`
treat STT providers as interchangeable and fail over between them without the
caller knowing which vendor answered.

Two design decisions here are load-bearing:

**Transports are injected, never constructed implicitly.** Each provider takes a
``connect`` factory rather than reaching for :mod:`websockets` itself. That makes
the entire wire protocol -- the exact query string, the exact headers, the exact
JSON frames we put on the socket -- assertable in an offline unit test. A vendor
WebSocket API cannot be mocked at the HTTP layer and cannot be reached from CI,
so if the frames are not pinned by a test they are pinned by nothing.

**The event stream is a flat union, not a hierarchy.** Realtime STT emits VAD
edges, revisable partials, immutable finals, recoverable errors and fatal errors
over one socket. Modelling those as one :class:`TranscriptEvent` with a ``kind``
discriminator means a consumer can ignore what it does not care about (a batch
caller keeps only finals) while the speculative driver sees every partial. A
class-per-event-type hierarchy would force every consumer to write dispatch code
for events it does not use.

Timing note: ``t_ms`` is measured from the moment :meth:`SttProvider.stream` was
entered, using ``perf_counter_ns``. It is the arrival time of the event relative
to the start of capture and is what the demo HUD plots; it is deliberately
independent of any :class:`~voicerag.harness.trace.Trace` so that STT timing
stays meaningful even when the caller never opened one.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

from ..harness.resilience import PermanentError, RetryPolicy, TransientError
from ..harness.trace import current_trace

__all__ = [
    "EventKind",
    "TranscriptEvent",
    "SttError",
    "SttFatalError",
    "AudioTooLongError",
    "SttProvider",
    "WebSocketLike",
    "ConnectFactory",
    "WebSocketSttProvider",
    "default_connect_factory",
    "first_text",
    "frame_pcm",
    "iter_chunks",
    "b64",
]

#: The discriminator on :class:`TranscriptEvent`.
#:
#: ``partial``       -- revisable hypothesis; text may shrink or change entirely.
#: ``final``         -- committed segment; never revised by the vendor.
#: ``speech_start``  -- VAD detected the onset of speech.
#: ``speech_end``    -- VAD detected the end of an utterance.
#: ``error``         -- vendor-reported error; see ``raw['is_fatal']``.
#: ``session_end``   -- the vendor closed the session cleanly.
EventKind = Literal[
    "partial", "final", "speech_start", "speech_end", "error", "session_end"
]


@dataclass(slots=True, frozen=True)
class TranscriptEvent:
    """One event from a speech-to-text stream.

    Frozen because events fan out to several consumers at once (the speculative
    driver, the trace, the socket that mirrors them to the browser) and a
    mutable event would let one consumer silently rewrite another's input.

    Args:
        kind: Which sort of event this is; see :data:`EventKind`.
        text: Transcript text. Empty for VAD and lifecycle events.
        is_final: Whether ``text`` is committed. Mirrors ``kind == "final"``;
            kept as an explicit field because consumers overwhelmingly branch on
            this one bit and ``ev.is_final`` reads better than a string compare.
        t_ms: Milliseconds since the stream was opened.
        raw: The vendor payload, verbatim. Never interpreted by the pipeline,
            but carried so that a demo, a bug report or a post-hoc analysis can
            see exactly what the vendor said -- including fields we chose not to
            model, such as word timestamps or ``audio_duration_s``.
    """

    kind: EventKind
    text: str = ""
    is_final: bool = False
    t_ms: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_fatal(self) -> bool:
        """True for an error the vendor marked unrecoverable.

        Non-fatal errors are informational (a dropped audio frame, a decode
        warning) and the session continues; fatal ones mean the socket is dead
        and the caller must fail over.
        """
        return self.kind == "error" and bool(self.raw.get("is_fatal", False))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view, used by the API layer and the trace dump."""
        return {
            "kind": self.kind,
            "text": self.text,
            "is_final": self.is_final,
            "t_ms": round(self.t_ms, 3),
            "raw": dict(self.raw),
        }


class SttError(TransientError):
    """A recoverable STT failure: socket drop, non-fatal vendor error, timeout.

    Subclasses :class:`~voicerag.harness.resilience.TransientError` so the
    harness retries it without the caller translating error taxonomies.
    """


class SttFatalError(PermanentError):
    """An unrecoverable STT failure: bad credentials, rejected model, fatal event.

    Permanent by construction -- retrying a 401 or an unsupported ``sample_rate``
    burns latency budget that a genuine fallback needed.
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class AudioTooLongError(SttFatalError):
    """Audio exceeded a batch endpoint's hard duration limit.

    Its own type because the remedy is specific and mechanical -- chunk the
    audio, or use the realtime socket -- and callers should branch on that
    rather than on a message string.
    """


# --- audio helpers ------------------------------------------------------------

def b64(data: bytes) -> str:
    """Base64-encode raw audio for a JSON text frame.

    Both vendors carry PCM inside JSON rather than as binary WebSocket frames,
    which costs 33 % bandwidth. That is their protocol, not our choice; the
    encoding lives in one function so the two clients cannot drift.
    """
    return base64.b64encode(data).decode("ascii")


def frame_pcm(
    audio: bytes,
    *,
    frame_ms: int = 100,
    sample_rate: int = 16_000,
    sample_width: int = 2,
    channels: int = 1,
) -> list[bytes]:
    """Split PCM into fixed-duration frames on a sample boundary.

    Frame size is a latency knob, not a formality. Too large and VAD cannot
    detect the end of speech until the frame lands, adding up to a full frame of
    dead time before the final transcript; too small and per-frame JSON+base64
    overhead dominates. 100 ms is the usual compromise and matches both vendors'
    guidance for 16 kHz linear16.

    Splitting is clamped to a whole number of sample blocks: a frame cut
    mid-sample decodes as noise.

    Args:
        audio: Raw interleaved PCM, no container header.
        frame_ms: Target frame duration in milliseconds.
        sample_rate: Samples per second.
        sample_width: Bytes per sample per channel (2 for ``linear16``).
        channels: Interleaved channel count.

    Returns:
        Frames in order; the last may be short. Empty input yields ``[b""]`` so
        callers always get a well-formed, non-empty frame list.

    Raises:
        ValueError: If ``frame_ms`` is not positive.
    """
    if frame_ms <= 0:
        raise ValueError(f"frame_ms must be > 0, got {frame_ms}")
    block = sample_width * channels
    size = max(block, (sample_rate * frame_ms // 1000) * block)
    return [audio[i : i + size] for i in range(0, len(audio), size)] or [b""]


async def iter_chunks(chunks: Iterable[bytes]) -> AsyncIterator[bytes]:
    """Adapt a synchronous iterable of audio frames to the async interface.

    Used by :meth:`SttProvider.transcribe` and by tests that hold a whole
    utterance in memory; a live microphone feeds :meth:`SttProvider.stream`
    directly.
    """
    for c in chunks:
        yield c


def first_text(payload: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Return the first present, non-empty string among ``keys``.

    Nested keys use dots (``"data.transcript"``). This exists because the
    partial/final payload *field names* are the part of these vendors' protocols
    least pinned by public documentation, and a client that hard-codes one
    spelling silently transcribes every utterance as the empty string. Ordered
    fallback turns a schema guess into a schema *preference*, and the ordering is
    asserted in tests so the intent is explicit rather than accidental.

    Args:
        payload: One decoded server message.
        keys: Candidate field paths, most-preferred first.

    Returns:
        The first non-blank string found, or ``""``.
    """
    for key in keys:
        cur: Any = payload
        for part in key.split("."):
            if not isinstance(cur, Mapping):
                cur = None
                break
            cur = cur.get(part)
        if isinstance(cur, str) and cur.strip():
            return cur
    return ""


# --- provider interface -------------------------------------------------------

class SttProvider(ABC):
    """Base class for every speech-to-text backend."""

    #: Stable short identifier used in traces, breakers and fallback tables.
    name: str = "stt"

    @abstractmethod
    def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        """Transcribe a live audio stream.

        Implemented as an async generator, so the call itself is not awaited::

            async for event in provider.stream(mic()):
                ...

        The generator owns the socket: closing it -- breaking out of the ``async
        for``, or an exception in the consumer -- tears the connection down.

        Args:
            audio_chunks: Raw audio frames in the provider's configured
                encoding. Exhaustion signals end of capture.

        Yields:
            :class:`TranscriptEvent` in arrival order.
        """

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe one complete utterance and return the committed text.

        The default implementation drives :meth:`stream` and concatenates final
        segments, so realtime providers get a batch API for free. Backends with
        a genuine batch endpoint (see
        :class:`~voicerag.stt.sarvam.SarvamBatchStt`) override this.

        Args:
            audio: The complete utterance as raw PCM in the configured encoding.

        Returns:
            The concatenation of every ``final`` segment, whitespace-normalised.

        Raises:
            SttFatalError: If the provider reported a fatal error.
        """
        parts: list[str] = []
        async for ev in self.stream(iter_chunks(frame_pcm(audio))):
            if ev.is_final and ev.text:
                parts.append(ev.text)
        return " ".join(" ".join(parts).split())


# --- websocket transport ------------------------------------------------------

@runtime_checkable
class WebSocketLike(Protocol):
    """The slice of a WebSocket connection this package actually uses.

    Deliberately three methods wide. A test double implements exactly this, and
    :mod:`websockets`' ``ClientConnection`` already satisfies it, so the fake
    and the real thing are interchangeable with no adapter in between.
    """

    async def send(self, message: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


#: Opens a connection: ``(url, headers, open_timeout_s)`` returning an async
#: context manager that yields a :class:`WebSocketLike`. Mirrors the shape of
#: ``websockets.asyncio.client.connect`` so the production factory is trivial.
ConnectFactory = Callable[[str, Mapping[str, str], float], Any]


def default_connect_factory() -> ConnectFactory:
    """Return a connect factory backed by the :mod:`websockets` library.

    Imported lazily: the package is only needed when someone actually opens a
    realtime socket, and every test in this repo injects a fake instead. Keeping
    the import out of module scope lets this package be imported in an
    environment where ``websockets`` is not installed.

    Raises:
        SttFatalError: If :mod:`websockets` is not installed.
    """
    try:
        from websockets.asyncio.client import connect as _connect
    except ImportError as exc:  # pragma: no cover - only without the dependency
        raise SttFatalError(
            "realtime STT needs the 'websockets' package: pip install websockets"
        ) from exc

    def factory(url: str, headers: Mapping[str, str], open_timeout_s: float) -> Any:
        return _connect(
            url,
            additional_headers=dict(headers),
            open_timeout=open_timeout_s,
            # Vendor sockets carry small JSON frames, so the 1 MiB default frame
            # cap is ample -- but an unbounded receive queue is not: a stalled
            # consumer must exert backpressure rather than grow the heap.
            max_queue=32,
        )

    return factory


class WebSocketSttProvider(SttProvider):
    """Shared realtime-WebSocket pump: connect, duplex, reconnect, tear down.

    Sarvam and ElevenLabs differ only in their URL, their auth header and their
    JSON schemas. Everything that is genuinely hard -- running the sender and
    receiver concurrently, not leaking a task when the consumer walks away,
    deciding whether a drop is worth reconnecting, and not hanging forever once
    the audio runs out -- is identical, subtle, and implemented once here.

    Subclasses supply four hooks: :meth:`_url`, :meth:`_headers`,
    :meth:`_encode_frame` and :meth:`_parse`.

    **On reconnection.** A socket dropped mid-utterance cannot be replayed: the
    audio is live, it was consumed from the microphone iterator, and it is gone.
    So reconnection *resumes* rather than *replays* -- we reopen the socket and
    keep feeding the live stream, accepting that audio in flight during the drop
    is lost. Text already emitted as ``final`` is never re-emitted. This is the
    honest behaviour for live capture; the alternative is buffering an unbounded
    utterance for a replay the vendor's protocol cannot accept anyway.

    Args:
        retry: Reconnection budget. ``attempts`` counts total connection
            attempts, and backoff comes from the same policy the rest of the
            harness uses.
        connect: Injected connect factory. Defaults to the :mod:`websockets`
            client, built lazily on first use.
        open_timeout_s: Handshake timeout.
        finalize_timeout_s: How long to keep receiving after the audio iterator
            is exhausted, waiting for the vendor's last final / session end.
            Uncapped, a vendor that never closes the socket hangs the request
            forever; too tight, and the last word of the utterance is lost.
    """

    def __init__(
        self,
        *,
        retry: RetryPolicy | None = None,
        connect: ConnectFactory | None = None,
        open_timeout_s: float = 5.0,
        finalize_timeout_s: float = 3.0,
    ) -> None:
        self._retry = retry or RetryPolicy(attempts=2, base_ms=50, max_ms=400)
        self._connect = connect
        self._open_timeout_s = open_timeout_s
        self._finalize_timeout_s = finalize_timeout_s

    # ------------------------------------------------------------ hooks

    @abstractmethod
    def _url(self) -> str:
        """Full connection URL including the vendor's query parameters."""

    @abstractmethod
    def _headers(self) -> dict[str, str]:
        """Auth and protocol headers for the handshake."""

    @abstractmethod
    def _encode_frame(self, audio: bytes) -> str:
        """Serialise one audio frame into the vendor's client message."""

    @abstractmethod
    def _parse(self, payload: Mapping[str, Any], t_ms: float) -> list[TranscriptEvent]:
        """Translate one vendor server message into zero or more events."""

    def _close_frame(self) -> str | None:
        """Optional end-of-audio message, or None if the vendor documents none.

        Defaulting to None is a deliberate refusal to guess: an unrecognised
        client event can abort the session server-side and lose the final
        transcript, which is strictly worse than sending nothing and letting VAD
        endpointing close the utterance.
        """
        return None

    # ------------------------------------------------------------ pump

    async def stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptEvent]:
        """Duplex the audio stream against the vendor socket.

        Reconnects on transient failures up to the retry policy, and re-raises
        :class:`SttFatalError` immediately -- a rejected API key or model is
        rejected exactly as fast the second time.

        Args:
            audio_chunks: Live audio frames. Deliberately shared across
                reconnects, since it represents an ongoing capture that does
                not restart.

        Yields:
            :class:`TranscriptEvent` in arrival order.

        Raises:
            SttFatalError: Vendor rejected the session, or sent a fatal error.
            SttError: Every reconnection attempt failed.
        """
        t0 = time.perf_counter_ns()
        trace = current_trace()
        connect = self._connect or default_connect_factory()
        last_exc: BaseException | None = None
        finished = False

        for attempt in range(max(1, self._retry.attempts)):
            if attempt:
                if trace is not None:
                    trace.mark(
                        f"stt.{self.name}.reconnect",
                        attempt=attempt + 1,
                        error=str(last_exc),
                    )
                delay = self._retry.backoff_ms(attempt - 1) / 1000.0
                if delay:
                    await asyncio.sleep(delay)
            try:
                # aclosing() guarantees the socket is torn down promptly when a
                # consumer abandons the outer generator, instead of waiting for
                # async-generator finalisation at event-loop shutdown.
                async with contextlib.aclosing(
                    self._session(audio_chunks, connect, t0)
                ) as session:
                    async for ev in session:
                        yield ev
                # Returning without an explicit end means the audio simply ran
                # out. That is success, not a reason to reconnect.
                finished = True
            except SttFatalError:
                raise
            except (SttError, OSError, asyncio.TimeoutError) as exc:
                last_exc = exc
                continue
            if finished:
                return

        raise SttError(
            f"{self.name}: stream failed after {self._retry.attempts} attempt(s): "
            f"{last_exc}"
        ) from last_exc

    async def _session(
        self,
        audio_chunks: AsyncIterator[bytes],
        connect: ConnectFactory,
        t0: int,
    ) -> AsyncIterator[TranscriptEvent]:
        """One connection's lifetime: send audio and receive events concurrently.

        The sender runs as a task while this coroutine owns the receive loop.
        That asymmetry is deliberate -- the receive side is the generator body,
        so backpressure from a slow consumer naturally stops us pulling from the
        socket, while audio keeps flowing to the vendor uninterrupted.
        """
        trace = current_trace()
        audio_done = asyncio.Event()
        sender_error: list[BaseException] = []

        async with connect(self._url(), self._headers(), self._open_timeout_s) as ws:
            if trace is not None:
                trace.mark(f"stt.{self.name}.connected")

            async def pump_audio() -> None:
                try:
                    async for chunk in audio_chunks:
                        if chunk:
                            await ws.send(self._encode_frame(chunk))
                    closing = self._close_frame()
                    if closing is not None:
                        await ws.send(closing)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:  # noqa: BLE001 - surfaced in the loop
                    sender_error.append(exc)
                finally:
                    audio_done.set()

            sender = asyncio.create_task(pump_audio(), name=f"{self.name}-audio")
            try:
                while True:
                    if sender_error:
                        raise SttError(
                            f"{self.name}: audio send failed: {sender_error[0]}"
                        ) from sender_error[0]
                    # Only bound the wait once the audio is finished: mid-capture
                    # silence is normal and must not be mistaken for a stall.
                    timeout = self._finalize_timeout_s if audio_done.is_set() else None
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        if trace is not None:
                            trace.mark(f"stt.{self.name}.finalize_timeout")
                        return
                    except (ConnectionError, OSError) as exc:
                        raise SttError(f"{self.name}: socket closed: {exc}") from exc

                    payload = _decode_json(raw)
                    if payload is None:
                        continue

                    t_ms = (time.perf_counter_ns() - t0) / 1e6
                    for ev in self._parse(payload, t_ms):
                        if trace is not None and ev.kind != "partial":
                            trace.mark(f"stt.{self.name}.{ev.kind}", t_ms=round(t_ms, 2))
                        yield ev
                        if ev.is_fatal:
                            raise SttFatalError(
                                f"{self.name}: {ev.raw.get('message', 'fatal error')}",
                                code=str(ev.raw.get("code", "")) or None,
                            )
                        if ev.kind == "session_end":
                            return
            finally:
                # Order matters: cancel the sender, then *await* it. Cancelling
                # without awaiting leaves a task holding a closing socket and
                # emits "Task exception was never retrieved" noise that hides
                # real failures.
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
                with contextlib.suppress(Exception):
                    await ws.close()


def _decode_json(raw: str | bytes | None) -> dict[str, Any] | None:
    """Parse a server frame, tolerating keepalives and binary noise.

    Returns None for anything that is not a JSON object. Vendors send bare
    ``"ping"`` text frames and occasional empty frames; treating those as
    protocol violations would kill a perfectly healthy session.
    """
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    raw = raw.strip()
    if not raw or raw[0] != "{":
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
