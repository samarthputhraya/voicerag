"""Tests for speech-to-text clients and speculative retrieval.

None of these tests touch a network. The vendor APIs are unreachable from CI, so
the *wire format* is pinned here instead: the exact URL, the exact headers and
the exact JSON frames we put on the socket are asserted byte-for-byte. Those
assertions are the specification a human diffs against the vendor docs before
the demo -- if they are wrong, they are at least wrong somewhere visible.

The speculative-driver tests concentrate on the two things that are genuinely
hard to get right: cancellation (no leaked tasks, no double-launch) and the
hit/miss decision boundary (a false hit serves passages retrieved for a
different question, which is a correctness bug wearing a performance costume).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import struct
from typing import Any, Iterable, Sequence

import numpy as np
import pytest

from voicerag.harness.resilience import RetryPolicy, TransientError
from voicerag.harness.trace import Trace, traced
from voicerag.stt.base import (
    AudioTooLongError,
    SttError,
    SttFatalError,
    TranscriptEvent,
    frame_pcm,
    iter_chunks,
)
from voicerag.stt.elevenlabs import ElevenLabsRealtimeStt
from voicerag.stt.sarvam import (
    BATCH_MAX_SECONDS,
    SarvamBatchStt,
    SarvamRealtimeStt,
    audio_duration_s,
)
from voicerag.stt.speculative import (
    SpeculativeTranscriptDriver,
    normalise_query,
    token_count,
)

FAST_RETRY = RetryPolicy(attempts=2, base_ms=0.0, max_ms=0.0, jitter=False)


# --- fakes --------------------------------------------------------------------

class FakeWebSocket:
    """Scripted WebSocket double.

    ``script`` entries are sent to the client in order: a ``dict`` is JSON
    encoded, a ``str`` is delivered verbatim (to exercise keepalives and
    garbage), and an ``Exception`` is raised from ``recv``. Once the script is
    exhausted ``recv`` blocks forever, which is exactly how a real server behaves
    while waiting for more audio and is what the finalize timeout must survive.
    """

    def __init__(
        self,
        script: Sequence[Any] = (),
        *,
        send_error: Exception | None = None,
    ) -> None:
        self.script = list(script)
        self.sent: list[str] = []
        self.closed = False
        self._send_error = send_error

    async def send(self, message: str) -> None:
        if self._send_error is not None:
            raise self._send_error
        self.sent.append(message)

    async def recv(self) -> str:
        if not self.script:
            await asyncio.sleep(3600)  # cancelled by the finalize timeout
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        await asyncio.sleep(0)  # yield so the sender task can interleave
        return item if isinstance(item, str) else json.dumps(item)

    async def close(self) -> None:
        self.closed = True


class FakeConnector:
    """Injectable connect factory that records how it was called."""

    def __init__(self, *sessions: Any) -> None:
        self.sessions = list(sessions)
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> Any:
        self.calls.append((url, dict(headers), timeout))
        item = self.sessions.pop(0) if self.sessions else FakeWebSocket()
        if isinstance(item, Exception):
            raise item
        connector = self

        class _Ctx:
            async def __aenter__(self) -> Any:
                return item

            async def __aexit__(self, *exc: Any) -> bool:
                connector.exited = True
                return False

        return _Ctx()


async def collect(provider: Any, audio: Iterable[bytes]) -> list[TranscriptEvent]:
    return [ev async for ev in provider.stream(iter_chunks(audio))]


def sarvam(**kw: Any) -> SarvamRealtimeStt:
    kw.setdefault("retry", FAST_RETRY)
    kw.setdefault("finalize_timeout_s", 0.05)
    return SarvamRealtimeStt("k-123", **kw)


# --- framing ------------------------------------------------------------------

def test_frame_pcm_splits_on_sample_boundaries():
    frames = frame_pcm(b"\x00\x01" * 1600, frame_ms=100, sample_rate=16_000)
    assert len(frames) == 1
    assert all(len(f) % 2 == 0 for f in frames)


def test_frame_pcm_last_frame_may_be_short():
    frames = frame_pcm(b"\x00" * 5000, frame_ms=100, sample_rate=16_000)
    assert [len(f) for f in frames] == [3200, 1800]


def test_frame_pcm_rejects_non_positive_frame_ms():
    with pytest.raises(ValueError, match="frame_ms"):
        frame_pcm(b"\x00\x00", frame_ms=0)


def test_frame_pcm_of_empty_audio_is_a_single_empty_frame():
    assert frame_pcm(b"") == [b""]


# --- sarvam: wire format ------------------------------------------------------

def test_sarvam_url_pins_every_required_query_parameter():
    url = sarvam(language_code="hi-IN")._url()
    assert url == (
        "wss://api.sarvam.ai/speech-to-text-realtime/ws"
        "?model=saaras%3Av3-realtime"
        "&language_code=hi-IN"
        "&stream_type=fast"
        "&endpointing=vad"
        "&encoding=linear16"
        "&sample_rate=16000"
    )


def test_sarvam_auth_header_is_the_subscription_key():
    assert sarvam()._headers() == {"api-subscription-key": "k-123"}


def test_sarvam_omits_optional_params_that_were_not_set():
    url = sarvam()._url()
    for absent in ("threshold=", "silence_duration_ms=", "prompt=", "&mode="):
        assert absent not in url


def test_sarvam_serialises_optional_params_including_booleans():
    url = sarvam(
        threshold=0.45,
        silence_duration_ms=700,
        min_speech_duration_ms=200,
        return_timestamps=True,
        prompt="MS MARCO",
        mode="dictation",
    )._url()
    assert "threshold=0.45" in url
    assert "silence_duration_ms=700" in url
    assert "min_speech_duration_ms=200" in url
    assert "return_timestamps=true" in url  # not "True"
    assert "prompt=MS+MARCO" in url
    assert "mode=dictation" in url


def test_sarvam_audio_frame_is_exactly_the_documented_shape():
    frame = sarvam()._encode_frame(b"\x00\x01\x02")
    assert frame == '{"event":"audio_input","audio":"AAEC"}'
    assert json.loads(frame) == {"event": "audio_input", "audio": "AAEC"}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"stream_type": "turbo"}, "stream_type"),
        ({"endpointing": "auto"}, "endpointing"),
        ({"encoding": "opus"}, "encoding"),
        ({"sample_rate": 44_100}, "sample_rate"),
    ],
)
def test_sarvam_rejects_undocumented_enum_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sarvam(**kwargs)


def test_sarvam_requires_an_api_key():
    with pytest.raises(ValueError, match="api_key"):
        SarvamRealtimeStt("")


# --- sarvam: server events ----------------------------------------------------

async def test_sarvam_maps_every_documented_server_event():
    ws = FakeWebSocket(
        [
            {"type": "session.begin"},
            {"type": "vad.speech_start"},
            {"type": "transcript.partial", "data": {"transcript": "what is"}},
            {"type": "transcript.final", "data": {"transcript": "what is faiss"}},
            {"type": "vad.speech_end"},
            {"type": "session.end", "data": {"audio_duration_s": 1.5}},
        ]
    )
    events = await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x00"])

    assert [e.kind for e in events] == [
        "speech_start",
        "partial",
        "final",
        "speech_end",
        "session_end",
    ]
    assert events[1].text == "what is" and not events[1].is_final
    assert events[2].text == "what is faiss" and events[2].is_final
    assert events[4].raw["data"]["audio_duration_s"] == 1.5
    assert ws.closed


async def test_sarvam_accepts_event_key_as_well_as_type_key():
    ws = FakeWebSocket([{"event": "transcript.final", "transcript": "hello"}])
    events = await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x00"])
    assert [(e.kind, e.text) for e in events] == [("final", "hello")]


async def test_sarvam_prefers_nested_transcript_over_flat_text():
    ws = FakeWebSocket(
        [
            {
                "type": "transcript.final",
                "data": {"transcript": "nested"},
                "text": "flat",
            }
        ]
    )
    events = await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x00"])
    assert events[0].text == "nested"


async def test_sarvam_ignores_keepalives_and_unknown_events():
    ws = FakeWebSocket(
        ["ping", "", "not json", {"type": "future.event"}, {"type": "session.end"}]
    )
    events = await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x00"])
    assert [e.kind for e in events] == ["session_end"]


async def test_sarvam_non_fatal_error_is_surfaced_but_the_session_continues():
    ws = FakeWebSocket(
        [
            {"type": "error", "code": "AUDIO_DROP", "is_fatal": False, "message": "eh"},
            {"type": "transcript.final", "data": {"transcript": "still here"}},
            {"type": "session.end"},
        ]
    )
    events = await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x00"])
    assert [e.kind for e in events] == ["error", "final", "session_end"]
    assert events[0].is_fatal is False


async def test_sarvam_fatal_error_terminates_the_stream():
    ws = FakeWebSocket(
        [
            {
                "type": "error",
                "data": {"code": "AUTH", "is_fatal": True, "message": "bad key"},
            },
            {"type": "transcript.final", "data": {"transcript": "never seen"}},
        ]
    )
    seen: list[TranscriptEvent] = []
    with pytest.raises(SttFatalError, match="bad key"):
        async for ev in sarvam(connect=FakeConnector(ws)).stream(
            iter_chunks([b"\x00\x00"])
        ):
            seen.append(ev)
    assert [e.kind for e in seen] == ["error"]
    assert seen[0].is_fatal is True


async def test_sarvam_error_without_is_fatal_defaults_to_recoverable():
    ws = FakeWebSocket([{"type": "error", "message": "hmm"}, {"type": "session.end"}])
    events = await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x00"])
    assert [e.kind for e in events] == ["error", "session_end"]


# --- sarvam: connection lifecycle --------------------------------------------

async def test_sarvam_sends_one_frame_per_audio_chunk():
    ws = FakeWebSocket([{"type": "session.end"}])
    await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x01", b"\x02\x03", b""])
    assert len(ws.sent) == 2, "empty chunks must not be sent"
    assert json.loads(ws.sent[1])["audio"] == "AgM="


async def test_sarvam_stops_at_the_finalize_timeout_when_the_server_goes_quiet():
    ws = FakeWebSocket([{"type": "transcript.final", "transcript": "done"}])
    events = await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x00"])
    assert [e.kind for e in events] == ["final"]
    assert ws.closed


async def test_sarvam_reconnects_after_a_dropped_socket():
    dropped = FakeWebSocket([ConnectionResetError("boom")])
    good = FakeWebSocket([{"type": "transcript.final", "transcript": "second try"}])
    connector = FakeConnector(dropped, good)
    events = await collect(sarvam(connect=connector), [b"\x00\x00"])
    assert [e.text for e in events] == ["second try"]
    assert len(connector.calls) == 2


async def test_sarvam_gives_up_after_the_retry_budget():
    connector = FakeConnector(
        FakeWebSocket([ConnectionResetError("1")]),
        FakeWebSocket([ConnectionResetError("2")]),
    )
    with pytest.raises(SttError, match="2 attempt"):
        await collect(sarvam(connect=connector), [b"\x00\x00"])
    assert len(connector.calls) == 2


async def test_sarvam_does_not_reconnect_after_a_fatal_error():
    connector = FakeConnector(
        FakeWebSocket([{"type": "error", "is_fatal": True, "message": "401"}]),
        FakeWebSocket([{"type": "session.end"}]),
    )
    with pytest.raises(SttFatalError):
        await collect(sarvam(connect=connector), [b"\x00\x00"])
    assert len(connector.calls) == 1, "a rejected key fails identically on retry"


async def test_sarvam_surfaces_a_send_failure_as_transient():
    ws = FakeWebSocket([{"type": "session.begin"}], send_error=OSError("pipe"))
    connector = FakeConnector(ws, FakeWebSocket([ConnectionResetError("again")]))
    with pytest.raises(SttError):
        await collect(sarvam(connect=connector), [b"\x00\x00"])


async def test_abandoning_the_stream_closes_the_socket_promptly():
    ws = FakeWebSocket(
        [
            {"type": "transcript.partial", "transcript": "a"},
            {"type": "transcript.partial", "transcript": "a b"},
            {"type": "session.end"},
        ]
    )
    stream = sarvam(connect=FakeConnector(ws)).stream(iter_chunks([b"\x00\x00"]))
    async with contextlib.aclosing(stream):
        async for _ in stream:
            break  # walk away after the first event
    assert ws.closed, "closing the stream must tear the connection down"


async def test_no_tasks_are_leaked_by_a_completed_stream():
    before = len(asyncio.all_tasks())
    ws = FakeWebSocket([{"type": "session.end"}])
    await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x00"] * 5)
    await asyncio.sleep(0)
    assert len(asyncio.all_tasks()) == before


async def test_stream_records_lifecycle_marks_on_the_trace():
    ws = FakeWebSocket(
        [
            {"type": "session.begin"},
            {"type": "transcript.final", "transcript": "hi"},
            {"type": "session.end"},
        ]
    )
    with traced(Trace()) as trace:
        await collect(sarvam(connect=FakeConnector(ws)), [b"\x00\x00"])
    names = {s.name for s in trace.spans}
    assert "stt.sarvam.connected" in names
    assert "stt.sarvam.session_begin" in names
    assert "stt.sarvam.final" in names


async def test_transcribe_joins_final_segments_only():
    ws = FakeWebSocket(
        [
            {"type": "transcript.partial", "transcript": "how tall"},
            {"type": "transcript.final", "transcript": "how tall is"},
            {"type": "transcript.final", "transcript": "everest"},
            {"type": "session.end"},
        ]
    )
    text = await sarvam(connect=FakeConnector(ws)).transcribe(b"\x00\x01" * 100)
    assert text == "how tall is everest"


# --- sarvam: batch ------------------------------------------------------------

def wav(seconds: float, *, rate: int = 16_000) -> bytes:
    """Minimal 16-bit mono RIFF/WAVE file of the requested duration."""
    n = int(seconds * rate) * 2
    return (
        b"RIFF"
        + struct.pack("<I", 36 + n)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", n)
        + b"\x00" * n
    )


def test_audio_duration_reads_the_wav_header():
    assert audio_duration_s(wav(2.0)) == pytest.approx(2.0)
    assert audio_duration_s(wav(1.0, rate=8_000)) == pytest.approx(1.0)


def test_audio_duration_falls_back_to_raw_pcm_geometry():
    assert audio_duration_s(b"\x00" * 32_000) == pytest.approx(1.0)
    assert audio_duration_s(b"") == 0.0


def test_audio_duration_of_a_truncated_wav_header_does_not_under_report():
    truncated = wav(1.0)[:50]
    # Must not silently read as "short"; erring high keeps the 30 s guard safe.
    assert audio_duration_s(truncated) > 0.0


def http_client(handler: Any) -> Any:
    import httpx

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_batch_rejects_audio_at_the_thirty_second_limit():
    stt = SarvamBatchStt("k", client=http_client(lambda r: None))
    with pytest.raises(AudioTooLongError, match="under 30s"):
        await stt.transcribe(wav(BATCH_MAX_SECONDS))


async def test_batch_posts_multipart_with_the_documented_fields():
    import httpx

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("api-subscription-key")
        seen["body"] = request.content
        return httpx.Response(200, json={"transcript": "  hello world  "})

    stt = SarvamBatchStt("k-9", model="saaras:v4", language_code="hi-IN",
                         client=http_client(handler))
    assert await stt.transcribe(wav(1.0)) == "hello world"
    assert seen["url"] == "https://api.sarvam.ai/speech-to-text"
    assert seen["key"] == "k-9"
    assert b'name="file"' in seen["body"]
    assert b'name="model"' in seen["body"] and b"saaras:v4" in seen["body"]
    assert b'name="language_code"' in seen["body"]


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_batch_maps_retryable_statuses_to_transient(status):
    import httpx

    stt = SarvamBatchStt(
        "k", client=http_client(lambda r: httpx.Response(status, text="nope"))
    )
    with pytest.raises(TransientError):
        await stt.transcribe(wav(0.5))


@pytest.mark.parametrize("status", [400, 401, 403])
async def test_batch_maps_client_errors_to_fatal(status):
    import httpx

    stt = SarvamBatchStt(
        "k", client=http_client(lambda r: httpx.Response(status, text="bad"))
    )
    with pytest.raises(SttFatalError):
        await stt.transcribe(wav(0.5))


async def test_batch_provider_refuses_to_pretend_to_stream():
    with pytest.raises(SttFatalError, match="batch endpoint"):
        SarvamBatchStt("k").stream(iter_chunks([b""]))


# --- elevenlabs ---------------------------------------------------------------

def eleven(**kw: Any) -> ElevenLabsRealtimeStt:
    kw.setdefault("retry", FAST_RETRY)
    kw.setdefault("finalize_timeout_s", 0.05)
    return ElevenLabsRealtimeStt("xi-1", **kw)


def test_elevenlabs_url_and_header():
    assert eleven()._url() == (
        "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
        "?model_id=scribe_v2_realtime"
        "&encoding=pcm_s16le_16"
        "&sample_rate=16000"
    )
    assert eleven()._headers() == {"xi-api-key": "xi-1"}


def test_elevenlabs_audio_frame_uses_audio_base_64():
    frame = eleven()._encode_frame(b"\x00\x01\x02")
    assert frame == '{"message_type":"input_audio_chunk","audio_base_64":"AAEC"}'


async def test_elevenlabs_maps_its_three_transcript_message_types():
    ws = FakeWebSocket(
        [
            {"message_type": "partial_transcript", "text": "who wrote"},
            {"message_type": "final_transcript", "text": "who wrote hamlet"},
            {"message_type": "committed_transcript", "text": "who wrote hamlet"},
        ]
    )
    events = await collect(eleven(connect=FakeConnector(ws)), [b"\x00\x00"])
    assert [(e.kind, e.is_final) for e in events] == [
        ("partial", False),
        ("final", True),
        ("final", True),
    ]


async def test_elevenlabs_treats_errors_as_fatal_by_default():
    ws = FakeWebSocket([{"message_type": "error", "message": "quota"}])
    with pytest.raises(SttFatalError, match="quota"):
        await collect(eleven(connect=FakeConnector(ws)), [b"\x00\x00"])


def test_elevenlabs_adds_optional_language_and_extra_params():
    url = eleven(language_code="en", extra_params={"diarize": "false"})._url()
    assert "language_code=en" in url and "diarize=false" in url


async def test_both_providers_satisfy_the_same_interface():
    """The property that makes them interchangeable in first_healthy()."""
    script = [{"type": "transcript.final", "transcript": "x"}, {"type": "session.end"}]
    a = await collect(sarvam(connect=FakeConnector(FakeWebSocket(script))), [b"\x00\x00"])
    ws = FakeWebSocket([{"message_type": "final_transcript", "text": "x"}])
    b = await collect(eleven(connect=FakeConnector(ws)), [b"\x00\x00"])
    assert a[0].kind == b[0].kind == "final"
    assert a[0].text == b[0].text == "x"


# --- speculative retrieval ----------------------------------------------------

class AngleEmbedder:
    """Embedder returning two unit vectors with a prescribed cosine.

    Lets a test place the final transcript at an exact angular distance from the
    speculated one, which is the only way to probe the decision boundary
    precisely rather than approximately.
    """

    def __init__(self, cosine: float) -> None:
        self.cosine = cosine
        self.calls = 0

    @property
    def dim(self) -> int:
        return 2

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        c = float(self.cosine)
        s = float(np.sqrt(max(0.0, 1.0 - c * c)))
        rows = [[1.0, 0.0], [c, s]]
        return np.array(rows[: len(texts)], dtype=np.float64)


class Retriever:
    """Async retrieval double recording calls, delays and cancellations."""

    def __init__(self, delay_s: float = 0.0, *, fail: bool = False) -> None:
        self.delay_s = delay_s
        self.fail = fail
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self.completed: list[str] = []

    async def __call__(self, text: str) -> str:
        self.calls.append(text)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled.append(text)
            raise
        if self.fail:
            raise RuntimeError("index unavailable")
        self.completed.append(text)
        return f"hits::{text}"


def partial(text: str) -> TranscriptEvent:
    return TranscriptEvent(kind="partial", text=text)


def final(text: str) -> TranscriptEvent:
    return TranscriptEvent(kind="final", text=text, is_final=True)


def test_normalise_query_is_stable_under_case_spacing_and_punctuation():
    assert normalise_query("  What  is FAISS? ") == "what is faiss"
    assert normalise_query("भारत की राजधानी क्या है।") == "भारत की राजधानी क्या है"
    assert token_count("what is  faiss?") == 3


async def test_partial_below_the_token_gate_does_not_speculate():
    retrieve = Retriever()
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=3)
    await driver.on_event(partial("what is"))
    assert retrieve.calls == []
    assert driver.stats.launched == 0
    await driver.aclose()


async def test_partial_at_the_token_gate_launches_one_speculation():
    retrieve = Retriever(delay_s=0.05)
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=3)
    await driver.on_event(partial("what is faiss"))
    assert retrieve.calls == ["what is faiss"]
    assert driver.stats.launched == 1
    assert driver.in_flight
    await driver.aclose()


async def test_a_newer_partial_cancels_the_one_in_flight():
    retrieve = Retriever(delay_s=5.0)
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=3)
    await driver.on_event(partial("what is the"))
    await driver.on_event(partial("what is the capital of france"))

    assert retrieve.cancelled == ["what is the"], "the stale task must be cancelled"
    assert driver.stats.launched == 2
    assert driver.stats.cancelled == 1
    assert driver.stats.wasted_ms >= 0.0
    await driver.aclose()
    assert not driver.in_flight


async def test_at_most_one_speculation_is_ever_in_flight():
    retrieve = Retriever(delay_s=5.0)
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=1)
    for i in range(1, 7):
        await driver.on_event(partial(" ".join(f"w{n}" for n in range(i))))
        live = [t for t in asyncio.all_tasks() if t.get_name().startswith("speculate-")]
        assert len(live) <= 1
    await driver.aclose()


async def test_identical_final_reuses_the_speculation_and_saves_time():
    retrieve = Retriever(delay_s=0.03)
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=3)
    await driver.on_event(partial("what is the capital of france"))
    await asyncio.sleep(0.05)  # let the speculation finish while "speaking"
    outcome = await driver.on_event(final("What is the capital of France?"))

    assert outcome is not None and outcome.hit
    assert outcome.result == "hits::what is the capital of france"
    assert outcome.similarity == 1.0
    assert outcome.saved_ms > 0.0
    assert outcome.wait_ms < outcome.saved_ms, "a completed speculation costs no wait"
    assert retrieve.calls == ["what is the capital of france"], "no second retrieval"
    assert driver.stats.hits == 1 and driver.stats.misses == 0
    await driver.aclose()


async def test_final_arriving_mid_speculation_waits_rather_than_restarting():
    retrieve = Retriever(delay_s=0.08)
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=3)
    await driver.on_event(partial("who wrote hamlet"))
    await asyncio.sleep(0.02)
    outcome = await driver.on_event(final("who wrote hamlet"))

    assert outcome is not None and outcome.hit
    assert len(retrieve.calls) == 1
    assert outcome.wait_ms > 0.0, "the remainder of the speculation is still paid"
    assert 0.0 < outcome.saved_ms <= outcome.saved_ms + outcome.wait_ms
    await driver.aclose()


async def test_dissimilar_final_discards_the_speculation_and_retrieves_fresh():
    retrieve = Retriever(delay_s=5.0)
    driver = SpeculativeTranscriptDriver(
        retrieve, embedder=AngleEmbedder(0.4), min_token_delta=3
    )
    await driver.on_event(partial("what is the capital of france"))
    retrieve.delay_s = 0.0  # the fresh retrieval must not inherit the long delay
    outcome = await driver.on_event(final("who is the president of brazil"))

    assert outcome is not None and outcome.hit is False
    assert outcome.result == "hits::who is the president of brazil"
    assert outcome.saved_ms == 0.0
    assert retrieve.cancelled == ["what is the capital of france"]
    assert driver.stats.misses == 1 and driver.stats.hits == 0
    await driver.aclose()


async def test_similarity_gate_is_inclusive_at_the_threshold():
    """The `>=` boundary, probed exactly rather than approximately.

    The threshold is set to the cosine the embedder actually produces, so the
    two cases differ only by whether the comparison includes its endpoint.
    """
    embedder = AngleEmbedder(0.985)
    probe = SpeculativeTranscriptDriver(Retriever(), embedder=embedder)
    exact = probe._cosine("a", "b")  # the value the driver itself computes
    await probe.aclose()

    for threshold, expect_hit in ((exact, True), (exact + 1e-9, False)):
        retrieve = Retriever()
        driver = SpeculativeTranscriptDriver(
            retrieve,
            embedder=AngleEmbedder(0.985),
            min_token_delta=3,
            similarity_threshold=threshold,
        )
        await driver.on_event(partial("what is the capital of france"))
        await asyncio.sleep(0.01)
        outcome = await driver.on_event(final("what is the capital of germany"))
        assert outcome is not None
        assert outcome.hit is expect_hit, f"threshold={threshold!r}"
        await driver.aclose()


async def test_without_an_embedder_only_exact_matches_hit():
    retrieve = Retriever()
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=3)
    await driver.on_event(partial("what is the capital of france"))
    await asyncio.sleep(0.01)
    hit = await driver.on_event(final("what is the capital of france"))
    assert hit is not None and hit.hit

    await driver.on_event(partial("what is the capital of spain"))
    await asyncio.sleep(0.01)
    miss = await driver.on_event(final("what is the capital of spain today"))
    assert miss is not None and miss.hit is False
    assert miss.similarity == 0.0
    await driver.aclose()


async def test_a_failed_speculation_degrades_to_a_normal_retrieval():
    retrieve = Retriever(fail=True)
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=3)
    await driver.on_event(partial("what is the capital of france"))
    await asyncio.sleep(0.01)
    retrieve.fail = False
    outcome = await driver.on_event(final("what is the capital of france"))

    assert outcome is not None and outcome.hit is False
    assert outcome.result == "hits::what is the capital of france"
    assert driver.stats.misses == 1
    await driver.aclose()


async def test_an_embedder_that_raises_is_treated_as_a_miss():
    class Broken:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            raise RuntimeError("model not loaded")

    retrieve = Retriever()
    driver = SpeculativeTranscriptDriver(
        retrieve, embedder=Broken(), min_token_delta=3
    )
    await driver.on_event(partial("what is the capital of france"))
    outcome = await driver.on_event(final("what is the capital of germany"))
    assert outcome is not None and outcome.hit is False
    await driver.aclose()


async def test_run_resolves_at_the_first_final_and_cleans_up():
    retrieve = Retriever(delay_s=0.02)
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=2)

    async def events():
        yield TranscriptEvent(kind="speech_start")
        yield partial("how tall is")
        yield partial("how tall is mount everest")
        await asyncio.sleep(0.04)
        yield final("how tall is mount everest")
        yield final("how tall is mount everest")  # committed duplicate

    before = len(asyncio.all_tasks())
    outcome = await driver.run(events())
    await asyncio.sleep(0)

    assert outcome is not None and outcome.hit
    assert outcome.text == "how tall is mount everest"
    assert len(asyncio.all_tasks()) == before, "run() must not leak tasks"


async def test_run_returns_none_when_the_stream_ends_without_a_final():
    driver = SpeculativeTranscriptDriver(Retriever(delay_s=5.0), min_token_delta=2)

    async def events():
        yield partial("what is faiss")
        yield TranscriptEvent(kind="session_end")

    assert await driver.run(events()) is None
    assert not driver.in_flight, "aclose() must cancel the in-flight speculation"


async def test_a_second_question_in_the_same_session_still_speculates():
    retrieve = Retriever()
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=3)
    await driver.on_event(partial("what is the capital of france"))
    await asyncio.sleep(0.01)
    await driver.on_event(final("what is the capital of france"))

    await driver.on_event(partial("who wrote hamlet"))
    assert driver.stats.launched == 2, "the token high-water mark must reset"
    await driver.aclose()


async def test_stats_report_hit_rate_and_serialise():
    retrieve = Retriever()
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=2)
    await driver.on_event(partial("what is faiss"))
    await asyncio.sleep(0.01)
    await driver.on_event(final("what is faiss"))
    await driver.on_event(final("something entirely different here"))

    stats = driver.stats.to_dict()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert driver.stats.resolutions == 2
    await driver.aclose()


async def test_speculation_is_instrumented_on_the_trace():
    retrieve = Retriever()
    driver = SpeculativeTranscriptDriver(retrieve, min_token_delta=2)
    with traced(Trace()) as trace:
        await driver.on_event(partial("what is faiss"))
        await asyncio.sleep(0.01)
        await driver.on_event(final("what is faiss"))
    names = {s.name for s in trace.spans}
    assert {"speculate.launch", "speculate.hit"} <= names
    await driver.aclose()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_token_delta": 0}, "min_token_delta"),
        ({"similarity_threshold": 1.5}, "similarity_threshold"),
    ],
)
def test_driver_rejects_nonsensical_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SpeculativeTranscriptDriver(Retriever(), **kwargs)


async def test_non_transcript_events_are_ignored():
    driver = SpeculativeTranscriptDriver(Retriever())
    for kind in ("speech_start", "speech_end", "error", "session_end"):
        assert await driver.on_event(TranscriptEvent(kind=kind)) is None
    await driver.aclose()


async def test_completed_speculations_are_never_double_counted():
    """`completed` must not exceed `launched`; an impossible ratio in the demo
    HUD would discredit every other number next to it."""
    retrieve = Retriever()
    driver = SpeculativeTranscriptDriver(
        retrieve, embedder=AngleEmbedder(0.99), min_token_delta=2
    )
    await driver.on_event(partial("how tall"))
    await asyncio.sleep(0.01)
    await driver.on_event(partial("how tall is mount"))
    await asyncio.sleep(0.01)
    outcome = await driver.on_event(final("how tall is mount everest"))

    assert outcome is not None and outcome.hit, "a near-miss partial still hits"
    assert driver.stats.launched == 2
    assert driver.stats.completed == 2
    assert driver.stats.completed <= driver.stats.launched
    await driver.aclose()


# --- relay: an idle session is not a failure -----------------------------------


def test_idle_upstream_close_is_not_reported_as_a_failure() -> None:
    """Sarvam ends an idle realtime session with 1008 "Inactivity timeout".

    The browser VAD deliberately keeps the microphone open after a question, so
    this fires on any session where the user asks once and then listens to the
    answer. Treating it as a fatal relay error put a red "Speech recognition
    failed: ConnectionClosedError ..." banner on screen -- observed in testing
    sitting directly above a correct, fully grounded answer.
    """
    from voicerag.api.stt_relay import _is_benign_close

    closed_error = type("ConnectionClosedError", (Exception,), {})

    idle = closed_error(
        "received 1008 (policy violation) Inactivity timeout; "
        "then sent 1008 (policy violation) Inactivity timeout"
    )
    assert _is_benign_close(idle)

    assert _is_benign_close(type("ConnectionClosedOK", (Exception,), {})("bye"))

    # A genuine upstream fault must still be reported.
    assert not _is_benign_close(closed_error("received 1011 internal error"))
    assert not _is_benign_close(RuntimeError("upstream refused the subscription key"))
