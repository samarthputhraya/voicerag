"""Tests for the grounded prompt, streaming generators and provider fallback.

Two things are pinned here that nothing else can pin, because neither Groq nor
Gemini is reachable from this environment:

1. **The exact request bodies.** ``build_payload`` is pure, so the whole JSON
   body is asserted as a dict literal. If a field name is wrong, it is wrong in
   a place a human can diff against the vendor docs in ten seconds.
2. **The latency measurement itself.** The 200 ms claim rests on TTFT and total
   completion being measured correctly, so the fake transports deliberately
   delay the first token and then stream the rest quickly -- the two numbers
   must come out clearly different, and in the right order.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, MutableMapping, Sequence

import httpx
import pytest

from voicerag.harness.resilience import (
    AllProvidersFailed,
    CircuitBreaker,
    Deadline,
    PermanentError,
    RetryPolicy,
    TransientError,
)
from voicerag.harness.trace import Trace, traced
from voicerag.generate.base import DEFAULT_MAX_TOKENS, GenerationResult, Generator
from voicerag.generate.gemini import GeminiGenerator
from voicerag.generate.groq import GroqGenerator
from voicerag.generate.prompt import (
    ABSTAIN_TOKEN,
    SYSTEM_PROMPT,
    format_passages,
    parse_answer,
    render,
    usable_passages,
)
from voicerag.generate.router import GenerationRouter


# --- doubles ------------------------------------------------------------------

class FakeGenerator(Generator):
    """Generator double with scriptable deltas, delays and failures."""

    def __init__(
        self,
        name: str = "fake",
        *,
        deltas: Sequence[str] = ("hello", " world"),
        first_delay_s: float = 0.0,
        delay_s: float = 0.0,
        error: Exception | None = None,
        error_after: int = 0,
        finish_reason: str = "stop",
    ) -> None:
        super().__init__(model=f"{name}-model")
        self.name = name
        self.deltas = list(deltas)
        self.first_delay_s = first_delay_s
        self.delay_s = delay_s
        self.error = error
        self.error_after = error_after
        self.finish_reason = finish_reason
        self.calls = 0
        self.closed = 0

    async def _stream_deltas(
        self,
        system: str,
        user: str,
        *,
        meta: MutableMapping[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.calls += 1
        meta["model"] = self.model
        for i, delta in enumerate(self.deltas):
            if self.error is not None and i == self.error_after:
                raise self.error
            await asyncio.sleep(self.first_delay_s if i == 0 else self.delay_s)
            yield delta
        if self.error is not None and self.error_after >= len(self.deltas):
            raise self.error
        meta["finish_reason"] = self.finish_reason
        meta["usage"] = {"completion_tokens": len(self.deltas)}

    async def aclose(self) -> None:
        self.closed += 1


def sse(*chunks: dict[str, Any] | str) -> list[bytes]:
    """Encode objects as SSE ``data:`` frames, terminated by ``[DONE]``."""
    out = []
    for chunk in chunks:
        body = chunk if isinstance(chunk, str) else json.dumps(chunk)
        out.append(f"data: {body}\n\n".encode())
    out.append(b"data: [DONE]\n\n")
    return out


class DelayedStream(httpx.AsyncByteStream):
    """SSE byte stream that makes the client wait before the first frame.

    This is what gives the TTFT assertions teeth: a stream that arrives all at
    once cannot distinguish a correct TTFT measurement from a broken one.
    """

    def __init__(
        self, chunks: Sequence[bytes], *, first_delay_s: float = 0.0,
        delay_s: float = 0.0,
    ) -> None:
        self._chunks = list(chunks)
        self._first_delay_s = first_delay_s
        self._delay_s = delay_s

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i, chunk in enumerate(self._chunks):
            await asyncio.sleep(self._first_delay_s if i == 0 else self._delay_s)
            yield chunk


def sse_client(
    chunks: Sequence[bytes],
    *,
    status: int = 200,
    first_delay_s: float = 0.0,
    delay_s: float = 0.0,
    record: dict[str, Any] | None = None,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record["url"] = str(request.url)
            record["headers"] = dict(request.headers)
            record["body"] = json.loads(request.content)
        if status >= 400:
            return httpx.Response(status, text="upstream said no")
        return httpx.Response(
            status,
            stream=DelayedStream(chunks, first_delay_s=first_delay_s, delay_s=delay_s),
            headers={"content-type": "text/event-stream"},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def groq_chunk(content: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": "c1",
        "model": "openai/gpt-oss-20b",
        "choices": [{"index": 0, "delta": {"content": content}, **extra}],
    }


# --- prompt -------------------------------------------------------------------

def test_system_prompt_is_byte_identical_across_requests():
    """The whole prompt-cache argument depends on this being literally true."""
    a, _ = render("what is faiss", ["a vector index"])
    b, _ = render("who wrote hamlet", ["shakespeare wrote hamlet", "more text"])
    assert a == b == SYSTEM_PROMPT
    assert a is SYSTEM_PROMPT, "returned by identity, never rebuilt"
    assert "{" not in SYSTEM_PROMPT and "}" not in SYSTEM_PROMPT


def test_system_prompt_states_every_rule_the_guardrails_rely_on():
    assert ABSTAIN_TOKEN in SYSTEM_PROMPT
    assert "[n]" in SYSTEM_PROMPT
    assert "ONE short sentence" in SYSTEM_PROMPT


def test_user_turn_numbers_passages_from_one_and_ends_with_the_question():
    _, user = render("what is faiss?", ["vector search library", "unrelated text"])
    assert user == (
        "PASSAGES:\n"
        "[1] vector search library\n"
        "[2] unrelated text\n"
        "\n"
        "QUESTION: what is faiss?\n"
        "ANSWER:"
    )


def test_render_collapses_whitespace_in_the_question():
    _, user = render("  what   is\nfaiss ", ["x"])
    assert "QUESTION: what is faiss" in user


def test_no_passages_yields_a_context_block_that_forces_abstention():
    _, user = render("what is faiss", [])
    assert "PASSAGES:\n(none)" in user


def test_passages_are_capped_in_count_and_length():
    body = format_passages([f"passage {i}" for i in range(20)], max_chunks=3)
    assert body.count("\n") == 2
    long = format_passages(["word " * 500], max_chars=50)
    assert len(long) < 80 and long.endswith("...")


def test_chunk_like_objects_prefer_their_contextual_text():
    class Chunk:
        text = "raw body"
        context_text = "title: raw body"

    assert format_passages([Chunk()]) == "[1] title: raw body"


def test_blank_passages_are_dropped_and_numbering_stays_contiguous():
    assert format_passages(["", "   ", "real"]) == "[1] real"


def test_usable_passages_maps_a_citation_back_to_its_source_index():
    included = usable_passages(["", "first", "", "second"])
    assert included == [(1, "first"), (3, "second")]
    # Citation [2] is the prompt's second line, i.e. the input's index 3.
    assert included[2 - 1][0] == 3


def test_parse_answer_extracts_ordered_unique_citations():
    parsed = parse_answer("Paris [2] is the capital [1][2].")
    assert parsed.citations == (2, 1)
    assert parsed.abstained is False
    assert parsed.is_grounded_shape


@pytest.mark.parametrize(
    "raw", [ABSTAIN_TOKEN, f" {ABSTAIN_TOKEN}.", f"ANSWER: {ABSTAIN_TOKEN}"]
)
def test_abstention_is_detected_however_the_model_dresses_it_up(raw):
    parsed = parse_answer(raw)
    assert parsed.abstained is True
    assert parsed.text == ""
    assert parsed.is_grounded_shape


def test_uncited_answer_is_flagged_as_ungrounded_in_shape():
    parsed = parse_answer("Paris is the capital of France.")
    assert parsed.citations == ()
    assert parsed.is_grounded_shape is False


def test_citations_outside_the_supplied_range_are_detectable():
    parsed = parse_answer("Everest is 8848 m [1][7].")
    assert parsed.invalid_citations(n_passages=3) == (7,)
    assert parsed.invalid_citations(n_passages=7) == ()


def test_parse_answer_strips_a_prompt_echo():
    assert parse_answer("ANSWER: Paris [1]").text == "Paris [1]"


def test_parse_answer_tolerates_empty_output():
    parsed = parse_answer("")
    assert parsed.text == "" and parsed.citations == () and not parsed.abstained


# --- base: timing and tracing -------------------------------------------------

async def test_ttft_and_total_are_measured_separately():
    gen = FakeGenerator(deltas=["a", "b", "c"], first_delay_s=0.05, delay_s=0.005)
    result = await gen.complete("sys", "usr")

    assert result.text == "abc"
    assert result.ttft_ms == pytest.approx(50, abs=25)
    assert result.total_ms >= result.ttft_ms + 8
    assert result.decode_ms == pytest.approx(result.total_ms - result.ttft_ms)
    assert result.n_deltas == 3
    assert result.provider == "fake" and result.model == "fake-model"
    assert result.usage == {"completion_tokens": 3}


async def test_both_spans_are_emitted_and_ttft_closes_first():
    gen = FakeGenerator(deltas=["a", "b"], first_delay_s=0.04, delay_s=0.02)
    with traced(Trace()) as trace:
        await gen.complete("sys", "usr")

    spans = {s.name: s for s in trace.spans}
    assert {"generate.ttft", "generate.total"} <= set(spans)
    assert spans["generate.ttft"].ms < spans["generate.total"].ms
    assert spans["generate.total"].attrs["n_deltas"] == 2
    assert spans["generate.ttft"].parent == spans["generate.total"].span_id


async def test_empty_deltas_do_not_count_as_the_first_token():
    """A role-only opening chunk must not be reported as time-to-first-token."""
    gen = FakeGenerator(deltas=["", "", "real"], first_delay_s=0.0, delay_s=0.02)
    result = await gen.complete("sys", "usr")
    assert result.text == "real"
    assert result.n_deltas == 1
    assert result.ttft_ms >= 35


async def test_a_failed_generation_still_closes_both_spans():
    gen = FakeGenerator(deltas=["a"], error=TransientError("boom"), error_after=0)
    with traced(Trace()) as trace:
        with pytest.raises(TransientError):
            await gen.complete("sys", "usr")
    for name in ("generate.ttft", "generate.total"):
        span = next(s for s in trace.spans if s.name == name)
        assert span.end_ns is not None
        assert span.error is not None


def test_truncation_is_reported_from_the_finish_reason():
    assert GenerationResult("x", 1, 2, 1, "p", "m", finish_reason="length").truncated
    assert not GenerationResult("x", 1, 2, 1, "p", "m", finish_reason="stop").truncated


def test_generator_rejects_a_nonsensical_token_ceiling():
    with pytest.raises(ValueError, match="max_tokens"):
        GroqGenerator("k", max_tokens=0)


# --- groq ---------------------------------------------------------------------

def test_groq_payload_is_exactly_the_documented_body():
    gen = GroqGenerator("k", client=object())
    payload = gen.build_payload("SYS", "USR")
    assert payload == {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USR"},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "low",
    }
    assert list(payload["messages"])[0]["role"] == "system", "cacheable prefix first"


def test_groq_omits_reasoning_effort_for_models_that_reject_it():
    gen = GroqGenerator("k", model="llama-3.1-8b-instant", client=object())
    assert gen.supports_reasoning_effort is False
    assert "reasoning_effort" not in gen.build_payload("s", "u")


def test_groq_can_send_the_alternate_token_ceiling_field_name():
    gen = GroqGenerator("k", max_tokens_field="max_completion_tokens", client=object())
    payload = gen.build_payload("s", "u")
    assert payload["max_completion_tokens"] == DEFAULT_MAX_TOKENS
    assert "max_tokens" not in payload


def test_groq_rejects_an_unknown_token_ceiling_field_name():
    with pytest.raises(ValueError, match="max_tokens_field"):
        GroqGenerator("k", max_tokens_field="maxTokens")


def test_groq_per_call_override_beats_the_instance_default():
    gen = GroqGenerator("k", client=object())
    assert gen.build_payload("s", "u", max_tokens=12)["max_tokens"] == 12


def test_groq_requires_an_api_key():
    with pytest.raises(ValueError, match="api_key"):
        GroqGenerator("")


async def test_groq_streams_deltas_and_captures_usage_and_finish_reason():
    record: dict[str, Any] = {}
    chunks = sse(
        {"choices": [{"delta": {"role": "assistant"}}]},
        groq_chunk("Paris"),
        groq_chunk(" [1]."),
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 4, "total_tokens": 210}},
    )
    gen = GroqGenerator("k-1", client=sse_client(chunks, record=record))
    result = await gen.complete("SYS", "USR")

    assert result.text == "Paris [1]."
    assert result.n_deltas == 2, "structural chunks are not tokens"
    assert result.finish_reason == "stop"
    assert result.usage["total_tokens"] == 210
    assert result.provider == "groq"
    assert record["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert record["body"]["stream"] is True


async def test_groq_reads_usage_from_the_x_groq_envelope():
    chunks = sse(groq_chunk("hi"), {"x_groq": {"usage": {"completion_tokens": 1}}})
    gen = GroqGenerator("k", client=sse_client(chunks))
    result = await gen.complete("s", "u")
    assert result.usage == {"completion_tokens": 1}


async def test_groq_reports_a_truncated_answer():
    chunks = sse(groq_chunk("a very long answer that ran"),
                 {"choices": [{"delta": {}, "finish_reason": "length"}]})
    gen = GroqGenerator("k", max_tokens=8, client=sse_client(chunks))
    result = await gen.complete("s", "u")
    assert result.truncated is True


async def test_groq_ttft_reflects_the_wait_for_the_first_token():
    chunks = sse(groq_chunk("A"), groq_chunk("B"), groq_chunk("C"))
    gen = GroqGenerator(
        "k", client=sse_client(chunks, first_delay_s=0.06, delay_s=0.002)
    )
    result = await gen.complete("s", "u")
    assert result.ttft_ms >= 50
    assert result.total_ms > result.ttft_ms
    assert result.decode_ms < result.ttft_ms, "decode is fast once the stream opens"


async def test_groq_survives_keepalives_and_malformed_frames():
    chunks = [
        b": keepalive\n\n",
        b"\n",
        b"data: {not json}\n\n",
        b"data: " + json.dumps(groq_chunk("ok")).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    ]
    gen = GroqGenerator("k", client=sse_client(chunks))
    assert (await gen.complete("s", "u")).text == "ok"


async def test_groq_handles_a_coalesced_non_streaming_response():
    chunks = sse({"choices": [{"message": {"content": "whole answer"}}]})
    gen = GroqGenerator("k", client=sse_client(chunks))
    assert (await gen.complete("s", "u")).text == "whole answer"


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_groq_retryable_statuses_raise_transient(status):
    gen = GroqGenerator("k", client=sse_client([], status=status))
    with pytest.raises(TransientError):
        await gen.complete("s", "u")


@pytest.mark.parametrize("status", [400, 401, 404])
async def test_groq_client_errors_raise_permanent(status):
    gen = GroqGenerator("k", client=sse_client([], status=status))
    with pytest.raises(PermanentError):
        await gen.complete("s", "u")


async def test_groq_transport_failure_is_transient():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    gen = GroqGenerator("k", client=httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    with pytest.raises(TransientError, match="groq stream failed"):
        await gen.complete("s", "u")


async def test_groq_builds_a_pooled_client_once_and_closes_it():
    gen = GroqGenerator("k", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"data": []})
    ))
    first = gen._ensure_client()
    assert gen._ensure_client() is first, "the warm pool must be reused"
    assert await gen.warm() is True
    await gen.aclose()
    assert first.is_closed


async def test_warm_never_raises_when_the_endpoint_is_unreachable():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    gen = GroqGenerator("k", transport=httpx.MockTransport(boom))
    assert await gen.warm() is False
    await gen.aclose()


# --- gemini -------------------------------------------------------------------

def test_gemini_url_requests_server_sent_events():
    url = GeminiGenerator("k", client=object()).url()
    assert url == (
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/gemini-2.5-flash-lite:streamGenerateContent?alt=sse"
    )


def test_gemini_payload_disables_thinking_and_caps_output():
    payload = GeminiGenerator("k", client=object()).build_payload("SYS", "USR")
    assert payload == {
        "systemInstruction": {"parts": [{"text": "SYS"}]},
        "contents": [{"role": "user", "parts": [{"text": "USR"}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": DEFAULT_MAX_TOKENS,
            "candidateCount": 1,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }


def test_gemini_can_leave_the_thinking_budget_to_the_model():
    gen = GeminiGenerator("k", thinking_budget=None, client=object())
    assert "thinkingConfig" not in gen.build_payload("s", "u")["generationConfig"]


def test_gemini_sends_the_key_in_a_header_not_the_query_string():
    gen = GeminiGenerator("k-secret", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={})
    ))
    assert "k-secret" not in gen.url()
    assert gen._ensure_client().headers["x-goog-api-key"] == "k-secret"


async def test_gemini_streams_parts_and_normalises_the_finish_reason():
    record: dict[str, Any] = {}
    chunks = sse(
        {
            "candidates": [{"content": {"parts": [{"text": "8848"}, {"text": " m"}]}}],
            "modelVersion": "gemini-2.5-flash-lite",
        },
        {
            "candidates": [
                {"content": {"parts": [{"text": " [1]."}]}, "finishReason": "MAX_TOKENS"}
            ],
            "usageMetadata": {"candidatesTokenCount": 5},
        },
    )
    gen = GeminiGenerator("k", client=sse_client(chunks, record=record))
    result = await gen.complete("SYS", "USR")

    assert result.text == "8848 m [1]."
    assert result.finish_reason == "length" and result.truncated
    assert result.usage == {"candidatesTokenCount": 5}
    assert result.model == "gemini-2.5-flash-lite"
    assert record["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


async def test_gemini_never_leaks_thought_parts_into_the_answer():
    chunks = sse(
        {"candidates": [{"content": {"parts": [
            {"text": "let me think", "thought": True},
            {"text": "Paris [1]."},
        ]}}]}
    )
    gen = GeminiGenerator("k", client=sse_client(chunks))
    assert (await gen.complete("s", "u")).text == "Paris [1]."


@pytest.mark.parametrize(
    "status, expected", [(429, TransientError), (500, TransientError),
                         (400, PermanentError), (403, PermanentError)]
)
async def test_gemini_maps_http_status_to_the_harness_taxonomy(status, expected):
    gen = GeminiGenerator("k", client=sse_client([], status=status))
    with pytest.raises(expected):
        await gen.complete("s", "u")


# --- router -------------------------------------------------------------------

def test_router_requires_at_least_one_provider():
    with pytest.raises(ValueError, match="at least one provider"):
        GenerationRouter([])


async def test_complete_falls_back_to_the_next_provider():
    primary = FakeGenerator("groq", error=TransientError("down"), error_after=0)
    backup = FakeGenerator("gemini", deltas=["Paris", " [1]."])
    router = GenerationRouter([primary, backup], policy=RetryPolicy(
        attempts=1, base_ms=0.0, max_ms=0.0, jitter=False
    ))

    result = await router.complete("SYS", "USR")
    assert result.text == "Paris [1]."
    assert result.provider == "gemini"
    assert primary.calls == 1 and backup.calls == 1


async def test_complete_raises_when_every_provider_fails():
    router = GenerationRouter(
        [
            FakeGenerator("groq", error=TransientError("a"), error_after=0),
            FakeGenerator("gemini", error=TransientError("b"), error_after=0),
        ],
        policy=RetryPolicy(attempts=1, base_ms=0.0, max_ms=0.0, jitter=False),
    )
    with pytest.raises(AllProvidersFailed) as excinfo:
        await router.complete("s", "u")
    assert set(excinfo.value.errors) == {"groq", "gemini"}


async def test_an_open_circuit_skips_the_dead_provider_entirely():
    primary = FakeGenerator("groq", error=TransientError("down"), error_after=0)
    backup = FakeGenerator("gemini")
    breakers = {"groq": CircuitBreaker(name="groq", threshold=1, reset_ms=60_000)}
    router = GenerationRouter(
        [primary, backup],
        breakers=breakers,
        policy=RetryPolicy(attempts=1, base_ms=0.0, max_ms=0.0, jitter=False),
    )

    await router.complete("s", "u")
    await router.complete("s", "u")
    assert primary.calls == 1, "the second request must not pay the dead provider"
    assert breakers["groq"].state == "open"
    assert {h["provider"] for h in router.health()} == {"groq", "gemini"}


async def test_complete_respects_an_exhausted_deadline():
    router = GenerationRouter([FakeGenerator("groq")])
    with pytest.raises(AllProvidersFailed):
        await router.complete("s", "u", deadline=Deadline(0.0))


async def test_stream_fails_over_before_the_first_token():
    primary = FakeGenerator("groq", error=TransientError("down"), error_after=0)
    backup = FakeGenerator("gemini", deltas=["Pa", "ris"])
    router = GenerationRouter([primary, backup])

    meta: dict[str, Any] = {}
    out = [d async for d in router.stream("s", "u", meta=meta)]
    assert "".join(out) == "Paris"
    assert meta["provider"] == "gemini"


async def test_stream_does_not_fail_over_after_text_has_been_emitted():
    """A second provider would continue someone else's sentence."""
    primary = FakeGenerator(
        "groq", deltas=["Pa", "ris"], error=TransientError("mid"), error_after=1
    )
    backup = FakeGenerator("gemini", deltas=["London"])
    router = GenerationRouter([primary, backup])

    seen: list[str] = []
    with pytest.raises(TransientError, match="mid"):
        async for delta in router.stream("s", "u"):
            seen.append(delta)
    assert seen == ["Pa"]
    assert backup.calls == 0


async def test_stream_treats_an_empty_answer_as_a_provider_failure():
    empty = FakeGenerator("groq", deltas=[])
    backup = FakeGenerator("gemini", deltas=["real answer"])
    router = GenerationRouter([empty, backup])
    out = [d async for d in router.stream("s", "u")]
    assert "".join(out) == "real answer"


async def test_stream_raises_when_no_provider_produces_a_token():
    router = GenerationRouter([FakeGenerator("groq", deltas=[])])
    with pytest.raises(AllProvidersFailed):
        [d async for d in router.stream("s", "u")]


async def test_router_marks_the_selected_provider_on_the_trace():
    router = GenerationRouter(
        [
            FakeGenerator("groq", error=TransientError("down"), error_after=0),
            FakeGenerator("gemini"),
        ]
    )
    with traced(Trace()) as trace:
        [d async for d in router.stream("s", "u")]
    marks = {(s.name, s.attrs.get("provider")) for s in trace.spans}
    assert ("generate.failed", "groq") in marks
    assert ("generate.selected", "gemini") in marks


async def test_router_closes_every_provider():
    providers = [FakeGenerator("groq"), FakeGenerator("gemini")]
    await GenerationRouter(providers).aclose()
    assert all(p.closed == 1 for p in providers)


async def test_end_to_end_prompt_through_router_produces_a_cited_answer():
    """The composition the API layer actually performs, offline."""
    system, user = render(
        "how tall is mount everest",
        ["Mount Everest is 8,848 metres high.", "K2 is 8,611 metres high."],
    )
    router = GenerationRouter([FakeGenerator("groq", deltas=["8,848 m", " [1]."])])
    result = await router.complete(system, user)
    parsed = parse_answer(result.text)

    assert parsed.text == "8,848 m [1]."
    assert parsed.citations == (1,)
    assert parsed.invalid_citations(n_passages=2) == ()
    assert not parsed.abstained
