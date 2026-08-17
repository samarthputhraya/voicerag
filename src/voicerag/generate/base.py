"""Generator interface and the latency instrumentation that makes it measurable.

The organisers' bar is *full completion* under 200 ms -- not time-to-first-token.
That single sentence dictates the whole design of this package:

* **Answer length is a hard engineering constraint.** At Groq's throughput each
  output token costs roughly a millisecond, so an 80-token answer spends 80 ms
  of a 200 ms budget on decoding alone. ``max_tokens`` is therefore a
  correctness-adjacent parameter, not a safety net, and it is enforced here at
  the interface level so no backend can quietly opt out of it.
* **Two numbers must be measured, not one.** TTFT is what the demo *feels* like;
  total completion is what the demo is *scored* on. Reporting only the flattering
  one would be dishonest, so every call emits both, as two separate spans named
  ``generate.ttft`` and ``generate.total``.

TTFT is defined as the time to the first **non-empty** delta. Both providers open
a stream with structural chunks that carry an empty ``content`` -- a role marker
on Groq, an empty part on Gemini -- and counting those as "first token" would
report a TTFT that no user ever experienced. That distinction is worth several
tens of milliseconds of apparent latency, in the wrong direction.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping, MutableMapping

from ..harness.trace import current_trace

__all__ = [
    "GenerationResult",
    "Generator",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
]

#: Output-token ceiling. 80 tokens is roughly two sentences -- comfortably more
#: than an MS MARCO factoid answer needs, and about 80 ms of decode time. Raising
#: it is the single easiest way to fail the latency requirement.
DEFAULT_MAX_TOKENS = 80

#: Greedy decoding. A grounded extractive answer has no use for sampling
#: diversity, and determinism is what makes the evaluation numbers reproducible.
DEFAULT_TEMPERATURE = 0.0


@dataclass(slots=True)
class GenerationResult:
    """One completed generation and its timing.

    Args:
        text: The full concatenated answer.
        ttft_ms: Milliseconds to the first non-empty delta.
        total_ms: Milliseconds to the last delta. This is the number compared
            against the 200 ms bar.
        n_deltas: Non-empty streamed chunks. A lower bound on output tokens and
            the fallback when the provider reports no usage.
        provider: Which generator produced this, for the fallback audit trail.
        model: Concrete model id.
        finish_reason: Provider's stop reason, e.g. ``"stop"`` or ``"length"``.
        usage: Provider-reported token counts, verbatim, when available.
    """

    text: str
    ttft_ms: float
    total_ms: float
    n_deltas: int
    provider: str
    model: str
    finish_reason: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        """True if the model was cut off by ``max_tokens``.

        Worth surfacing rather than hiding: a truncated answer means the prompt
        failed to keep the model terse, which is a prompt bug with a latency
        cost, and it should show up in the evaluation rather than as a sentence
        that trails off mid-word in the demo.
        """
        return self.finish_reason == "length"

    @property
    def decode_ms(self) -> float:
        """Time spent decoding after the first token arrived."""
        return max(0.0, self.total_ms - self.ttft_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "ttft_ms": round(self.ttft_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "decode_ms": round(self.decode_ms, 3),
            "n_deltas": self.n_deltas,
            "provider": self.provider,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
            "usage": dict(self.usage),
        }


class Generator(ABC):
    """Base class for streaming LLM backends.

    Subclasses implement :meth:`_stream_deltas`; the timing, tracing and
    accounting in :meth:`stream` and :meth:`complete` are shared so that every
    provider's numbers are produced by identical code and are therefore
    comparable.

    Args:
        model: Concrete model identifier sent to the provider.
        max_tokens: Output ceiling; see :data:`DEFAULT_MAX_TOKENS`.
        temperature: Sampling temperature; see :data:`DEFAULT_TEMPERATURE`.

    Raises:
        ValueError: If ``max_tokens`` is not positive.
    """

    #: Stable short identifier used in traces, breakers and fallback tables.
    name: str = "generator"

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------ hooks

    @abstractmethod
    def _stream_deltas(
        self,
        system: str,
        user: str,
        *,
        meta: MutableMapping[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield raw text deltas from the provider.

        Args:
            system: System prompt. Byte-identical across requests, for cache hits.
            user: User turn carrying context and question.
            meta: Out-parameter the backend fills with ``finish_reason``,
                ``usage`` and any provider detail worth keeping. An
                out-parameter rather than a return value because this is an
                async *generator*: it has no return channel, and the metadata
                only exists after the last chunk has been consumed.
            **kwargs: Per-call overrides such as ``max_tokens``.

        Yields:
            Text deltas, possibly empty. The base class filters empties.
        """

    async def aclose(self) -> None:
        """Release provider resources. Safe to call more than once.

        A no-op by default; HTTP-backed generators close their connection pool.
        """

    # ------------------------------------------------------------ public

    async def stream(
        self,
        system: str,
        user: str,
        *,
        meta: MutableMapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream an answer, emitting ``generate.ttft`` and ``generate.total``.

        Args:
            system: System prompt.
            user: User turn.
            meta: Optional dict to receive provider metadata and the measured
                ``ttft_ms`` / ``total_ms``.
            **kwargs: Passed through to the backend.

        Yields:
            Non-empty text deltas, in order.
        """
        sink: MutableMapping[str, Any] = meta if meta is not None else {}
        trace = current_trace()
        t0 = time.perf_counter_ns()
        ttft_ns: int | None = None
        n_deltas = 0
        error: BaseException | None = None

        # Spans are entered manually because they do not nest lexically: the
        # TTFT span must close on the first token while the total span stays
        # open until the last one. Entry order (total, then ttft) keeps the
        # trace's parent stack correctly LIFO on every exit path.
        total_cm = trace.span("generate.total", provider=self.name, model=self.model) if trace else None
        ttft_cm = trace.span("generate.ttft", provider=self.name) if trace else None
        total_span = total_cm.__enter__() if total_cm is not None else None
        if ttft_cm is not None:
            ttft_cm.__enter__()
        ttft_open = ttft_cm is not None

        try:
            async for delta in self._stream_deltas(system, user, meta=sink, **kwargs):
                if not delta:
                    continue
                if ttft_ns is None:
                    ttft_ns = time.perf_counter_ns()
                    if ttft_open and ttft_cm is not None:
                        ttft_cm.__exit__(None, None, None)
                        ttft_open = False
                n_deltas += 1
                yield delta
        except BaseException as exc:  # noqa: BLE001 - recorded on the span, re-raised
            error = exc
            raise
        finally:
            now = time.perf_counter_ns()
            sink["ttft_ms"] = (ttft_ns - t0) / 1e6 if ttft_ns is not None else 0.0
            sink["total_ms"] = (now - t0) / 1e6
            sink["n_deltas"] = n_deltas
            exc_info = (
                (type(error), error, error.__traceback__)
                if error is not None
                else (None, None, None)
            )
            if ttft_open and ttft_cm is not None:
                # No token ever arrived: the TTFT span is closed at the same
                # instant as the total span so the trace shows the failure
                # rather than a zero-length span at the start.
                ttft_cm.__exit__(*exc_info)
            if total_cm is not None:
                if total_span is not None:
                    total_span.attrs["n_deltas"] = n_deltas
                    total_span.attrs["ttft_ms"] = round(sink["ttft_ms"], 3)
                total_cm.__exit__(*exc_info)

    async def complete(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> GenerationResult:
        """Stream to completion and return the answer with its timing.

        Args:
            system: System prompt.
            user: User turn.
            **kwargs: Passed through to the backend.

        Returns:
            A :class:`GenerationResult`. Timings come from the same measurement
            :meth:`stream` records, so the result and the trace can never
            disagree.
        """
        meta: dict[str, Any] = {}
        parts: list[str] = []
        async for delta in self.stream(system, user, meta=meta, **kwargs):
            parts.append(delta)
        return GenerationResult(
            text="".join(parts),
            ttft_ms=float(meta.get("ttft_ms", 0.0)),
            total_ms=float(meta.get("total_ms", 0.0)),
            n_deltas=int(meta.get("n_deltas", len(parts))),
            provider=self.name,
            model=str(meta.get("model", self.model)),
            finish_reason=meta.get("finish_reason"),
            usage=meta.get("usage") or {},
        )
