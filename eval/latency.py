"""The headline latency benchmark: transcript in, final answer token out.

The requirement, quoted verbatim from the organisers' brief:

    "The full process - chunking + vector DB retrieval + everything through to
    final output - should complete in under 200ms."

    "Submit P50 / P70 / P100 latency numbers for your pipeline, measured across
    a reasonable number of test queries - not a single best-case run."

Two readings of that sentence drive this module, and both are stated in the
emitted report so nobody has to reverse-engineer what was measured.

**It ends at the final output token, not the first.** Time-to-first-token is
the usual voice metric and it is far easier to hit. We measure to the *last*
token of the complete grounded answer. At Groq's throughput each output token
costs roughly a millisecond, which makes answer length an engineering
constraint rather than a style preference -- hence the 80-token cap in
:data:`voicerag.generate.base.DEFAULT_MAX_TOKENS`. TTFT is reported beside it
because it is what a user perceives, never *instead* of it.

**It begins at chunking/retrieval, not at the microphone.** The brief's own
pipeline diagram lists speech-to-text as a separate stage. So the 200 ms number
covers guardrails, query embedding, hybrid retrieval, fusion, abstention,
prompt construction and generation -- and STT and client network are reported
*separately and unmassaged* via :meth:`LatencyReport.with_client_measurements`,
so the framing hides nothing.

Four series come out of a run:

===================  =========================================================
``pipeline_total``   Transcript -> final token, answered queries only. **This
                     is the 200 ms claim.** Abstained queries are excluded
                     because they never call the LLM and would flatter the
                     number by tens of milliseconds each.
``pipeline_all``     The same, including abstentions. Reported because it is
                     what a user population actually experiences.
``ttft``             Transcript -> first token.
``stage.*``          Per-stage breakdown taken from the
                     :class:`~voicerag.harness.trace.Trace`, not from separate
                     instrumentation, so the benchmark and the live HUD read
                     the same spans.
===================  =========================================================

**Warmup runs are excluded from the percentiles and reported separately as
cold.** The first request through a fresh process pays costs that are real but
one-time: FAISS lazily allocating its per-thread visit tables, bm25s touching
its score matrix for the first time, the LSA component matrix paging in, CPython
warming its call caches, and -- in production -- the provider TLS handshake and
a prompt-cache miss. Mixing those into P50 measures process startup; hiding
them entirely would be dishonest. So they are measured, labelled ``cold`` and
printed next to the warm numbers.

When no LLM credentials are present the benchmark runs with
:class:`SimulatedGenerator` and every artefact it writes is stamped
``"generation": "simulated"``. A simulated number that is not labelled as one
is worse than no number at all.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, MutableMapping, Sequence

from voicerag.generate.base import DEFAULT_MAX_TOKENS, Generator
from voicerag.harness.trace import Trace, traced

from .metrics import DEFAULT_PERCENTILES, LatencySummary

__all__ = [
    "GROQ_GPT_OSS_20B_PROFILE",
    "GenerationProfile",
    "LatencyReport",
    "PipelineBenchmark",
    "RunRecord",
    "SimulatedGenerator",
    "benchmark",
    "report_to_markdown",
]


# --- simulated generation -----------------------------------------------------


#: Worst-case overshoot of a single ``asyncio.sleep``, per platform.
#:
#: Windows arms event-loop timers off the system tick, which defaults to
#: ~15.6 ms and is *not* reduced by ``perf_counter``'s 100 ns resolution. A
#: ``sleep(0.0001)`` there measures ~14 ms, so a 15-sleep token cadence that
#: should cost 3.4 ms costs 210 ms instead -- enough on its own to blow the
#: 200 ms budget this module exists to verify. POSIX timers land within about a
#: millisecond. Overestimating this constant only costs a little more
#: yield-spinning; underestimating it silently inflates every published number.
_TIMER_SLACK_NS = 20_000_000 if platform.system() == "Windows" else 2_000_000


async def _sleep_until(deadline_ns: int) -> None:
    """Sleep until ``deadline_ns`` on the ``perf_counter_ns`` clock.

    Two phases. While there is more margin than the platform timer can
    overshoot, sleep for real and let the loop idle. Inside that margin, spin --
    but spin on ``asyncio.sleep(0)``, which reschedules on the next loop pass
    *without arming a timer*. That costs microseconds instead of a tick, and
    unlike a busy-wait it still lets the concurrent grounding check make
    progress, which matters because overlapping it with generation is precisely
    what this benchmark is measuring.
    """
    remaining = deadline_ns - time.perf_counter_ns()
    if remaining > _TIMER_SLACK_NS:
        await asyncio.sleep((remaining - _TIMER_SLACK_NS) / 1e9)
    while time.perf_counter_ns() < deadline_ns:
        await asyncio.sleep(0)


@dataclass(slots=True, frozen=True)
class GenerationProfile:
    """Timing shape of a hosted LLM, for offline benchmarking.

    Attributes:
        ttft_ms: Delay before the first token.
        ms_per_token: Inter-token delay thereafter.
        answer: Text the simulated generator emits, tokenised on whitespace.
        label: Human-readable provenance, printed in the report so the reader
            knows where the numbers came from.
    """

    ttft_ms: float = 120.0
    ms_per_token: float = 1.0
    answer: str = (
        "The measured value is 212 degrees at standard pressure, according to "
        "the retrieved passage. [1]"
    )
    label: str = "generic"

    def tokens(self) -> list[str]:
        """Whitespace tokens, capped at the production ``max_tokens`` ceiling."""
        parts = self.answer.split()
        return [(p if i == 0 else " " + p) for i, p in enumerate(parts[:DEFAULT_MAX_TOKENS])]


#: Vendor-published shape for ``openai/gpt-oss-20b`` on Groq with
#: ``reasoning_effort="low"``. These are **documented estimates, not measured on
#: this machine** -- the benchmark container has no route to api.groq.com. Any
#: report produced with this profile is stamped ``simulated`` for exactly that
#: reason; re-run with ``GROQ_API_KEY`` set to replace it with real numbers.
GROQ_GPT_OSS_20B_PROFILE = GenerationProfile(
    ttft_ms=120.0,
    ms_per_token=1.0,
    label="groq/openai/gpt-oss-20b (vendor-published estimate, NOT measured here)",
)


class SimulatedGenerator(Generator):
    """A generator that emits a fixed answer on a configurable clock.

    Exists so the whole pipeline -- guardrails, retrieval, prompt assembly,
    streaming, trace assembly -- can be benchmarked end to end with no network,
    and so the *shape* of the latency budget (retrieval is free, decode
    dominates) is visible even offline.

    It is not a mock in the testing sense: it implements the real
    :class:`~voicerag.generate.base.Generator` contract and its output flows
    through the identical timing code that measures Groq, which is what makes
    the offline and online reports structurally comparable.

    Args:
        profile: Timing shape to reproduce.
        model: Model id recorded in the result.

    Note:
        Token deadlines are absolute offsets from the start of the stream, not
        a chain of relative sleeps, so per-token scheduling jitter cannot
        accumulate across an 80-token answer. What bias remains is a small
        positive one -- a simulated run slightly *over*-estimates, which is the
        right direction for a bias in a latency claim. See :func:`_sleep_until`
        for why a plain ``asyncio.sleep`` per token is not good enough.
    """

    name = "simulated"

    def __init__(
        self,
        profile: GenerationProfile = GROQ_GPT_OSS_20B_PROFILE,
        *,
        model: str = "simulated",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens, temperature=0.0)
        self.profile = profile

    async def _stream_deltas(
        self,
        system: str,
        user: str,
        *,
        meta: MutableMapping[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Sleep for TTFT, then emit tokens on the configured cadence."""
        del system, user
        tokens = self.profile.tokens()[: int(kwargs.get("max_tokens", self.max_tokens))]
        # Absolute schedule: token i is due ttft + i * ms_per_token after the
        # stream opens, measured from one fixed origin.
        origin = time.perf_counter_ns()
        if not tokens:  # a zero-token answer still pays for the prefill
            await _sleep_until(origin + int(self.profile.ttft_ms * 1e6))
        for i, token in enumerate(tokens):
            due_ms = self.profile.ttft_ms + i * self.profile.ms_per_token
            await _sleep_until(origin + int(due_ms * 1e6))
            yield token
        meta["finish_reason"] = "stop"
        meta["usage"] = {"completion_tokens": len(tokens), "simulated": True}
        meta["model"] = self.model


# --- one measured request -----------------------------------------------------


@dataclass(slots=True)
class RunRecord:
    """One measured pipeline request.

    Attributes:
        query: The transcript that was fed in.
        pipeline_ms: Transcript in -> final token out. The 200 ms quantity.
        ttft_ms: Transcript in -> first token out. ``0.0`` when the request
            abstained and no token was ever produced.
        stages: Per-stage milliseconds from the trace breakdown.
        abstained: Whether the guardrails stopped the request before generation.
        answer: The produced text (empty when abstained).
        n_deltas: Streamed chunks; a lower bound on output tokens.
        cold: True for warmup runs, which are excluded from the warm percentiles.
        error: Set when the request failed; failed runs never enter percentiles.
    """

    query: str
    pipeline_ms: float
    ttft_ms: float
    stages: dict[str, float] = field(default_factory=dict)
    abstained: bool = False
    answer: str = ""
    n_deltas: int = 0
    cold: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "pipeline_ms": round(self.pipeline_ms, 3),
            "ttft_ms": round(self.ttft_ms, 3),
            "stages": self.stages,
            "abstained": self.abstained,
            "n_deltas": self.n_deltas,
            "cold": self.cold,
            "error": self.error,
        }


#: What :func:`benchmark` needs from a pipeline: given a transcript, produce a
#: measured record. Anything satisfying this can be benchmarked -- the local
#: :class:`PipelineBenchmark`, or a thin wrapper around a deployed HTTP API.
RunnerFn = Callable[[str], Awaitable[RunRecord]]


class PipelineBenchmark:
    """An in-process request path assembled from the shipped components.

    Runs the same stages, in the same order, and under the **same span names**
    as :class:`~voicerag.pipeline.RagPipeline`: ``guard.input`` -> ``embed`` ->
    ``retrieve`` -> ``guard.abstention`` -> ``prompt`` -> ``generate``. The span
    names are shared deliberately -- the benchmark, the served response and the
    demo HUD all key off the same strings, so a stage cost means the same thing
    in all three.

    It is assembled here rather than importing ``RagPipeline`` so the benchmark
    can measure a stack built from any embedder/index/generator combination --
    including a duck-typed policy and the simulated generator -- without the API
    server, its settings object or its provider routing. The cost of that
    freedom is a second assembly of the same stage order; the shared span names
    are what keep the two comparable, and ``retrieve`` here is one span because
    this path calls ``asearch`` directly, where the served pipeline splits it
    into ``retrieve.dense``/``retrieve.sparse``/``fuse``.

    Args:
        embedder: Fitted embedder for the query vector.
        hybrid: Built :class:`~voicerag.index.hybrid.HybridIndex`.
        store: :class:`~voicerag.index.store.ChunkStore` for chunk payloads.
        generator: Any :class:`~voicerag.generate.base.Generator`.
        k: Passages retrieved and passed to the prompt.
        use_guardrails: Run the input guard and abstention gate. On by default
            because they are on the production path and they cost real
            microseconds that belong in the measurement.
        policy: A pre-built
            :class:`~voicerag.guardrails.policy.GuardrailPolicy`. Injectable
            because the abstention gate should be the *calibrated* one for the
            corpus being served -- benchmarking with the uncalibrated priors on
            a corpus whose score scale differs measures the wrong system, and
            an over-eager gate would flatter the latency numbers by skipping
            generation.
        max_tokens: Output ceiling forwarded to the generator.
    """

    __slots__ = ("embedder", "hybrid", "store", "generator", "k", "use_guardrails", "max_tokens", "_policy")

    def __init__(
        self,
        embedder: Any,
        hybrid: Any,
        store: Any,
        generator: Generator,
        *,
        k: int = 5,
        use_guardrails: bool = True,
        policy: Any | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.embedder = embedder
        self.hybrid = hybrid
        self.store = store
        self.generator = generator
        self.k = int(k)
        self.use_guardrails = bool(use_guardrails)
        self.max_tokens = int(max_tokens)
        self._policy: Any | None = policy

    def _policy_or_none(self) -> Any | None:
        """Lazily construct the guardrail policy.

        Imported inside the call rather than at module scope: the guardrails
        package is written independently of this one, and a benchmark of the
        retrieval path should not fail to import because a guardrail refactor
        is mid-flight.
        """
        if not self.use_guardrails:
            return None
        if self._policy is None:
            from voicerag.guardrails.policy import GuardrailPolicy

            self._policy = GuardrailPolicy()
        return self._policy

    async def run(self, query: str, *, cold: bool = False) -> RunRecord:
        """Execute one full request and return its measurement.

        Args:
            query: The transcript, exactly as STT would deliver it.
            cold: Mark the record as a warmup run.

        Returns:
            A :class:`RunRecord`. Exceptions are captured into ``error`` rather
            than propagated, so one pathological query cannot abort a 200-query
            benchmark; failed runs are excluded from every percentile and
            counted in the report.
        """
        trace = Trace()
        t0 = time.perf_counter_ns()
        record = RunRecord(query=query, pipeline_ms=0.0, ttft_ms=0.0, cold=cold)

        try:
            with traced(trace):
                policy = self._policy_or_none()
                if policy is not None:
                    with trace.span("guard.input"):
                        verdict = policy.check_input(query)
                    if not verdict.allowed:
                        record.abstained = True
                        record.stages = trace.breakdown()
                        record.pipeline_ms = (time.perf_counter_ns() - t0) / 1e6
                        return record

                with trace.span("embed"):
                    vec = self.embedder.encode([query])[0]

                with trace.span("retrieve"):
                    hits = await self.hybrid.asearch(
                        query_text=query, query_vec=vec, k=self.k
                    )

                if policy is not None:
                    with trace.span("guard.abstention"):
                        abstention = policy.check_retrieval(hits)
                    if abstention.should_abstain:
                        record.abstained = True
                        record.stages = trace.breakdown()
                        record.pipeline_ms = (time.perf_counter_ns() - t0) / 1e6
                        return record

                with trace.span("prompt"):
                    from voicerag.generate.prompt import render

                    chunks = self.store.get_many([h.chunk_id for h in hits])
                    system, user = render(query, [c.text for c in chunks])

                result = await self.generator.complete(
                    system, user, max_tokens=self.max_tokens
                )
                record.answer = result.text
                record.ttft_ms = (
                    (time.perf_counter_ns() - t0) / 1e6 - result.total_ms + result.ttft_ms
                )
                record.n_deltas = result.n_deltas
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            record.error = f"{type(exc).__name__}: {exc}"

        record.pipeline_ms = (time.perf_counter_ns() - t0) / 1e6
        record.stages = trace.breakdown()
        return record

    async def aclose(self) -> None:
        """Release the generator transport and the retriever's thread pool."""
        await self.generator.aclose()
        self.hybrid.close()


# --- the report ---------------------------------------------------------------


@dataclass(slots=True)
class LatencyReport:
    """Percentile summaries plus everything needed to interpret them.

    Attributes:
        summaries: Series name -> :class:`~eval.metrics.LatencySummary`.
        records: Every measured run, warm and cold, for re-analysis.
        meta: Provenance -- generation source, iteration counts, host, corpus
            size, timestamp.
        target_ms: The bar being tested against.
    """

    summaries: dict[str, LatencySummary] = field(default_factory=dict)
    records: list[RunRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    target_ms: float = 200.0

    @property
    def simulated(self) -> bool:
        """True when generation timing came from :class:`SimulatedGenerator`."""
        return bool(self.meta.get("generation") == "simulated")

    @property
    def passes(self) -> bool | None:
        """Whether P100 of the headline series is under the bar.

        Returns ``None`` when no answered run exists, because "pass" is not a
        claim that can be made from zero samples.
        """
        head = self.summaries.get("pipeline_total")
        if head is None or not head.pcts:
            return None
        return head.pcts.get("p100", float("inf")) < self.target_ms

    def with_client_measurements(
        self,
        *,
        stt_ms: Sequence[float] = (),
        network_ms: Sequence[float] = (),
    ) -> "LatencyReport":
        """Attach client-side series measured outside the server budget.

        STT and browser<->API round trip are real user-visible latency, but they
        are not part of the quantity the brief scopes to 200 ms, and they are
        measured on a different clock (the browser's). They are attached as
        separate series and rendered in their own table section -- never summed
        into the pipeline number.

        Args:
            stt_ms: End-of-speech to final transcript, measured client-side.
            network_ms: Browser to API round trip.

        Returns:
            ``self``, mutated, for chaining.
        """
        if stt_ms:
            self.summaries["stt_client"] = LatencySummary.from_samples("stt_client", list(stt_ms))
        if network_ms:
            self.summaries["client_network"] = LatencySummary.from_samples(
                "client_network", list(network_ms)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_ms": self.target_ms,
            "passes_p100": self.passes,
            "meta": self.meta,
            "summaries": {k: v.to_dict() for k, v in self.summaries.items()},
            "records": [r.to_dict() for r in self.records],
        }

    def write_json(self, path: str | Path) -> Path:
        """Write the full report, records included, as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    def to_markdown(self) -> str:
        """Render the report as markdown (see :func:`report_to_markdown`)."""
        return report_to_markdown(self)


#: Series rendered in the headline table, in order, with their explanations.
_HEADLINE_SERIES: tuple[tuple[str, str], ...] = (
    ("pipeline_total", "**Pipeline (answered)** - transcript in -> final token out"),
    ("pipeline_all", "Pipeline incl. abstentions"),
    ("ttft", "Time to first token"),
    ("pipeline_cold", "Cold (warmup runs, excluded above)"),
)

_CLIENT_SERIES: tuple[tuple[str, str], ...] = (
    ("stt_client", "Speech-to-text (client-measured, Sarvam)"),
    ("client_network", "Browser <-> API round trip"),
)


def _table(rows: Sequence[tuple[str, LatencySummary]]) -> list[str]:
    cols = [f"p{int(p)}" for p in DEFAULT_PERCENTILES]
    out = [
        "| Series | n | mean | " + " | ".join(c.upper() for c in cols) + " |",
        "|---|---:|---:|" + "|".join(["---:"] * len(cols)) + "|",
    ]
    for label, summary in rows:
        cells = [f"{summary.pcts.get(c, float('nan')):.1f}" for c in cols]
        out.append(
            f"| {label} | {summary.n} | {summary.mean:.1f} | " + " | ".join(cells) + " |"
        )
    return out


def report_to_markdown(report: LatencyReport) -> str:
    """Render a :class:`LatencyReport` as a self-explaining markdown section.

    The output states what was measured, what was excluded and why, and whether
    generation was real or simulated -- so the table cannot be quoted out of
    context by accident.
    """
    meta = report.meta
    out: list[str] = ["### Latency", ""]
    out.append(
        f"Target: **< {report.target_ms:.0f} ms** from transcript in to the "
        "**final** answer token (the brief's wording), measured over "
        f"{meta.get('iterations', '?')} warm runs across "
        f"{meta.get('n_queries', '?')} distinct queries."
    )
    out.append("")
    if report.simulated:
        # The reason matters. This banner used to assert "No LLM credentials
        # were available" unconditionally, which is a false statement in a
        # published artifact whenever the run was simulated deliberately with a
        # key present -- exactly the kind of claim this repo holds itself to.
        reason = meta.get("simulated_reason") or (
            "No LLM credentials were available."
        )
        out.append(
            f"> **Generation is SIMULATED in this run.** {reason} Decode "
            f"timing came from `{meta.get('profile', 'profile')}`. "
            "Retrieval, guardrail and prompt numbers are real measurements; the "
            "generation and total rows are a model of the provider, not an "
            "observation of it."
        )
        out.append("")

    rows = [
        (label, report.summaries[key])
        for key, label in _HEADLINE_SERIES
        if key in report.summaries and report.summaries[key].n
    ]
    out.extend(_table(rows))
    out.append("")

    stages = sorted(
        ((k, v) for k, v in report.summaries.items() if k.startswith("stage.")),
        key=lambda kv: -kv[1].mean,
    )
    if stages:
        out.extend(["#### Per-stage breakdown (warm, answered runs)", ""])
        out.extend(_table([(k.removeprefix("stage."), v) for k, v in stages]))
        out.append("")
        out.append(
            "_Stage times come from the same `Trace` spans the API returns and "
            "the HUD renders. Spans may overlap; they are not expected to sum to "
            "the pipeline total._"
        )
        out.append("")

    client = [(label, report.summaries[k]) for k, label in _CLIENT_SERIES if k in report.summaries]
    if client:
        out.extend(
            [
                "#### Reported separately (outside the 200 ms scope)",
                "",
                *_table(client),
                "",
            ]
        )

    verdict = report.passes
    if verdict is not None:
        head = report.summaries["pipeline_total"]
        out.append(
            f"**Verdict:** P100 = {head.pcts.get('p100', 0.0):.1f} ms "
            f"{'<' if verdict else '>='} {report.target_ms:.0f} ms target "
            f"(P50 = {head.pcts.get('p50', 0.0):.1f} ms, "
            f"P70 = {head.pcts.get('p70', 0.0):.1f} ms)."
        )
    if meta.get("n_failed"):
        out.append("")
        out.append(f"_{meta['n_failed']} run(s) errored and are excluded; see `records` in the JSON._")
    return "\n".join(out) + "\n"


# --- the driver ---------------------------------------------------------------


async def benchmark(
    runner: RunnerFn | PipelineBenchmark,
    queries: Sequence[str],
    *,
    iterations: int = 100,
    warmup: int = 5,
    target_ms: float = 200.0,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    meta: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> LatencyReport:
    """Measure a pipeline over many queries and summarise it.

    Args:
        runner: A :class:`PipelineBenchmark`, or any async callable taking a
            transcript and returning a :class:`RunRecord`.
        queries: Transcripts to cycle through. Cycled rather than sampled so a
            run is reproducible and every query gets equal weight.
        iterations: Warm measured runs. The brief asks for "a reasonable number
            of test queries"; 100+ is the intended setting.
        warmup: Runs executed first, recorded as ``cold``, excluded from the
            warm percentiles. See the module docstring for what they pay for.
        target_ms: The bar, echoed into the report.
        percentiles: Percentiles to compute (nearest-rank).
        meta: Extra provenance merged into the report.
        progress: Called with a status string every 10% of the run.

    Returns:
        A populated :class:`LatencyReport`.

    Raises:
        ValueError: If ``queries`` is empty or ``iterations`` is not positive.
    """
    if not queries:
        raise ValueError("benchmark needs at least one query")
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")

    bench = runner if isinstance(runner, PipelineBenchmark) else None
    run: RunnerFn = (
        (lambda q: bench.run(q, cold=False)) if bench is not None else runner  # type: ignore[union-attr]
    )

    records: list[RunRecord] = []
    total = warmup + iterations
    step = max(1, total // 10)
    for i in range(total):
        cold = i < warmup
        query = queries[(i - warmup) % len(queries)] if not cold else queries[i % len(queries)]
        record = await run(query)
        record.cold = cold
        records.append(record)
        if progress and i % step == 0:
            progress(f"{i + 1}/{total} runs ({'cold' if cold else 'warm'})")

    warm = [r for r in records if not r.cold and r.error is None]
    cold_runs = [r for r in records if r.cold and r.error is None]
    answered = [r for r in warm if not r.abstained]

    summaries: dict[str, LatencySummary] = {
        "pipeline_total": LatencySummary.from_samples(
            "pipeline_total", [r.pipeline_ms for r in answered], percentiles
        ),
        "pipeline_all": LatencySummary.from_samples(
            "pipeline_all", [r.pipeline_ms for r in warm], percentiles
        ),
        "ttft": LatencySummary.from_samples(
            "ttft", [r.ttft_ms for r in answered if r.ttft_ms > 0], percentiles
        ),
        "pipeline_cold": LatencySummary.from_samples(
            "pipeline_cold", [r.pipeline_ms for r in cold_runs], percentiles
        ),
    }

    stage_names = {name for r in answered for name in r.stages}
    for name in sorted(stage_names):
        samples = [r.stages[name] for r in answered if name in r.stages]
        summaries[f"stage.{name}"] = LatencySummary.from_samples(
            f"stage.{name}", samples, percentiles
        )

    provenance: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "iterations": len(warm),
        "warmup": len(cold_runs),
        "n_queries": len(queries),
        "n_abstained": sum(1 for r in warm if r.abstained),
        "n_failed": sum(1 for r in records if r.error is not None),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
    }
    if bench is not None:
        gen = bench.generator
        simulated = isinstance(gen, SimulatedGenerator)
        provenance["generation"] = "simulated" if simulated else f"real:{gen.name}"
        provenance["model"] = gen.model
        provenance["max_tokens"] = bench.max_tokens
        provenance["k"] = bench.k
        if simulated:
            provenance["profile"] = gen.profile.label  # type: ignore[union-attr]
    provenance.update(meta or {})

    return LatencyReport(
        summaries=summaries, records=records, meta=provenance, target_ms=target_ms
    )
