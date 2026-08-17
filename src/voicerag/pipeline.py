"""The single orchestrated request path: transcript in, grounded answer out.

Everything the service does to a question happens here, in one place, in one
order, under one :class:`~voicerag.harness.resilience.Deadline`:

``guard.input`` -> ``embed`` -> ``retrieve.dense`` + ``retrieve.sparse`` ->
``fuse`` -> ``guard.abstention`` -> ``prompt`` -> ``generate`` ->
``guard.grounding``

Design decisions worth stating, because each one is a place a RAG pipeline
usually goes wrong:

**Abstention short-circuits before the LLM, not after it.** When the gate says
the corpus cannot answer, the generator is never invoked. That is the whole
point: refusing costs a few hundred microseconds while answering costs a hundred
milliseconds, so the guardrail is a latency feature as much as a safety one.
``test_pipeline.py`` asserts the generator's call count is zero on that path --
an assertion rather than a claim.

**One implementation, two entry points.** :meth:`RagPipeline.answer` is
:meth:`RagPipeline.astream` drained to completion. A streaming and a
non-streaming path that share no code eventually disagree about citations,
guardrail outcomes or provider attribution; here they cannot.

**Every stage gets a slice of the deadline, not the whole thing.** A stage that
would exceed the budget is cut short and the failure is recorded as *that*
stage's, so a slow request names its own culprit in the trace. Deadline
exhaustion degrades: an answer that has already streamed partially is returned
with what it has, flagged, rather than raising after the user has read half a
sentence.

**Grounding overlaps generation.** Sentences are verified as they complete
rather than after the last token. The saving is small in absolute terms (the
lexical checker costs ~0.3 ms) and is reported as such -- it is here because it
is the correct shape for the stage, not because it rescues the budget.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from .chunking.base import Chunk
from .config import Settings, get_settings
from .generate.base import Generator
from .generate.prompt import parse_answer, render, usable_passages
from .generate.router import GenerationRouter
from .guardrails.grounding import ClaimVerdict, GroundingVerdict
from .guardrails.policy import GuardrailPolicy, GuardrailReport
from .harness.resilience import AllProvidersFailed, BudgetExhausted, Deadline
from .harness.trace import Span, Trace, traced
from .index.hybrid import HybridIndex, RetrievalHit
from .index.store import ChunkStore
from .stt.speculative import normalise_query

__all__ = [
    "Citation",
    "GuardrailBlock",
    "RagResponse",
    "RagEvent",
    "RagPipeline",
    "SpeculationCache",
    "SpeculationBlock",
    "TraceBlock",
    "RetrievalResult",
    "PipelineNotReady",
]


class PipelineNotReady(RuntimeError):
    """The pipeline was asked to answer before an index was loaded."""


# --- response schema ----------------------------------------------------------
# These models are a fixed contract with `web/lib/api.ts`. Field names and
# nesting are asserted against that file's TypeScript interfaces in
# tests/test_api.py; renaming one here silently blanks a panel in the HUD.


class Citation(BaseModel):
    """One retrieved passage offered as evidence for the answer."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    title: str | None = None


class GuardrailBlock(BaseModel):
    """The guardrail outcome, exactly as the HUD renders it."""

    input_allowed: bool = True
    input_reason: str | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    abstain_confidence: float = 0.0
    abstain_signals: dict[str, float] = Field(default_factory=dict)
    grounded: bool | None = None
    grounding_score: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)

    @classmethod
    def from_report(cls, report: GuardrailReport) -> GuardrailBlock:
        """Project the internal report onto the nine fields the frontend reads."""
        return cls(
            input_allowed=report.input_allowed,
            input_reason=report.input_reason,
            abstained=report.abstained,
            abstain_reason=report.abstain_reason,
            abstain_confidence=round(report.abstain_confidence, 4),
            abstain_signals={k: round(v, 5) for k, v in report.abstain_signals.items()},
            grounded=report.grounded,
            grounding_score=(
                None if report.grounding_score is None else round(report.grounding_score, 4)
            ),
            unsupported_claims=list(report.unsupported_claims),
        )


class TraceBlock(BaseModel):
    """Span timings for the latency HUD."""

    trace_id: str
    total_ms: float
    critical_path_ms: float
    breakdown: dict[str, float] = Field(default_factory=dict)


class SpeculationBlock(BaseModel):
    """What speculative retrieval did for this request."""

    launched: int = 0
    cancelled: int = 0
    hit: bool = False
    saved_ms: float = 0.0


class RagResponse(BaseModel):
    """The complete answer to one question.

    Attributes:
        answer: The spoken answer, or the refusal text when ``abstained``.
        citations: Evidence passages. When the model emitted valid ``[n]``
            markers these are the cited passages, in citation order; otherwise
            the retrieved context, so the UI always has something to show.
        abstained: True whenever no generated answer is being asserted -- input
            blocked, retrieval insufficient, or generation ungrounded.
        abstain_reason: The specific, quantitative cause.
        guardrails: All three guardrail stages, folded into one report.
        provider: Which backend answered (``"none"`` when nothing was called).
        trace: Span timings, the same ones the benchmark reads.
        speculation: Speculative-retrieval accounting, ``None`` when disabled.
    """

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None
    guardrails: GuardrailBlock = Field(default_factory=GuardrailBlock)
    provider: str = "none"
    trace: TraceBlock
    speculation: SpeculationBlock | None = None


@dataclass(slots=True, frozen=True)
class RagEvent:
    """One item from :meth:`RagPipeline.astream`.

    Attributes:
        kind: ``"token"`` for an incremental delta, ``"final"`` for the single
            terminating event that carries the assembled response.
        delta: Text delta; empty on the final event.
        response: The complete :class:`RagResponse`; ``None`` on token events.
    """

    kind: Literal["token", "final"]
    delta: str = ""
    response: RagResponse | None = None


# --- retrieval ----------------------------------------------------------------


@dataclass(slots=True)
class RetrievalResult:
    """A retrieval outcome plus the timings needed to attribute it.

    Cached by :class:`SpeculationCache` and handed to
    :meth:`RagPipeline.answer` as ``speculative_hits``.

    Attributes:
        hits: Fused hits, best first.
        query_vec: The query embedding, kept so a cache hit skips the embed
            stage too -- which is the expensive half when the fallback LSA
            embedder is in use.
        embed_ms: Measured cost of producing ``query_vec``.
        retrieve_ms: Measured cost of the fused search.
        dense_span: ``(start_ns, end_ns)`` of the dense run, or ``None``.
        sparse_span: ``(start_ns, end_ns)`` of the sparse run, or ``None``.
        created_ns: ``perf_counter_ns`` at completion, for TTL checks.
    """

    hits: list[RetrievalHit]
    query_vec: np.ndarray | None = None
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    dense_span: tuple[int, int] | None = None
    sparse_span: tuple[int, int] | None = None
    created_ns: int = field(default_factory=time.perf_counter_ns)

    @property
    def total_ms(self) -> float:
        """Milliseconds a cache hit avoids paying."""
        return self.embed_ms + self.retrieve_ms


class _TimedIndex:
    """Delegating proxy that records when a sub-index's ``search`` ran.

    :class:`~voicerag.index.hybrid.HybridIndex` runs its dense and sparse legs
    on a thread pool and reports one fused result, so the pipeline cannot see
    the split from the outside -- but the HUD colours ``retrieve.dense`` and
    ``retrieve.sparse`` separately, and "hybrid retrieval was 3 ms" hides which
    leg is the tail.

    This wrapper records ``perf_counter_ns`` around the delegated call into a
    plain tuple. It deliberately does *not* open a
    :class:`~voicerag.harness.trace.Trace` span from the worker thread: the
    trace keeps a parent stack that is only correct when pushed and popped from
    one thread. The pipeline converts the recorded interval into a
    :class:`~voicerag.harness.trace.Span` afterwards, on the event loop, which
    is both thread-safe and produces identical numbers.

    Attributes:
        span: ``(start_ns, end_ns)`` of the most recent search, or ``None``.
    """

    __slots__ = ("_inner", "span")

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.span: tuple[int, int] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __len__(self) -> int:
        return len(self._inner)

    def search(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter_ns()
        try:
            return self._inner.search(*args, **kwargs)
        finally:
            self.span = (start, time.perf_counter_ns())


class SpeculationCache:
    """Retrieval results keyed by the normalised transcript that produced them.

    The frontend POSTs partial transcripts to ``/speculate`` while the user is
    still speaking. By the time the final transcript arrives it is very often
    byte-identical to the last partial, so the embed and retrieve stages are
    already done and ``/ask`` skips them entirely.

    Only *exact* normalised-key matches hit. The similarity-based variant lives
    in :class:`~voicerag.stt.speculative.SpeculativeTranscriptDriver`, which has
    an embedder and a cancellation model; a server-side cache shared between
    requests should not silently answer one user's question with another's
    near-neighbour retrieval, so the server sticks to exact keys.

    Args:
        max_size: Entries retained; oldest evicted first.
        ttl_s: Age after which an entry is ignored. A stale speculation is
            worse than none, because the index may have been rebuilt under it.
    """

    __slots__ = ("_entries", "_max_size", "_ttl_ns", "launched", "hits", "misses")

    def __init__(self, *, max_size: int = 64, ttl_s: float = 30.0) -> None:
        self._entries: dict[str, RetrievalResult] = {}
        self._max_size = max(1, int(max_size))
        self._ttl_ns = int(ttl_s * 1e9)
        self.launched = 0
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def key(self, text: str) -> str:
        """Normalised cache key for a transcript."""
        return normalise_query(text)

    def put(self, text: str, result: RetrievalResult) -> None:
        """Store a speculation, evicting the oldest entry if full."""
        key = self.key(text)
        if key in self._entries:
            del self._entries[key]
        elif len(self._entries) >= self._max_size:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = result
        self.launched += 1

    def take(self, text: str) -> RetrievalResult | None:
        """Consume the speculation for ``text``, if a fresh one exists.

        Consuming rather than peeking: a speculation answers exactly one final
        transcript, and leaving it behind would let a later, different question
        with the same normalised form reuse a stale retrieval.
        """
        key = self.key(text)
        entry = self._entries.pop(key, None)
        if entry is None:
            self.misses += 1
            return None
        if time.perf_counter_ns() - entry.created_ns > self._ttl_ns:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def clear(self) -> None:
        """Drop every entry. Called when the index is reloaded."""
        self._entries.clear()

    def stats(self) -> dict[str, Any]:
        """Counters for ``/stats`` and the demo HUD."""
        resolutions = self.hits + self.misses
        return {
            "size": len(self._entries),
            "launched": self.launched,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / resolutions, 4) if resolutions else 0.0,
        }


# --- streaming grounding ------------------------------------------------------


class _StreamingGrounder:
    """Verifies answer sentences as they complete, rather than at the end.

    ``GroundingChecker`` exposes :meth:`split_claims` and :meth:`verify_claim`
    precisely so a streaming caller can do this. Aggregation into an
    answer-level :class:`~voicerag.guardrails.grounding.GroundingVerdict` is
    delegated back to the checker's own aggregator rather than re-derived here:
    the thresholds, the length weighting and the hard-failure rules must have
    exactly one implementation, or the streaming path and the batch path will
    eventually disagree about whether the same answer was grounded.
    ``test_pipeline.py::test_streaming_grounding_matches_batch`` asserts the two
    produce identical verdicts.

    Args:
        checker: The configured
            :class:`~voicerag.guardrails.grounding.GroundingChecker`.
        contexts: Passages in citation order (``[1]`` is ``contexts[0]``).
    """

    __slots__ = ("_checker", "_contexts", "_prepared", "_verdicts", "_t0")

    def __init__(self, checker: Any, contexts: Sequence[Any]) -> None:
        self._checker = checker
        self._contexts = contexts
        self._prepared = checker._prepare(contexts)  # noqa: SLF001 - see class docstring
        self._verdicts: list[ClaimVerdict] = []
        self._t0 = time.perf_counter_ns()

    def feed(self, text_so_far: str) -> None:
        """Verify any sentence that has become complete since the last call.

        The trailing fragment is left alone: a sentence is only checkable once
        its final token has arrived, and checking a prefix would report a
        missing number that the next token supplies.
        """
        claims = self._checker.split_claims(text_so_far)
        if len(claims) <= len(self._verdicts) + 1:
            return  # only the in-progress sentence is new
        for claim in claims[len(self._verdicts) : -1]:
            self._verdicts.append(
                self._checker.verify_claim(claim, self._contexts, prepared=self._prepared)
            )

    def finish(self, answer: str) -> GroundingVerdict:
        """Verify whatever is left and aggregate the answer-level verdict."""
        claims = self._checker.split_claims(answer)
        if not claims:
            return GroundingVerdict(
                grounded=False,
                score=0.0,
                latency_ms=(time.perf_counter_ns() - self._t0) / 1e6,
            )
        # A sentence boundary can move when later tokens arrive ("Dr." then
        # "Dr. Bose"), so anything already verified is trusted only while it
        # still matches the final split.
        keep = 0
        for verdict, claim in zip(self._verdicts, claims, strict=False):
            if verdict.text != claim:
                break
            keep += 1
        verdicts = self._verdicts[:keep]
        for claim in claims[keep:]:
            verdicts.append(
                self._checker.verify_claim(claim, self._contexts, prepared=self._prepared)
            )
        used_nli = False
        if self._checker.nli is not None:
            texts = [c if isinstance(c, str) else getattr(c, "text", "") for c in self._contexts]
            used_nli = self._checker._rescue_with_nli(verdicts, texts)  # noqa: SLF001
        return self._checker._finalise(verdicts, used_nli, self._t0)  # noqa: SLF001


# --- the pipeline -------------------------------------------------------------


class RagPipeline:
    """The orchestrated request path.

    Args:
        embedder: Anything with ``encode(list[str]) -> np.ndarray``.
        hybrid: A built :class:`~voicerag.index.hybrid.HybridIndex`.
        store: The :class:`~voicerag.index.store.ChunkStore` holding payloads.
        generator: A :class:`~voicerag.generate.router.GenerationRouter`, a bare
            :class:`~voicerag.generate.base.Generator`, or a sequence of
            generators in preference order. ``None`` means no LLM is configured;
            the pipeline still serves the guardrail and retrieval stages and
            reports a provider failure instead of answering.
        policy: Guardrail policy. Built from ``settings`` when omitted.
        settings: Configuration. :func:`~voicerag.config.get_settings` when
            omitted.
        speculation: Shared speculation cache, or ``None`` to disable.

    Raises:
        PipelineNotReady: From :meth:`answer` when ``hybrid`` or ``store`` is
            missing, which is how a server that started without an index
            reports itself rather than raising ``AttributeError`` mid-request.
    """

    __slots__ = (
        "embedder",
        "hybrid",
        "store",
        "router",
        "policy",
        "settings",
        "speculation",
        "_dense",
        "_sparse",
        "_search_index",
    )

    def __init__(
        self,
        *,
        embedder: Any = None,
        hybrid: HybridIndex | None = None,
        store: ChunkStore | None = None,
        generator: GenerationRouter | Generator | Sequence[Generator] | None = None,
        policy: GuardrailPolicy | None = None,
        settings: Settings | None = None,
        speculation: SpeculationCache | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.hybrid = hybrid
        self.store = store
        self.router = _as_router(generator)
        self.policy = policy or build_policy(self.settings)
        self.speculation = speculation
        self._dense: _TimedIndex | None = None
        self._sparse: _TimedIndex | None = None
        self._search_index: HybridIndex | None = None
        if hybrid is not None:
            self._search_index = self._instrument(hybrid)

    def __repr__(self) -> str:
        n = 0 if self.store is None else len(self.store)
        providers = "none" if self.router is None else ",".join(
            p.name for p in self.router.providers
        )
        return f"RagPipeline(chunks={n}, providers={providers})"

    # -- construction helpers -------------------------------------------------

    def _instrument(self, hybrid: HybridIndex) -> HybridIndex:
        """Wrap the sub-indexes so each retrieval leg can be timed separately.

        A second :class:`~voicerag.index.hybrid.HybridIndex` is constructed over
        the proxies with the original's configuration copied across, so fusion
        behaviour is bit-identical and only the timing instrumentation differs.
        """
        self._dense = _TimedIndex(hybrid.dense) if hybrid.dense is not None else None
        self._sparse = _TimedIndex(hybrid.sparse) if hybrid.sparse is not None else None
        instrumented = HybridIndex(
            self._dense,  # type: ignore[arg-type]
            self._sparse,  # type: ignore[arg-type]
            fusion=hybrid.fusion,
            rrf_k=hybrid.rrf_k,
            dense_weight=hybrid.weights["dense"],
            sparse_weight=hybrid.weights["sparse"],
            candidate_multiplier=hybrid.candidate_multiplier,
        )
        return instrumented

    @property
    def ready(self) -> bool:
        """True when an index and a chunk store are loaded."""
        return self._search_index is not None and self.store is not None

    @property
    def n_chunks(self) -> int:
        """Number of indexed chunks, ``0`` when no index is loaded."""
        return 0 if self.store is None else len(self.store)

    def health(self) -> list[dict[str, Any]]:
        """Circuit-breaker state per generation provider."""
        return [] if self.router is None else self.router.health()

    async def aclose(self) -> None:
        """Release provider transports and the retrieval thread pools."""
        if self.router is not None:
            await self.router.aclose()
        if self._search_index is not None:
            self._search_index.close()
        if self.hybrid is not None:
            self.hybrid.close()

    # -- stages ---------------------------------------------------------------

    def _embed(self, question: str) -> np.ndarray:
        """Encode one query. Blocking; the caller decides which thread runs it."""
        if self.embedder is None:
            raise PipelineNotReady("no embedder configured")
        return np.asarray(self.embedder.encode([question])[0])

    async def retrieve(
        self, question: str, *, trace: Trace | None = None, k: int | None = None
    ) -> RetrievalResult:
        """Embed and retrieve, emitting the embed/retrieve spans.

        Shared by the answer path and by ``/speculate``, so a speculation is
        produced by exactly the same code as a real retrieval and a cache hit
        cannot differ from a miss in anything but timing.

        Args:
            question: Transcript to retrieve for.
            trace: Trace to instrument; spans are skipped when ``None``.
            k: Override the configured ``retrieval_k``.

        Returns:
            A :class:`RetrievalResult` carrying the hits, the query vector and
            the measured cost of each half.

        Raises:
            PipelineNotReady: No index is loaded.
        """
        if self._search_index is None:
            raise PipelineNotReady("no index loaded")
        top_k = self.settings.retrieval_k if k is None else k

        loop = asyncio.get_running_loop()
        t0 = time.perf_counter_ns()
        if trace is not None:
            with trace.span("embed", model=self.settings.embedder_spec):
                vec = await loop.run_in_executor(None, self._embed, question)
        else:
            vec = await loop.run_in_executor(None, self._embed, question)
        t1 = time.perf_counter_ns()

        hits = await self._search_index.asearch(
            query_text=question,
            query_vec=vec,
            k=top_k,
            ef_search=self.settings.ef_search,
        )
        t2 = time.perf_counter_ns()

        dense_span = self._dense.span if self._dense is not None else None
        sparse_span = self._sparse.span if self._sparse is not None else None
        if trace is not None:
            self._record_retrieval_spans(trace, t1, t2, dense_span, sparse_span, len(hits))

        return RetrievalResult(
            hits=hits,
            query_vec=vec,
            embed_ms=(t1 - t0) / 1e6,
            retrieve_ms=(t2 - t1) / 1e6,
            dense_span=dense_span,
            sparse_span=sparse_span,
        )

    @staticmethod
    def _record_retrieval_spans(
        trace: Trace,
        start_ns: int,
        end_ns: int,
        dense: tuple[int, int] | None,
        sparse: tuple[int, int] | None,
        n_hits: int,
    ) -> None:
        """Turn recorded intervals into trace spans, on the calling thread.

        ``fuse`` is the interval between the last retrieval leg finishing and
        the fused result being returned, so it genuinely covers rank fusion plus
        the executor join rather than being a residual computed by subtraction.
        """
        legs = [ns for ns in (dense, sparse) if ns is not None]
        for name, interval in (("retrieve.dense", dense), ("retrieve.sparse", sparse)):
            if interval is None:
                continue
            trace.spans.append(Span(name=name, start_ns=interval[0], end_ns=interval[1]))
        fuse_start = max((ns[1] for ns in legs), default=start_ns)
        trace.spans.append(
            Span(
                name="fuse",
                start_ns=min(fuse_start, end_ns),
                end_ns=end_ns,
                attrs={"n_hits": n_hits},
            )
        )

    def _contexts(self, hits: Sequence[RetrievalHit]) -> tuple[list[Chunk], list[str]]:
        """Resolve hits to chunks and to the exact strings sent to the model.

        Returns:
            ``(chunks, prompt_texts)`` where ``prompt_texts[n - 1]`` is what
            citation ``[n]`` refers to. Both lists are filtered and clipped by
            :func:`~voicerag.generate.prompt.usable_passages`, so citation index
            arithmetic is done once, here, instead of in three callers.
        """
        if self.store is None:
            return [], []
        chunks = self.store.get_many([h.chunk_id for h in hits])
        selected = usable_passages(chunks)
        return [chunks[i] for i, _ in selected], [text for _, text in selected]

    # -- public entry points --------------------------------------------------

    async def answer(
        self,
        question: str,
        *,
        trace: Trace | None = None,
        deadline: Deadline | None = None,
        speculative_hits: RetrievalResult | Sequence[RetrievalHit] | None = None,
    ) -> RagResponse:
        """Answer a question, collected rather than streamed.

        Args:
            question: The transcript, exactly as STT delivered it.
            trace: Trace to record into. A fresh one is created when omitted.
            deadline: Shared budget. ``Deadline(settings.budget_total_ms)`` when
                omitted.
            speculative_hits: A retrieval already performed for this question,
                from ``/speculate``. Accepts a :class:`RetrievalResult` or a
                bare hit list.

        Returns:
            The complete :class:`RagResponse`.
        """
        response: RagResponse | None = None
        async for event in self.astream(
            question, trace=trace, deadline=deadline, speculative_hits=speculative_hits
        ):
            if event.kind == "final":
                response = event.response
        assert response is not None, "astream must always terminate with a final event"
        return response

    async def astream(
        self,
        question: str,
        *,
        trace: Trace | None = None,
        deadline: Deadline | None = None,
        speculative_hits: RetrievalResult | Sequence[RetrievalHit] | None = None,
    ) -> AsyncIterator[RagEvent]:
        """Run the pipeline, yielding tokens and then exactly one final event.

        The generator is guaranteed to terminate with a ``final`` event on every
        path -- abstention, provider failure, deadline exhaustion -- so a caller
        never has to distinguish "stream ended" from "stream ended badly".

        Args: See :meth:`answer`.

        Yields:
            :class:`RagEvent` values: zero or more ``token`` events, then one
            ``final``.
        """
        trace = trace or Trace()
        deadline = deadline or Deadline(self.settings.budget_total_ms)
        budgets = self.settings.stage_budgets()
        spec_block = SpeculationBlock() if self.settings.enable_speculative_retrieval else None

        with traced(trace):
            # 1. input guard ---------------------------------------------------
            input_verdict = None
            if self.settings.enable_guardrails:
                with trace.span("guard.input"):
                    input_verdict = self.policy.check_input(question)
                if not input_verdict.allowed:
                    report = self.policy.report(question, input_verdict=input_verdict)
                    yield _final(
                        answer=report.abstain_reason or "",
                        citations=[],
                        report=report,
                        provider="none",
                        trace=trace,
                        speculation=spec_block,
                    )
                    return

            if not self.ready:
                raise PipelineNotReady("no index loaded; run scripts/ingest.py")

            # 2. retrieval (or a speculation someone else already paid for) ----
            retrieval = _coerce_retrieval(speculative_hits)
            if retrieval is None and self.speculation is not None:
                retrieval = self.speculation.take(question)
            if retrieval is not None:
                # No embed/retrieve span is emitted, because for *this* request
                # neither stage ran. Replaying the speculation's spans would put
                # work in the breakdown that the user never waited for, which is
                # exactly the kind of flattering accounting the trace exists to
                # prevent. The saving is reported separately, as a saving.
                if spec_block is not None:
                    spec_block.hit = True
                    spec_block.launched = 1
                    spec_block.saved_ms = round(retrieval.total_ms, 3)
                trace.mark("speculation.hit", saved_ms=round(retrieval.total_ms, 3))
            else:
                allowance = budgets["embed"] + budgets["retrieve"]
                limit = _retrieval_limit_ms(deadline, budgets)
                started = time.perf_counter_ns()
                try:
                    retrieval = await asyncio.wait_for(
                        self.retrieve(question, trace=trace), timeout=limit / 1000.0
                    )
                except asyncio.TimeoutError:
                    trace.mark("retrieve.timeout", limit_ms=round(limit, 3))
                    retrieval = None
                else:
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    if elapsed > allowance:
                        # The stage finished, but over its allowance. Recorded so
                        # a slow request names its own culprit instead of showing
                        # up as an unexplained total.
                        trace.mark(
                            "retrieve.over_budget",
                            allowance_ms=round(allowance, 3),
                            actual_ms=round(elapsed, 3),
                        )
                if retrieval is None:
                    # Not "the corpus has nothing" -- we never finished looking.
                    # Saying so is the difference between an honest refusal and a
                    # wrong one, and it points at the budget, which is the fix.
                    report = self.policy.report(question, input_verdict=input_verdict)
                    report.abstained = True
                    report.abstain_reason = (
                        f"Retrieval did not finish within its {limit:.0f} ms slice of the "
                        f"{self.settings.budget_total_ms:.0f} ms budget, so there is nothing "
                        "to ground an answer on. Raise BUDGET_TOTAL_MS or use a faster "
                        "embedder than " + self.settings.embedder_spec + "."
                    )
                    yield _final(
                        answer=report.abstain_reason,
                        citations=[],
                        report=report,
                        provider="none",
                        trace=trace,
                        speculation=spec_block,
                    )
                    return
            hits = retrieval.hits

            # 3. abstention ----------------------------------------------------
            abstention = None
            if self.settings.enable_guardrails:
                with trace.span("guard.abstention", n_hits=len(hits)):
                    abstention = self.policy.check_retrieval(hits)
                if abstention.should_abstain:
                    # The generator is never constructed, never called, never
                    # awaited. That is the point of the gate.
                    report = self.policy.report(
                        question,
                        input_verdict=input_verdict,
                        abstention_verdict=abstention,
                    )
                    yield _final(
                        answer=report.abstain_reason or "",
                        citations=_citations(self._contexts(hits)[0], hits, ()),
                        report=report,
                        provider="none",
                        trace=trace,
                        speculation=spec_block,
                    )
                    return

            # 4. prompt --------------------------------------------------------
            with trace.span("prompt", k=len(hits)):
                chunks, passages = self._contexts(hits)
                system, user = render(question, passages)

            # 5. generation, with grounding riding alongside -------------------
            if self.router is None:
                report = self.policy.report(
                    question, input_verdict=input_verdict, abstention_verdict=abstention
                )
                report.abstained = True
                report.abstain_reason = (
                    "No generation provider is configured, so there is nothing to "
                    "ground an answer with. Set GROQ_API_KEY or GEMINI_API_KEY."
                )
                yield _final(
                    answer=report.abstain_reason,
                    citations=_citations(chunks, hits, ()),
                    report=report,
                    provider="none",
                    trace=trace,
                    speculation=spec_block,
                )
                return

            meta: dict[str, Any] = {}
            grounder = (
                _StreamingGrounder(self.policy.grounding, passages)
                if self.settings.enable_grounding and passages
                else None
            )
            parts: list[str] = []
            truncated = False
            failure: str | None = None
            gen_budget = deadline.slice_ms(budgets["generate"])

            with trace.span("generate", provider="router"):
                try:
                    async for delta in self.router.stream(
                        system,
                        user,
                        meta=meta,
                        max_tokens=self.settings.max_tokens,
                    ):
                        parts.append(delta)
                        if grounder is not None:
                            grounder.feed("".join(parts))
                        yield RagEvent(kind="token", delta=delta)
                        if deadline.remaining_ms <= budgets["guard.grounding"]:
                            # Stop consuming rather than hang past the budget.
                            # What has already been streamed stays streamed; the
                            # answer is finalised from it and flagged.
                            truncated = True
                            trace.mark("generate.truncated", reason="deadline")
                            break
                except (AllProvidersFailed, BudgetExhausted) as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    trace.mark("generate.failed", error=failure)
                except asyncio.TimeoutError:
                    failure = "generation timed out"
                    trace.mark("generate.failed", error=failure)

            raw = "".join(parts)
            provider = str(meta.get("provider", "none")) if raw else "none"
            trace.meta["generate_budget_ms"] = round(gen_budget, 3)
            if truncated:
                trace.meta["truncated"] = True

            if not raw:
                report = self.policy.report(
                    question, input_verdict=input_verdict, abstention_verdict=abstention
                )
                report.abstained = True
                report.abstain_reason = (
                    "Every generation provider failed, so no answer was produced."
                    if failure
                    else "The generator returned nothing."
                )
                yield _final(
                    answer=report.abstain_reason,
                    citations=_citations(chunks, hits, ()),
                    report=report,
                    provider="none",
                    trace=trace,
                    speculation=spec_block,
                )
                return

            parsed = parse_answer(raw)

            # 6. grounding -----------------------------------------------------
            grounding = None
            if grounder is not None and not parsed.abstained:
                with trace.span("guard.grounding") as span:
                    grounding = grounder.finish(parsed.text)
                    span.attrs["claims"] = len(grounding.per_claim)

            report = self.policy.report(
                question,
                input_verdict=input_verdict,
                abstention_verdict=abstention,
                grounding_verdict=grounding,
            )
            if parsed.abstained:
                # The model emitted the abstention token even though the gate
                # let the request through. Trusting it is the conservative
                # choice: it saw the passages, the gate only saw their scores.
                report.abstained = True
                report.abstain_reason = (
                    "The model read the retrieved passages and judged them "
                    "insufficient, so it declined to answer rather than guess."
                )

            answer_text = parsed.text
            if report.abstained:
                answer_text = report.abstain_reason or ""
            elif truncated:
                answer_text = f"{parsed.text} [truncated at the latency budget]"

            yield _final(
                answer=answer_text,
                citations=_citations(chunks, hits, parsed.citations),
                report=report,
                provider=provider,
                trace=trace,
                speculation=spec_block,
            )

    # -- speculation ----------------------------------------------------------

    async def speculate(self, partial: str) -> RetrievalResult | None:
        """Retrieve for a partial transcript and cache it for the real request.

        Args:
            partial: A partial transcript from the browser.

        Returns:
            The stored :class:`RetrievalResult`, or ``None`` when speculation is
            disabled, no cache is attached, or the pipeline has no index. Never
            raises: a failed speculation is a missed optimisation, not a broken
            request, and the caller is a fire-and-forget client.
        """
        if not self.settings.enable_speculative_retrieval or self.speculation is None:
            return None
        if not self.ready:
            return None
        try:
            result = await self.retrieve(partial)
        except Exception:  # noqa: BLE001 - speculation must never surface an error
            return None
        self.speculation.put(partial, result)
        return result

    async def warmup(self, query: str | None = None) -> dict[str, float]:
        """Run one throwaway request through every stage that has state.

        Mandatory before serving. The first request through a fresh process pays
        FAISS visit-table allocation, the bm25s score matrix's first touch, the
        embedder's matrix paging in and CPython's call caches -- measured at
        tens of milliseconds on this hardware. A judge's first question must not
        be the request that pays them.

        Args:
            query: Warmup text; ``settings.warmup_query`` when omitted.

        Returns:
            Measured milliseconds per warmed stage. Empty when no index is
            loaded, so a server without an index still starts.
        """
        text = query or self.settings.warmup_query
        timings: dict[str, float] = {}
        if self.settings.enable_guardrails:
            t0 = time.perf_counter_ns()
            self.policy.check_input(text)
            timings["guard.input"] = (time.perf_counter_ns() - t0) / 1e6
        if not self.ready:
            return timings
        t0 = time.perf_counter_ns()
        result = await self.retrieve(text)
        timings["retrieve"] = (time.perf_counter_ns() - t0) / 1e6
        timings["embed"] = round(result.embed_ms, 3)
        if self.settings.enable_guardrails:
            t0 = time.perf_counter_ns()
            self.policy.check_retrieval(result.hits)
            timings["guard.abstention"] = (time.perf_counter_ns() - t0) / 1e6
        chunks, passages = self._contexts(result.hits)
        t0 = time.perf_counter_ns()
        render(text, passages)
        timings["prompt"] = (time.perf_counter_ns() - t0) / 1e6
        if self.settings.enable_grounding and passages:
            t0 = time.perf_counter_ns()
            self.policy.check_answer("Warmup sentence with a citation [1].", passages)
            timings["guard.grounding"] = (time.perf_counter_ns() - t0) / 1e6
        del chunks
        return {k: round(v, 3) for k, v in timings.items()}


# --- helpers ------------------------------------------------------------------


def _as_router(
    generator: GenerationRouter | Generator | Sequence[Generator] | None,
) -> GenerationRouter | None:
    """Normalise the many shapes a caller may supply into one router."""
    if generator is None:
        return None
    if isinstance(generator, GenerationRouter):
        return generator
    if isinstance(generator, Generator):
        return GenerationRouter([generator])
    providers = list(generator)
    return GenerationRouter(providers) if providers else None


def _coerce_retrieval(
    value: RetrievalResult | Sequence[RetrievalHit] | None,
) -> RetrievalResult | None:
    """Accept either a full result or a bare hit list from the caller."""
    if value is None:
        return None
    if isinstance(value, RetrievalResult):
        return value
    hits = list(value)
    return RetrievalResult(hits=hits) if hits else None


def _citations(
    chunks: Sequence[Chunk],
    hits: Sequence[RetrievalHit],
    cited: Iterable[int],
) -> list[Citation]:
    """Build the evidence list the UI renders.

    When the model emitted valid ``[n]`` markers the result is exactly those
    passages, in citation order -- that is what "citations" means. When it cited
    nothing (or cited out of range, which the grounding layer has already
    rejected) the full retrieved context is returned instead, so the panel shows
    what the answer was judged against rather than going blank.

    Args:
        chunks: Chunks corresponding to the prompt passages, in prompt order.
        hits: The fused hits, used only for scores.
        cited: 1-based citation indices parsed from the answer.
    """
    if not chunks:
        return []
    by_id = {h.chunk_id: h for h in hits}
    order = [n - 1 for n in dict.fromkeys(cited) if 1 <= n <= len(chunks)]
    if not order:
        order = list(range(len(chunks)))
    out: list[Citation] = []
    for i in order:
        chunk = chunks[i]
        hit = by_id.get(chunk.chunk_id)
        out.append(
            Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=round(float(hit.score) if hit is not None else 0.0, 6),
                title=chunk.title or None,
            )
        )
    return out


def _final(
    *,
    answer: str,
    citations: list[Citation],
    report: GuardrailReport,
    provider: str,
    trace: Trace,
    speculation: SpeculationBlock | None,
) -> RagEvent:
    """Assemble the single terminating event. One construction site, so the
    response shape cannot vary between exit paths."""
    return RagEvent(
        kind="final",
        response=RagResponse(
            answer=answer,
            citations=citations,
            abstained=report.abstained,
            abstain_reason=report.abstain_reason,
            guardrails=GuardrailBlock.from_report(report),
            provider=provider,
            trace=TraceBlock(
                trace_id=trace.trace_id,
                total_ms=round(trace.total_ms, 3),
                critical_path_ms=round(trace.critical_path_ms, 3),
                breakdown=trace.breakdown(),
            ),
            speculation=speculation,
        ),
    )


def build_policy(settings: Settings | None = None) -> GuardrailPolicy:
    """Construct the guardrail policy from configuration.

    A policy file written by
    :meth:`~voicerag.guardrails.policy.GuardrailPolicy.save` wins outright when
    ``guardrail_policy_file`` points at one: that file carries the *calibrated*
    abstention model, and serving the uncalibrated priors when a fitted gate
    exists would halve recall on unanswerable questions.

    Args:
        settings: Configuration; :func:`~voicerag.config.get_settings` when
            omitted.

    Returns:
        A configured :class:`~voicerag.guardrails.policy.GuardrailPolicy`.
    """
    from .guardrails.abstention import AbstentionGate
    from .guardrails.grounding import GroundingChecker, GroundingConfig
    from .guardrails.input_guard import InputGuard, InputGuardConfig

    cfg = settings or get_settings()
    if cfg.guardrail_policy_file is not None and cfg.guardrail_policy_file.exists():
        return GuardrailPolicy.load(cfg.guardrail_policy_file)
    return GuardrailPolicy(
        InputGuard(
            InputGuardConfig(
                max_chars=cfg.max_query_chars,
                injection_threshold=cfg.injection_threshold,
                profanity_threshold=cfg.profanity_threshold,
            )
        ),
        AbstentionGate(decision_threshold=cfg.abstain_threshold, k=cfg.abstain_k),
        GroundingChecker(
            GroundingConfig(
                claim_threshold=cfg.grounding_claim_threshold,
                answer_threshold=cfg.grounding_answer_threshold,
            )
        ),
        answer_on_ungrounded=cfg.answer_on_ungrounded,
    )


def _retrieval_limit_ms(deadline: Deadline, budgets: Mapping[str, float]) -> float:
    """How long retrieval may actually take before it is abandoned.

    The per-stage allowance is a *target*, not the cut-off. Cutting retrieval at
    its allowance while 140 ms of budget sat unused would throw away a request
    that could still have been answered on time -- and worse, it would report
    the corpus as empty. The real constraint is that enough budget must survive
    for the stages that still have to run, so the limit is whatever remains
    after reserving those, floored at the allowance so a nearly-spent budget
    still gives retrieval its fair slice rather than one millisecond.

    Args:
        deadline: The request's shared budget.
        budgets: Per-stage allowances from
            :meth:`~voicerag.config.Settings.stage_budgets`.

    Returns:
        A timeout in milliseconds, always at least 1 ms.
    """
    reserve = budgets["prompt"] + budgets["generate"] + budgets["guard.grounding"]
    allowance = budgets["embed"] + budgets["retrieve"]
    return max(1.0, allowance, deadline.remaining_ms - reserve)


def stage_deadline(deadline: Deadline, budgets: Mapping[str, float], stage: str) -> float:
    """Milliseconds this stage may use: its allowance, capped by what remains.

    Exposed for callers that drive stages individually (the benchmark, and any
    future partial pipeline) so the slicing rule lives in one place.
    """
    return deadline.slice_ms(budgets.get(stage, deadline.remaining_ms))
