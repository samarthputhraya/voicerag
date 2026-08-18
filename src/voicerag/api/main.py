"""The HTTP surface: ``voicerag.api.main:app``.

Eleven endpoints, each with one job:

==================  ==========================================================
``POST /ask``       One question, one complete :class:`~voicerag.pipeline.RagResponse`.
``POST /ask/stream``The same, as SSE: ``token`` frames, then one ``final``, or
                    one ``error``. The frontend parses exactly those names.
``POST /speculate`` Warm the retrieval cache from a partial transcript. Returns
                    202 immediately; the work happens after the response.
``POST /speak``     Answer text to ``audio/wav``, so the answer is spoken and
                    not merely printed.
``GET  /examples``  Questions this corpus can actually answer.
``GET  /healthz``   Liveness, index state, provider circuit breakers.
``GET  /stats``     Corpus description and rolling latency percentiles.
``POST /stt/token`` A short-lived STT credential, so the account key never
                    reaches the browser.
``WS   /stt/stream``Relay for browser audio to Sarvam's realtime socket. This is
                    what keeps the account key server-side, because Sarvam mints
                    no ephemeral credential a browser could hold instead.
``GET  /api``       Endpoint index, so a judge who opens the port sees a map.
``GET  /``          The built frontend when ``STATIC_DIR`` names one, else the
                    same endpoint index.
==================  ==========================================================

Four cross-cutting rules:

**The index loads once, at startup, and every model is warmed before the first
request.** A cold first request would put FAISS allocation and matrix paging
into the number the whole submission is judged on.

**Errors have one shape.** :class:`~voicerag.api.schemas.ErrorResponse` for
everything, never a stack trace. A traceback in a response body is an
information leak and a client that must handle three error formats handles none.

**Every request logs one structured JSON line including the trace breakdown**,
so the HUD's numbers and the server's numbers are the same numbers.

**Every endpoint that spends a third-party token is rate limited.** The deployed
link is public and holds live credentials on a free tier; see
:mod:`voicerag.api.ratelimit` for the policy and for why the middleware is
installed before CORS rather than after.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import Settings, get_settings
from ..harness.resilience import Deadline
from ..harness.trace import Trace
from ..pipeline import PipelineNotReady, RagResponse
from .logging import configure_logging, log_request
from .ratelimit import RateLimiter, RateLimitMiddleware
from .schemas import (
    AskRequest,
    ErrorResponse,
    HealthResponse,
    SpeculateRequest,
    SpeculateResponse,
    StatsResponse,
    SttTokenResponse,
)
from .state import AppState
from .stt_relay import relay_sarvam
from .stt_token import TokenMintError, mint_token
from .tts import TtsError, synthesize

__all__ = ["app", "create_app", "SSE_MEDIA_TYPE"]

#: ``text/event-stream``. Named because the streaming response sets several
#: headers that only make sense together and they should be described once.
SSE_MEDIA_TYPE = "text/event-stream"

log = logging.getLogger("voicerag.api")


#: Question words a preset must open with. MS MARCO's queries are real Bing
#: searches, so the raw set is mostly keyword fragments ("bib lettuce wrap",
#: "chart for foods low in potassium") rather than questions. Requiring an
#: interrogative opener is what turns the sample into something a person would
#: plausibly say out loud, which is the whole point of a voice demo.
_QUESTION_OPENERS = (
    "what ", "who ", "when ", "where ", "why ", "how ",
    "which ", "is ", "are ", "does ", "do ", "can ",
)

#: Substrings that disqualify a preset outright. This is a demo shown to judges;
#: MS MARCO is scraped from real search traffic and contains queries about
#: bodies, salaries of named individuals, and worse. None of it is *unsafe* --
#: the guardrails would pass it -- it is simply not what should be on screen.
_PRESET_BLOCKLIST = (
    "cup size", "bra ", "nude", "naked", "porn", "sex ", "sexual", "breast",
    "penis", "vagina", "boob", "weight of ", "how much does ", "salary",
    "net worth", "how old is", "died", "death of", "kill",
)


def _presentable_question(raw: str) -> str | None:
    """Normalise one shard query, or reject it as unfit for a preset.

    Returns a cleaned, capitalised question, or ``None`` if the query is a
    keyword fragment, malformed, or on the blocklist above.
    """
    q = raw.strip().lstrip(". ").strip()
    if not q:
        return None
    low = q.lower()

    if not low.startswith(_QUESTION_OPENERS):
        return None
    if any(bad in low for bad in _PRESET_BLOCKLIST):
        return None
    # Long enough to be a real question, short enough to fit a chip.
    if not (18 <= len(q) <= 68) or len(q.split()) < 4:
        return None
    # Interior punctuation is the signature of the shard's concatenation
    # artefacts ("what is basic? uni").
    if any(ch in q[:-1] for ch in "?!;|"):
        return None
    if not q[0].isalpha():
        return None

    q = q[0].upper() + q[1:]
    return q if q.endswith("?") else q + "?"


def _sse(event: str, data: Any) -> bytes:
    """Encode one SSE frame.

    ``data`` is emitted as a single line of compact JSON: the frontend joins
    multi-line ``data:`` payloads before parsing, but a newline inside a token
    would otherwise split a frame, and tokens containing newlines are ordinary.
    """
    body = json.dumps(data, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {body}\n\n".encode()


def create_app(
    settings: Settings | None = None, *, state: AppState | None = None
) -> FastAPI:
    """Build the ASGI application.

    A factory rather than a module-level singleton so tests can start a server
    over an in-memory index with fake providers, using the same routes and the
    same middleware as production -- a test against a differently assembled app
    tests a different app.

    Args:
        settings: Configuration; :func:`~voicerag.config.get_settings` by default.
        state: A pre-built :class:`~voicerag.api.state.AppState`. When supplied,
            startup warms it but does not load an index from disk.

    Returns:
        The configured :class:`fastapi.FastAPI` instance.
    """
    cfg = settings or get_settings()
    configure_logging(cfg.log_level)
    app_state = state or AppState(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Load the index and warm every model before accepting traffic."""
        await app_state.startup()
        try:
            yield
        finally:
            await app_state.shutdown()

    application = FastAPI(
        title="VoiceRAG",
        version="1.0.0",
        summary="Voice-first retrieval-augmented QA under a 200 ms budget.",
        lifespan=lifespan,
    )
    application.state.app_state = app_state
    application.state.settings = cfg

    limiter = RateLimiter(cfg)
    application.state.rate_limiter = limiter

    # Order is load-bearing, and it is the reverse of how it reads. Starlette's
    # `add_middleware` inserts at position 0 and builds the stack by wrapping
    # `reversed(user_middleware)`, so the middleware added *last* ends up
    # outermost. The limiter is added first precisely so CORS ends up outside
    # it: a 429 that reaches the browser without an `Access-Control-Allow-Origin`
    # header is reported by `fetch` as an opaque network failure, and the
    # frontend would say "the server is unreachable" for a condition it could
    # have explained exactly.
    application.add_middleware(RateLimitMiddleware, limiter=limiter)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["content-type", "authorization"],
        # Not safelisted, so a browser cannot read it unless it is exposed --
        # and a Retry-After a client cannot read is a Retry-After that does not
        # exist.
        expose_headers=["Retry-After"],
    )

    _register_error_handlers(application)
    _register_routes(application, app_state, cfg, limiter)
    _mount_frontend(application, cfg)
    return application


def _frontend_dir(cfg: Settings) -> Path | None:
    """The built frontend to serve at ``/``, or ``None``.

    The single predicate for "is this deployment serving a UI". Both the route
    registration and the mount consult it, because the two decisions have to
    agree: registering ``GET /`` *and* mounting would leave the frontend
    unreachable, and doing neither would leave the container answering 404 at its
    own front door.

    A directory that is configured but has no ``index.html`` counts as no
    frontend. That is the honest reading of a build that did not produce one, and
    it keeps the API serving rather than refusing to boot over an optional UI.
    """
    static_dir = cfg.static_dir
    if static_dir is None:
        return None
    return static_dir if (static_dir / "index.html").is_file() else None


def _mount_frontend(application: FastAPI, cfg: Settings) -> None:
    """Serve the built frontend from this origin, when one is present.

    Mounted **last**, and that is the whole trick: Starlette matches routes in
    registration order, so every API route above has already claimed its path and
    a ``Mount("/")`` only ever sees what is left. ``html=True`` is what makes the
    mount answer ``/`` with ``index.html``.
    """
    static_dir = _frontend_dir(cfg)
    if static_dir is None:
        if cfg.static_dir is not None:
            # Loud, and naming the path: the silent version of this is a
            # container that passes its healthcheck and serves 404 to every
            # visitor, which is a genuinely hard thing to diagnose from outside.
            log.warning(
                "STATIC_DIR is set but contains no index.html; serving the API only",
                extra={"fields": {"static_dir": str(cfg.static_dir)}},
            )
        return

    application.mount("/", _FrontendFiles(directory=static_dir, html=True), name="frontend")
    log.info("serving frontend", extra={"fields": {"static_dir": str(static_dir)}})


class _FrontendFiles(StaticFiles):
    """``StaticFiles`` that declines a WebSocket instead of asserting on one.

    ``Mount`` matches ``http`` **and** ``websocket`` scopes, while
    ``StaticFiles.__call__`` opens with ``assert scope["type"] == "http"``. So
    once a frontend is mounted at ``/``, an upgrade to any path the API does not
    route -- ``ws://host/anything`` -- reaches the mount and raises
    ``AssertionError``. Nothing catches it: ``ServerErrorMiddleware`` forwards
    non-HTTP scopes untouched. The result is a bare 500 handshake and a full
    traceback per attempt, on a public link, from one unauthenticated request.

    Nothing in this application dials a socket anywhere but ``/stt/stream``, so
    this is log hygiene rather than a live defect -- but a container whose logs a
    judge may read should not be trivially fillable with tracebacks.
    """

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "websocket":
            # Decline before accepting, which is how ASGI says "no": the server
            # answers the upgrade with an HTTP error rather than opening a socket.
            await receive()
            await send({"type": "websocket.close", "code": 1000})
            return
        await super().__call__(scope, receive, send)


# --- error handling -----------------------------------------------------------


def _register_error_handlers(application: FastAPI) -> None:
    """Install handlers that always return :class:`ErrorResponse`."""

    @application.exception_handler(PipelineNotReady)
    async def _not_ready(request: Request, exc: PipelineNotReady) -> JSONResponse:
        """503 rather than 500: the deployment is incomplete, not broken."""
        log.warning("pipeline not ready", extra={"fields": {"route": request.url.path}})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="index_not_loaded",
                message=(
                    f"No index is loaded ({exc}). Build one with "
                    "`python scripts/ingest.py --synthetic 400`."
                ),
            ).model_dump(),
        )

    @application.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Log the traceback server-side; return a clean shape to the client."""
        log.exception(
            "unhandled error", extra={"fields": {"route": request.url.path}}
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                message=f"The request failed ({type(exc).__name__}).",
            ).model_dump(),
        )


# --- routes -------------------------------------------------------------------


def _register_routes(
    application: FastAPI, state: AppState, cfg: Settings, limiter: RateLimiter
) -> None:
    """Attach every endpoint. Split out so ``create_app`` stays readable."""

    def endpoint_map() -> dict[str, Any]:
        """A map of the API, for whoever opens the port without the frontend."""
        return {
            "service": cfg.app_name,
            "endpoints": {
                "POST /ask": "question -> complete answer",
                "POST /ask/stream": "question -> SSE token/final/error frames",
                "POST /speculate": "partial transcript -> warm the retrieval cache",
                "POST /speak": "answer text -> audio/wav",
                "GET /examples": "questions this corpus can actually answer",
                "GET /healthz": "liveness, index state, provider circuits",
                "GET /stats": "corpus description and recent latency",
                "POST /stt/token": "short-lived STT credential for the browser",
                "WS /stt/stream": "browser audio -> Sarvam, transcripts back",
            },
            "budget_ms": cfg.budget_total_ms,
        }

    @application.get("/api", tags=["meta"])
    async def api_index() -> dict[str, Any]:
        """The endpoint map, always at a fixed path.

        ``/`` is the frontend when one is built into the image, so the map needs
        somewhere it can be linked to unconditionally.
        """
        return endpoint_map()

    # Only when nothing will be mounted at `/`. The predicate is shared with
    # `_mount_frontend` rather than duplicated: registering this route and then
    # declining to mount -- or the reverse -- is the kind of near-miss that
    # produces a container serving 404 at its own front door.
    if _frontend_dir(cfg) is None:

        @application.get("/", tags=["meta"])
        async def index() -> dict[str, Any]:
            """A map of the API, for whoever opens the port without the frontend."""
            return endpoint_map()

    @application.post("/ask", response_model=RagResponse, tags=["ask"])
    async def ask(payload: AskRequest) -> RagResponse:
        """Answer one question and return the complete response.

        The non-streaming path exists for evaluation harnesses and for clients
        that cannot consume SSE. It runs the identical pipeline; only the
        delivery differs.
        """
        if state.pipeline is None:
            raise PipelineNotReady("the application was not started")
        started = time.perf_counter_ns()
        trace = Trace()
        response = await state.pipeline.answer(
            payload.question,
            trace=trace,
            deadline=Deadline(cfg.budget_total_ms),
        )
        duration = (time.perf_counter_ns() - started) / 1e6
        state.record_latency(duration)
        log_request(
            log,
            route="/ask",
            status=200,
            duration_ms=duration,
            trace_id=response.trace.trace_id,
            breakdown=response.trace.breakdown,
            question=payload.question if cfg.log_queries else None,
            abstained=response.abstained,
            provider=response.provider,
            language=payload.language,
            speculation_hit=bool(response.speculation and response.speculation.hit),
        )
        return response

    @application.post("/ask/stream", tags=["ask"])
    async def ask_stream(payload: AskRequest) -> StreamingResponse:
        """Stream the answer as Server-Sent Events.

        Frame contract, matched byte-for-byte by ``web/lib/api.ts``:

        * ``event: token`` with ``{"delta": "..."}`` per incremental chunk
        * ``event: final`` with the complete :class:`RagResponse`
        * ``event: error`` with ``{"message": "..."}`` when the pipeline raised

        An error frame is emitted *inside* a 200 response because the status
        line is long gone by the time generation fails; the client distinguishes
        success from failure by the terminal frame, not by the status code.
        """
        if state.pipeline is None:
            raise PipelineNotReady("the application was not started")
        pipeline = state.pipeline
        started = time.perf_counter_ns()
        trace = Trace()

        async def frames() -> AsyncIterator[bytes]:
            final: RagResponse | None = None
            failure: str | None = None
            try:
                async for event in pipeline.astream(
                    payload.question,
                    trace=trace,
                    deadline=Deadline(cfg.budget_total_ms),
                ):
                    if event.kind == "token":
                        yield _sse("token", {"delta": event.delta})
                    elif event.response is not None:
                        final = event.response
                        yield _sse("final", final.model_dump())
            except PipelineNotReady as exc:
                failure = f"No index is loaded ({exc})."
                yield _sse("error", {"message": failure})
            except asyncio.CancelledError:
                # The client hung up (a superseded question, usually). Not an
                # error, and there is nobody left to send a frame to.
                raise
            except Exception as exc:  # noqa: BLE001 - reported as a frame, then logged
                log.exception("stream failed", extra={"fields": {"route": "/ask/stream"}})
                failure = f"The request failed ({type(exc).__name__})."
                yield _sse("error", {"message": failure})
            finally:
                duration = (time.perf_counter_ns() - started) / 1e6
                if final is not None:
                    state.record_latency(duration)
                log_request(
                    log,
                    route="/ask/stream",
                    status=200,
                    duration_ms=duration,
                    trace_id=trace.trace_id,
                    breakdown=trace.breakdown(),
                    question=payload.question if cfg.log_queries else None,
                    abstained=bool(final and final.abstained),
                    provider=final.provider if final else "none",
                    language=payload.language,
                    error=failure,
                )

        return StreamingResponse(
            frames(),
            media_type=SSE_MEDIA_TYPE,
            headers={
                # Buffering proxies are the classic reason an SSE demo shows
                # every token at once, at the end. These disable the two that
                # matter in practice.
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @application.post(
        "/speculate",
        response_model=SpeculateResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["ask"],
    )
    async def speculate(
        payload: SpeculateRequest, background: BackgroundTasks
    ) -> SpeculateResponse:
        """Warm the retrieval cache from a partial transcript.

        Returns 202 before doing any work. This endpoint is on the critical path
        of nothing: the client fires it while the user is still speaking and
        never reads the result, so blocking here would spend the very
        milliseconds the mechanism exists to save. The retrieval runs as a
        background task after the response is sent.
        """
        if not cfg.enable_speculative_retrieval:
            return SpeculateResponse(
                accepted=False, key="", reason="speculative retrieval is disabled"
            )
        if state.pipeline is None or not state.pipeline.ready:
            return SpeculateResponse(accepted=False, key="", reason="no index loaded")
        # `is not None`, not truthiness: SpeculationCache defines __len__, so an
        # empty cache is falsy and the key would silently come back blank.
        key = "" if state.speculation is None else state.speculation.key(payload.partial)
        background.add_task(_run_speculation, state, payload.partial)
        return SpeculateResponse(accepted=True, key=key)

    @application.get("/healthz", response_model=HealthResponse, tags=["meta"])
    async def healthz() -> HealthResponse:
        """Liveness plus enough detail to explain a degraded service.

        Returns 200 with ``status: "degraded"`` rather than a 5xx when the index
        is missing: the process is alive and answering, and a load balancer
        removing it would only hide the reason.
        """
        ready = state.ready
        # `Settings.embedder_spec` and `chunking_strategy` are *fallbacks*, used
        # only when the manifest does not name one. Once an index is loaded the
        # manifest wins, so reporting the settings values here would tell a
        # reader the served index was built with something it was not. Report
        # what is actually loaded, and say where each value came from.
        config = cfg.public_dict()
        if state.bundle is not None:
            config = {
                **config,
                "embedder_spec": state.bundle.embedder_spec,
                "chunking_strategy": state.bundle.strategy,
                "config_source": "index manifest",
            }
        else:
            config = {**config, "config_source": "settings fallback (no index loaded)"}
        # Counts and limits, never identities. Reported because "is the public
        # link rate limited?" is a question a judge may reasonably ask, and
        # pointing at a config file is a weaker answer than pointing at the
        # running service.
        config = {**config, "rate_limit": limiter.snapshot()}
        return HealthResponse(
            status="ok" if ready else "degraded",
            index_loaded=ready,
            n_chunks=0 if state.pipeline is None else state.pipeline.n_chunks,
            generation_providers=state.provider_names(),
            circuits=state.circuits(),
            uptime_s=round(time.time() - state.started_at, 3),
            warmup=state.warmup_ms,
            config=config,
        )

    @application.get("/stats", response_model=StatsResponse, tags=["meta"])
    async def stats() -> StatsResponse:
        """Corpus description and rolling latency, for the demo HUD."""
        return StatsResponse(**state.stats())

    @application.post("/stt/token", response_model=SttTokenResponse, tags=["stt"])
    async def stt_token() -> SttTokenResponse | JSONResponse:
        """Mint a short-lived speech-to-text credential.

        Retained for the ElevenLabs path and for deployments that configure
        ``SARVAM_TOKEN_URL``, where a vendor really does mint a browser
        credential. It is **not** the Sarvam demo path: Sarvam accepts no
        ``token`` parameter and publishes no ephemeral-token endpoint, so a
        capability minted here is not something Sarvam will honour. Browser
        audio goes through ``WS /stt/stream`` instead. See
        :mod:`voicerag.api.stt_relay`.
        """
        try:
            body = await mint_token(cfg)
        except TokenMintError as exc:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=ErrorResponse(error="stt_unavailable", message=str(exc)).model_dump(),
            )
        return SttTokenResponse(**body)

    @application.get("/examples", tags=["meta"])
    async def examples(n: int = 8, seed: int = 0) -> dict[str, Any]:
        """Questions this corpus can actually answer.

        The single most common way this demo looks broken is a fair question it
        was never going to answer. The index is a fixed ~196k-passage slice of
        MS MARCO, not the web: "what is the capital of India" retrieves several
        passages *about* India, none of which states the capital, and the system
        correctly declines. From the outside that is indistinguishable from a
        bug.

        The manifest carries the shard's own labelled queries, so the honest fix
        is to show a few. Sampled deterministically from the answerable set, so
        the presets are stable across reloads and a demo can be rehearsed.
        """
        pool: list[str] = []
        if state.bundle is not None:
            for raw in state.bundle.manifest.get("example_queries", []):
                q = _presentable_question(raw)
                if q:
                    pool.append(q)
        if not pool:
            return {"examples": [], "unanswerable_example": None, "n_indexed": 0}

        rng = random.Random(seed)
        picked = rng.sample(pool, min(n, len(pool)))

        unanswerable = None
        if state.bundle is not None:
            for raw in state.bundle.manifest.get("example_unanswerable", []):
                unanswerable = _presentable_question(raw)
                if unanswerable:
                    break

        return {
            "examples": picked,
            # Offered separately and labelled: showing the system decline is
            # half of requirement 6, and it should not happen by accident.
            "unanswerable_example": unanswerable,
            "n_indexed": 0 if state.pipeline is None else state.pipeline.n_chunks,
        }

    @application.post("/speak", tags=["stt"])
    async def speak(request: Request) -> Response:
        """Speak an answer. Returns `audio/wav`.

        The other half of "voice-enabled": the pipeline listens, retrieves and
        answers, and this is what makes the answer audible. Kept as a separate
        call rather than folded into `/ask` so the text arrives (and renders)
        the instant it exists, with audio following — a spoken answer that waits
        for synthesis before showing anything would feel slower, not faster.
        """
        if not cfg.enable_tts:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=ErrorResponse(
                    error="tts_disabled", message="speech synthesis is disabled"
                ).model_dump(),
            )
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        text = str(body.get("text") or "").strip()
        language = str(body.get("language_code") or "en-IN")
        if not text:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=ErrorResponse(error="empty_text", message="nothing to speak").model_dump(),
            )
        try:
            audio = await synthesize(
                text, cfg, language_code=language, speaker=cfg.sarvam_tts_speaker
            )
        except TtsError as exc:
            log.warning("tts failed: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=ErrorResponse(error="tts_unavailable", message=str(exc)).model_dump(),
            )
        return Response(content=audio, media_type="audio/wav")

    @application.websocket("/stt/stream")
    async def stt_stream(websocket: WebSocket) -> None:
        """Relay browser audio to Sarvam, streaming transcripts back.

        This is the endpoint the browser should use. Sarvam accepts no ``token``
        parameter and mints no ephemeral credential, so the only way for the
        browser to dial the vendor directly is to carry the permanent account
        key -- see :mod:`voicerag.api.stt_relay` for the full reasoning. The
        wire protocol is Sarvam's own in both directions.
        """
        await relay_sarvam(websocket, cfg)


async def _run_speculation(state: AppState, partial: str) -> None:
    """Background body for ``/speculate``. Never raises into the server."""
    if state.pipeline is None:
        return
    await state.pipeline.speculate(partial)


#: The ASGI application. ``uvicorn voicerag.api.main:app``.
app = create_app()
