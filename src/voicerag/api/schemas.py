"""Request and response bodies for the HTTP surface.

The answer shape itself lives in :mod:`voicerag.pipeline` -- it is produced by
the pipeline and merely serialised here -- so this module only holds the bodies
that are genuinely transport-level: what a client sends, and what the
operational endpoints return.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "AskRequest",
    "SpeculateRequest",
    "SpeculateResponse",
    "HealthResponse",
    "StatsResponse",
    "SttTokenResponse",
    "ErrorResponse",
]


class AskRequest(BaseModel):
    """One question, as transcribed by the browser's STT session."""

    question: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "The final transcript. Length is bounded here only to reject abuse; "
            "the real content check is the input guard, which explains itself."
        ),
    )
    language: str | None = Field(
        default=None,
        max_length=32,
        description="BCP-47 tag the transcript came from. Recorded, not yet routed on.",
    )


class SpeculateRequest(BaseModel):
    """A partial transcript, sent while the user is still speaking."""

    partial: str = Field(min_length=1, max_length=4000)
    language: str | None = Field(default=None, max_length=32)


class SpeculateResponse(BaseModel):
    """Acknowledgement that a speculation was scheduled.

    Deliberately carries no retrieval result: the client cannot use one, and
    serialising hits would turn a fire-and-forget call into a payload.
    """

    accepted: bool
    key: str = Field(description="Normalised cache key the speculation will be stored under.")
    reason: str | None = Field(
        default=None, description="Why it was not accepted, when it was not."
    )


class HealthResponse(BaseModel):
    """Liveness plus everything needed to explain a degraded service."""

    status: str = Field(description="'ok' when the service can answer, else 'degraded'.")
    index_loaded: bool
    n_chunks: int
    generation_providers: list[str]
    circuits: list[dict[str, Any]]
    uptime_s: float
    warmup: dict[str, float] = Field(
        default_factory=dict, description="Milliseconds per stage measured at startup."
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Non-secret configuration; keys as booleans only."
    )


class StatsResponse(BaseModel):
    """Corpus and recent-latency summary for the demo HUD."""

    chunks: int
    strategy: str
    embedding_model: str
    index_ready: bool
    recent_latency: dict[str, float] = Field(default_factory=dict)
    speculation: dict[str, Any] = Field(default_factory=dict)


class SttTokenResponse(BaseModel):
    """A short-lived speech-to-text credential for the browser.

    Attributes:
        token: The credential. Never the account API key -- see
            :mod:`voicerag.api.stt_token` for what it actually is in each mode.
        expires_in: Seconds until the token stops being accepted.
        provider: Which STT vendor the token is for.
        mode: ``"ephemeral"`` when a vendor mint endpoint issued it,
            ``"capability"`` when this server signed it. Exposed so a client --
            or a judge -- can tell the two apart instead of assuming.
    """

    token: str
    expires_in: int
    provider: str
    mode: str


class ErrorResponse(BaseModel):
    """The single error shape every failing endpoint returns.

    One shape, no stack traces: a traceback in an HTTP body is an information
    leak, and a client that must parse three different error formats will parse
    none of them correctly.
    """

    error: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable, safe to display.")
    trace_id: str | None = Field(
        default=None, description="Correlates with the server log line for this request."
    )
