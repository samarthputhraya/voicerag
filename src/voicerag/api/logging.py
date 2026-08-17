"""Structured JSON request logging.

One line per request, machine-parseable, carrying the trace breakdown. That last
part is the point: the same span timings that render in the demo HUD are in the
log, so a slow request can be diagnosed from a log aggregator without asking the
user to reproduce it with devtools open.

Transcripts are **not** logged unless ``LOG_QUERIES=true``. A latency log is not
a place to accumulate recordings of what people said.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

__all__ = ["JsonFormatter", "configure_logging", "log_request"]


class JsonFormatter(logging.Formatter):
    """Render records as one JSON object per line.

    Anything passed via ``extra={"fields": {...}}`` is merged into the object,
    which is how request handlers attach the trace breakdown without inventing
    a log-message format that has to be parsed with a regex later.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            # The type and message only. A traceback in a JSON log line is
            # unreadable, and it belongs on the exception handler's own record.
            exc_type, exc_value = record.exc_info[0], record.exc_info[1]
            payload["error"] = (
                f"{exc_type.__name__}: {exc_value}" if exc_type is not None else "unknown"
            )
        return json.dumps(payload, separators=(",", ":"), default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Install the JSON formatter on the ``voicerag`` logger tree.

    Idempotent: calling it twice does not double every line, which matters
    because ``uvicorn --reload`` re-imports the app module.

    Args:
        level: Log level name.

    Returns:
        The ``voicerag`` logger.
    """
    logger = logging.getLogger("voicerag")
    logger.setLevel(level)
    logger.propagate = False
    for existing in logger.handlers:
        if isinstance(existing.formatter, JsonFormatter):
            return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_request(
    logger: logging.Logger,
    *,
    route: str,
    status: int,
    duration_ms: float,
    trace_id: str | None = None,
    breakdown: dict[str, float] | None = None,
    question: str | None = None,
    **extra: Any,
) -> None:
    """Emit the one structured line for a completed request.

    Args:
        logger: Target logger, normally ``voicerag.api``.
        route: Path template, e.g. ``/ask``.
        status: HTTP status code.
        duration_ms: Server-side wall clock for the whole request.
        trace_id: Correlates this line with the response body's trace block.
        breakdown: Per-span milliseconds.
        question: Transcript; pass ``None`` unless ``LOG_QUERIES`` is on.
        **extra: Additional structured fields.
    """
    fields: dict[str, Any] = {
        "route": route,
        "status": status,
        "duration_ms": round(duration_ms, 3),
        **extra,
    }
    if trace_id:
        fields["trace_id"] = trace_id
    if breakdown:
        fields["breakdown"] = {k: round(v, 3) for k, v in breakdown.items()}
    if question is not None:
        fields["question"] = question
    logger.info("request", extra={"fields": fields})
