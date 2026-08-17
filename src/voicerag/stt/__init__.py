"""Speech-to-text: realtime providers and speculative retrieval.

The public surface is deliberately small: a provider-agnostic event type, the
two vendor clients, and the driver that turns partial transcripts into
pre-warmed retrieval.

Vendor clients are imported lazily through ``__getattr__`` so that importing
:mod:`voicerag.stt` costs nothing and never fails on a machine without
``websockets`` or ``httpx`` installed -- the API server imports this package on
every start, including in deployments where audio never touches the server.
"""

from __future__ import annotations

from typing import Any

from .base import (
    AudioTooLongError,
    EventKind,
    SttError,
    SttFatalError,
    SttProvider,
    TranscriptEvent,
    WebSocketSttProvider,
    frame_pcm,
    iter_chunks,
)
from .speculative import (
    SpeculationOutcome,
    SpeculationStats,
    SpeculativeTranscriptDriver,
    normalise_query,
)

__all__ = [
    "AudioTooLongError",
    "ElevenLabsRealtimeStt",
    "EventKind",
    "SarvamBatchStt",
    "SarvamRealtimeStt",
    "SpeculationOutcome",
    "SpeculationStats",
    "SpeculativeTranscriptDriver",
    "SttError",
    "SttFatalError",
    "SttProvider",
    "TranscriptEvent",
    "WebSocketSttProvider",
    "frame_pcm",
    "iter_chunks",
    "normalise_query",
]

_LAZY = {
    "SarvamRealtimeStt": ".sarvam",
    "SarvamBatchStt": ".sarvam",
    "ElevenLabsRealtimeStt": ".elevenlabs",
}


def __getattr__(name: str) -> Any:
    """Import vendor clients on first access. See the module docstring."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
