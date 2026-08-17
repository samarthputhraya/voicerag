"""HTTP surface for the VoiceRAG pipeline.

``uvicorn voicerag.api.main:app`` serves it. Import
:func:`~voicerag.api.main.create_app` instead of the module-level ``app`` when
you need an application over an injected index -- tests and load harnesses do.

Nothing is re-exported eagerly here: importing this package must not import
FastAPI, because ``scripts/bench_latency.py`` and the evaluation harness share
the pipeline with the server and have no use for a web framework.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app", "app", "AppState"]

_LAZY = {"create_app": ".main", "app": ".main", "AppState": ".state"}


def __getattr__(name: str) -> Any:
    """Import the web layer on first access. See the module docstring."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
