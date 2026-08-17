"""Answer generation: the grounded prompt, streaming backends and fallback.

The prompt and the interface import eagerly -- they are pure and dependency-free
-- while the HTTP-backed vendor clients load on first access, so importing this
package never requires ``httpx`` to be installed or a network to exist.
"""

from __future__ import annotations

from typing import Any

from .base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    GenerationResult,
    Generator,
)
from .prompt import (
    ABSTAIN_TOKEN,
    MAX_CONTEXT_CHUNKS,
    SYSTEM_PROMPT,
    ParsedAnswer,
    format_passages,
    parse_answer,
    render,
    usable_passages,
)
from .router import GenerationRouter

__all__ = [
    "ABSTAIN_TOKEN",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "GeminiGenerator",
    "GenerationResult",
    "GenerationRouter",
    "Generator",
    "GroqGenerator",
    "MAX_CONTEXT_CHUNKS",
    "ParsedAnswer",
    "SYSTEM_PROMPT",
    "format_passages",
    "parse_answer",
    "render",
    "usable_passages",
]

_LAZY = {"GroqGenerator": ".groq", "GeminiGenerator": ".gemini"}


def __getattr__(name: str) -> Any:
    """Import vendor generators on first access. See the module docstring."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
