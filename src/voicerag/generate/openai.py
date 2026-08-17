"""OpenAI chat completions, as an independent fallback for Groq.

Why a third provider
--------------------

Groq is the fast path and stays first. But the free-tier key this project runs on
is capped at **8,000 tokens per minute**, and a RAG prompt carrying five
retrieved passages costs roughly 2,000 of them. That is about four questions a
minute before the vendor starts refusing — measured, not estimated, from
``x-ratelimit-limit-tokens`` on a live response.

A judge asking questions in quick succession will hit that ceiling. When they do,
:class:`~voicerag.harness.resilience.CircuitBreaker` opens and every subsequent
request returns "every generation provider failed" — which is honest, and useless
in a demo. An ordered fallback chain only helps if there is somewhere to fall
back *to*, and until now a deployment with only a Groq key had a chain of one.

OpenAI is a genuinely independent failure domain: different company, different
network path, different quota. That is the only kind of fallback worth having.

Implementation note
-------------------

Groq's API *is* OpenAI's chat-completions API, which is why
:class:`~voicerag.generate.groq.GroqGenerator` already accepts a ``base_url``.
This subclass changes three things and inherits the rest: the endpoint, the
provider ``name`` (so circuit-breaker state and ``/healthz`` report the two
separately), and the output-cap spelling — OpenAI's newer models reject
``max_tokens`` and require ``max_completion_tokens``.
"""

from __future__ import annotations

from typing import Any

from .base import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from .groq import GroqGenerator

__all__ = ["OpenAIGenerator", "OPENAI_BASE_URL", "DEFAULT_OPENAI_MODEL"]

OPENAI_BASE_URL = "https://api.openai.com/v1"

#: Small, fast and cheap. The task is extractive summarisation over passages the
#: retriever already found, not open-ended reasoning, so a frontier model buys
#: latency and cost without buying accuracy that matters here.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAIGenerator(GroqGenerator):
    """Streaming client for OpenAI's chat completions API.

    Args:
        api_key: OpenAI API key.
        model: Model id. Defaults to :data:`DEFAULT_OPENAI_MODEL`.
        base_url: API root, without a trailing slash.
        max_tokens: Hard output ceiling.
        temperature: Sampling temperature.
        max_tokens_field: Defaults to ``"max_completion_tokens"``, which the
            newer OpenAI models require and the older ones accept.
        timeout_s: Total request timeout. Higher than Groq's default because
            this is the fallback -- it is already a degraded path, and giving up
            early on it means giving up entirely.
        **kwargs: Forwarded to :class:`~voicerag.generate.groq.GroqGenerator`.
    """

    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_OPENAI_MODEL,
        base_url: str = OPENAI_BASE_URL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens_field: str = "max_completion_tokens",
        timeout_s: float = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            # OpenAI's chat models reject `reasoning_effort` outright; it is a
            # Groq/gpt-oss parameter. Sending it here is a 400.
            reasoning_effort=None,
            max_tokens_field=max_tokens_field,
            timeout_s=timeout_s,
            **kwargs,
        )
