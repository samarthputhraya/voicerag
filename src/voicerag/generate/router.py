"""Generator fallback routing: try the fast one, survive when it is down.

Wraps :func:`~voicerag.harness.resilience.first_healthy` over an ordered list of
generators, each guarded by its own circuit breaker. The ordering is the
contract: fastest first (Groq), most independent last (Gemini).

Two decisions here are specific to *generation* and differ from how the rest of
the harness treats a failing call:

**Retries are off by default.** Elsewhere a retry is cheap. Here, re-running a
failed generation costs a whole generation -- the single most expensive stage in
the pipeline -- and in a 200 ms budget that attempt cannot possibly land in
time. Falling straight through to the next provider spends the same wall-clock
on something that might actually succeed. Hence ``attempts=1`` and a fallback
chain rather than a retry loop.

**Streaming fails over only before the first token.** Once the user has seen a
token, switching providers would splice two different answers into one sentence.
So :meth:`GenerationRouter.stream` will try the next provider if the current one
fails while opening the stream, and propagates the error once any text has been
emitted. :meth:`GenerationRouter.complete` has no such constraint -- nothing has
been shown yet -- and can fail over at any point.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, MutableMapping, Sequence

from ..harness.resilience import (
    AllProvidersFailed,
    CircuitBreaker,
    Deadline,
    RetryPolicy,
    first_healthy,
)
from ..harness.trace import current_trace
from .base import GenerationResult, Generator

__all__ = ["GenerationRouter", "DEFAULT_GENERATION_POLICY"]

#: One attempt per provider; see the module docstring. ``timeout_ms`` is a
#: backstop against a hung socket, not a latency target -- the caller's
#: :class:`~voicerag.harness.resilience.Deadline` is what enforces the budget.
DEFAULT_GENERATION_POLICY = RetryPolicy(
    attempts=1, base_ms=0.0, max_ms=0.0, timeout_ms=10_000.0
)


class GenerationRouter:
    """Ordered generator fallback with per-provider circuit breakers.

    Args:
        providers: Generators in preference order, fastest first.
        breakers: Pre-built breakers by provider name. Supply your own to share
            health state across routers, or to inspect it in a health endpoint.
        policy: Retry policy per provider; see
            :data:`DEFAULT_GENERATION_POLICY`.
        stage: Span/mark prefix used in the trace.

    Raises:
        ValueError: If ``providers`` is empty.
    """

    __slots__ = ("_providers", "_breakers", "_policy", "_stage")

    def __init__(
        self,
        providers: Sequence[Generator],
        *,
        breakers: dict[str, CircuitBreaker] | None = None,
        policy: RetryPolicy | None = None,
        stage: str = "generate",
    ) -> None:
        if not providers:
            raise ValueError("GenerationRouter requires at least one provider")
        self._providers = list(providers)
        self._breakers = breakers if breakers is not None else {}
        self._policy = policy or DEFAULT_GENERATION_POLICY
        self._stage = stage

    @property
    def providers(self) -> list[Generator]:
        """The configured generators, in preference order."""
        return list(self._providers)

    def breaker(self, name: str) -> CircuitBreaker:
        """Return (creating if needed) the breaker for a provider name."""
        return self._breakers.setdefault(name, CircuitBreaker(name=name))

    def health(self) -> list[dict[str, Any]]:
        """Breaker state per provider, for a health endpoint or the demo HUD."""
        return [self.breaker(p.name).snapshot() for p in self._providers]

    async def complete(
        self,
        system: str,
        user: str,
        *,
        deadline: Deadline | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a complete answer from the first healthy provider.

        Args:
            system: System prompt.
            user: User turn.
            deadline: Shared request budget. Providers whose attempt cannot fit
                in the remaining time are skipped rather than started.
            **kwargs: Passed through to the generator.

        Returns:
            The :class:`~voicerag.generate.base.GenerationResult`, whose
            ``provider`` field records which backend actually answered.

        Raises:
            AllProvidersFailed: Every provider failed or was skipped.
        """
        candidates = [
            (p.name, _bind(p, system, user, kwargs)) for p in self._providers
        ]
        _, result = await first_healthy(
            candidates,
            breakers=self._breakers,
            policy=self._policy,
            deadline=deadline,
            stage=self._stage,
        )
        return result

    async def stream(
        self,
        system: str,
        user: str,
        *,
        meta: MutableMapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream an answer, failing over only before the first token.

        Args:
            system: System prompt.
            user: User turn.
            meta: Optional dict receiving provider metadata and timings. The
                selected provider's name is written to ``meta["provider"]``.
            **kwargs: Passed through to the generator.

        Yields:
            Text deltas from whichever provider answered.

        Raises:
            AllProvidersFailed: No provider produced a first token.
            Exception: Whatever the selected provider raised *after* it had
                already emitted text -- deliberately not retried, because a
                partially delivered answer cannot be un-said.
        """
        sink: MutableMapping[str, Any] = meta if meta is not None else {}
        trace = current_trace()
        errors: dict[str, str] = {}

        for provider in self._providers:
            breaker = self.breaker(provider.name)
            if not breaker.allows():
                errors[provider.name] = "circuit open"
                if trace is not None:
                    trace.mark(
                        f"{self._stage}.skipped",
                        provider=provider.name,
                        reason="circuit_open",
                    )
                continue

            started = False
            try:
                async for delta in provider.stream(system, user, meta=sink, **kwargs):
                    if not started:
                        started = True
                        sink["provider"] = provider.name
                        breaker.record_success()
                        if trace is not None:
                            trace.mark(
                                f"{self._stage}.selected", provider=provider.name
                            )
                    yield delta
            except Exception as exc:  # noqa: BLE001 - classified by `started`
                if started:
                    # Text is already on the user's screen. A second provider
                    # would continue someone else's sentence.
                    raise
                breaker.record_failure()
                errors[provider.name] = f"{type(exc).__name__}: {exc}"
                if trace is not None:
                    trace.mark(
                        f"{self._stage}.failed",
                        provider=provider.name,
                        error=errors[provider.name],
                    )
                continue
            if started:
                return
            # The provider completed without emitting anything. An empty answer
            # is a failure of that provider, not an answer, so try the next one.
            breaker.record_failure()
            errors[provider.name] = "empty stream"
            if trace is not None:
                trace.mark(
                    f"{self._stage}.failed", provider=provider.name, error="empty stream"
                )

        raise AllProvidersFailed(errors)

    async def aclose(self) -> None:
        """Close every provider's transport."""
        for provider in self._providers:
            await provider.aclose()


def _bind(
    provider: Generator,
    system: str,
    user: str,
    kwargs: dict[str, Any],
):
    """Bind a provider call into the zero-argument thunk ``first_healthy`` wants.

    A named factory rather than an inline lambda in a comprehension: the classic
    late-binding closure bug would otherwise have every thunk call the *last*
    provider, and the fallback would look like it worked while never actually
    switching.
    """

    async def call() -> GenerationResult:
        return await provider.complete(system, user, **kwargs)

    return call
