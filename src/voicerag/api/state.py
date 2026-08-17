"""Process-wide state: the index, the pipeline, and the startup warmup.

Everything expensive is built exactly once, in :meth:`AppState.startup`, and
shared by every request. That is not a micro-optimisation -- loading a FAISS
index and fitting-or-unpickling an embedder costs seconds, and doing it per
request would make the 200 ms target meaningless.

The state is deliberately tolerant of a missing index: the server starts,
``/healthz`` reports ``index_loaded: false`` with the path it looked in, and
``/ask`` returns a clean 503. A server that refuses to boot tells you nothing;
one that boots and explains itself tells you which command to run next.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..generate.base import Generator
from ..generate.router import GenerationRouter
from ..index.hybrid import HybridIndex
from ..index.store import ChunkStore
from ..pipeline import RagPipeline, SpeculationCache, build_policy

__all__ = [
    "AppState",
    "IndexBundle",
    "load_index",
    "build_generators",
    "make_test_state",
]

log = logging.getLogger("voicerag.api")


class IndexBundle:
    """A loaded index and the embedder that produced its vectors.

    The two travel together and are useless apart: the LSA projection is fitted
    per corpus, so serving an index with a different embedder returns confident
    nonsense rather than an error. Bundling them makes that impossible to get
    wrong by accident.

    Attributes:
        hybrid: The dense+sparse index.
        store: Chunk payloads for citations.
        embedder: The query encoder.
        manifest: ``manifest.json`` as written by ``scripts/ingest.py``, minus
            the bulky ``queries``/``qrels``/``unanswerable`` blocks.
    """

    __slots__ = ("hybrid", "store", "embedder", "manifest")

    def __init__(
        self,
        hybrid: HybridIndex,
        store: ChunkStore,
        embedder: Any,
        manifest: dict[str, Any],
    ) -> None:
        self.hybrid = hybrid
        self.store = store
        self.embedder = embedder
        self.manifest = manifest

    @property
    def embedder_spec(self) -> str:
        """Spec string the index was built with, for ``/stats``."""
        return str(self.manifest.get("embedder_spec", "unknown"))

    @property
    def strategy(self) -> str:
        """Chunking strategy the index was built with."""
        return str(self.manifest.get("strategy", "unknown"))


def load_index(settings: Settings | None = None) -> IndexBundle | None:
    """Load the persisted index, or return ``None`` if there is not one.

    Args:
        settings: Configuration naming ``index_dir``.

    Returns:
        An :class:`IndexBundle`, or ``None`` when the directory is absent or
        incomplete. Never raises for a missing index -- that is a deployment
        state, not an exception -- but does propagate a genuinely corrupt one,
        because silently serving half an index is worse than failing to start.
    """
    cfg = settings or get_settings()
    directory = Path(cfg.index_dir)
    if not (directory / "hybrid.json").exists():
        log.warning("no index at %s; run scripts/ingest.py", directory)
        return None

    manifest: dict[str, Any] = {}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The manifest embeds the evaluation query set, which can be megabytes.
        # Only the build description is useful at serving time.
        manifest = {
            k: v for k, v in raw.items() if k not in ("queries", "qrels", "unanswerable")
        }

    hybrid = HybridIndex.load(directory, mmap_sparse=cfg.mmap_sparse)
    store = ChunkStore.load(directory / "store")
    embedder = _load_embedder(directory, manifest, cfg)
    log.info(
        "index loaded: %d chunks, strategy=%s, embedder=%s",
        len(store),
        manifest.get("strategy", "?"),
        manifest.get("embedder_spec", cfg.embedder_spec),
    )
    return IndexBundle(hybrid, store, embedder, manifest)


def _load_embedder(directory: Path, manifest: dict[str, Any], settings: Settings) -> Any:
    """Restore the exact embedder the index was built with.

    A pickled model beside the index wins over the spec string, because a
    corpus-fitted projection cannot be reconstructed from its dimension alone.
    Falling back to a fresh model of the same spec would produce an embedder
    that runs, returns vectors, and matches nothing.
    """
    filename = str(manifest.get("embedder_file") or "")
    if filename:
        path = directory / filename
        if path.exists():
            from ..embed.lsa import HashingLSAEmbedder

            return HashingLSAEmbedder.load(path)
        log.warning("manifest names %s but it is missing; falling back to spec", filename)

    from ..embed.base import resolve_embedder

    spec = str(manifest.get("embedder_spec") or settings.embedder_spec)
    if spec.split(":", 1)[0].lower() == "lsa":
        raise RuntimeError(
            f"index at {directory} was built with {spec!r} but no fitted embedder was "
            "saved beside it; an unfitted LSA projection cannot serve this index. "
            "Rebuild with scripts/ingest.py."
        )
    return resolve_embedder(spec, cache_size=512)


def build_generators(settings: Settings | None = None) -> list[Generator]:
    """Instantiate the configured LLM backends, fastest first.

    Groq leads because it is the fast path. OpenAI and Gemini follow because they
    are independent failure domains, which is the only kind of fallback worth
    having -- and because Groq's free tier caps at 8,000 tokens per minute, which
    a RAG prompt carrying five passages exhausts in about four questions. That
    ceiling is reached by a judge asking questions quickly, not just by a load
    test, so the chain needs a real second link rather than a nominal one.

    An empty list is a valid outcome and means the deployment has no credentials
    -- reported by ``/healthz``, not raised here.
    """
    cfg = settings or get_settings()
    generators: list[Generator] = []
    if cfg.groq_api_key is not None:
        from ..generate.groq import GroqGenerator

        generators.append(
            GroqGenerator(
                cfg.groq_api_key.get_secret_value(),
                model=cfg.groq_model,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                reasoning_effort=cfg.groq_reasoning_effort,
            )
        )
    if cfg.openai_api_key is not None:
        from ..generate.openai import OpenAIGenerator

        generators.append(
            OpenAIGenerator(
                cfg.openai_api_key.get_secret_value(),
                model=cfg.openai_model,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
        )
    if cfg.gemini_api_key is not None:
        from ..generate.gemini import GeminiGenerator

        generators.append(
            GeminiGenerator(
                cfg.gemini_api_key.get_secret_value(),
                model=cfg.gemini_model,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
        )
    return generators


class AppState:
    """Everything the request handlers need, built once at startup.

    Args:
        settings: Configuration.
        pipeline: A pre-built pipeline. Supplying one skips index loading
            entirely, which is how the tests run a real server over a tiny
            in-memory index with fake providers.
        latency_window: How many recent requests feed the ``/stats`` rolling
            percentiles.
    """

    __slots__ = (
        "settings",
        "pipeline",
        "bundle",
        "speculation",
        "started_at",
        "warmup_ms",
        "_latencies",
        "_requests",
    )

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        pipeline: RagPipeline | None = None,
        latency_window: int = 128,
    ) -> None:
        self.settings = settings or get_settings()
        self.pipeline = pipeline
        self.bundle: IndexBundle | None = None
        self.speculation = (
            pipeline.speculation
            if pipeline is not None
            else SpeculationCache(
                max_size=self.settings.speculation_cache_size,
                ttl_s=self.settings.speculation_ttl_s,
            )
        )
        self.started_at = time.time()
        self.warmup_ms: dict[str, float] = {}
        self._latencies: deque[float] = deque(maxlen=latency_window)
        self._requests = 0

    # -- lifecycle ------------------------------------------------------------

    async def startup(self) -> None:
        """Load the index, build the pipeline and warm every model.

        Warmup is not optional and not conditional. A cold first request pays
        FAISS visit-table allocation, the bm25s score matrix's first touch and
        the embedder matrix paging in; if that request is the judged one, the
        headline number is process startup rather than the pipeline.
        """
        if self.pipeline is None:
            self.bundle = load_index(self.settings)
            generators = build_generators(self.settings)
            self.pipeline = RagPipeline(
                embedder=None if self.bundle is None else self.bundle.embedder,
                hybrid=None if self.bundle is None else self.bundle.hybrid,
                store=None if self.bundle is None else self.bundle.store,
                generator=GenerationRouter(generators) if generators else None,
                policy=build_policy(self.settings),
                settings=self.settings,
                speculation=self.speculation,
            )
        elif self.pipeline.speculation is None:
            self.pipeline.speculation = self.speculation

        self.warmup_ms = await self.pipeline.warmup()
        await self._warm_providers()
        log.info(
            "warmup complete: %s",
            ", ".join(f"{k}={v:.2f}ms" for k, v in sorted(self.warmup_ms.items())) or "no index",
        )

    async def _warm_providers(self) -> None:
        """Pre-establish provider TLS so the first answer does not pay a handshake.

        Best-effort: a provider that cannot be reached at startup will simply be
        reached (or fail over) on the first real request, and refusing to start
        because a third party is slow would be its own outage.
        """
        if self.pipeline is None or self.pipeline.router is None:
            return
        for provider in self.pipeline.router.providers:
            warm = getattr(provider, "warm", None)
            if warm is None:
                continue
            try:
                ok = await warm()
            except Exception as exc:  # noqa: BLE001 - advisory only
                log.warning("provider %s warmup raised %s", provider.name, type(exc).__name__)
                continue
            log.info("provider %s warm=%s", provider.name, ok)

    async def shutdown(self) -> None:
        """Close provider transports and retrieval thread pools."""
        if self.pipeline is not None:
            await self.pipeline.aclose()

    # -- accessors ------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True when a pipeline with an index is available."""
        return self.pipeline is not None and self.pipeline.ready

    def record_latency(self, ms: float) -> None:
        """Add one completed request to the rolling window behind ``/stats``."""
        self._latencies.append(float(ms))
        self._requests += 1

    def recent_latency(self) -> dict[str, float]:
        """Percentiles over the rolling window.

        Nearest-rank, matching :func:`eval.metrics.percentile`, so a number in
        the HUD means the same thing as the identically named number in the
        benchmark report.
        """
        if not self._latencies:
            return {}
        values = sorted(self._latencies)
        n = len(values)

        def pct(p: float) -> float:
            rank = max(1, min(n, int(-(-p * n // 100))))
            return round(values[rank - 1], 3)

        return {
            "n": float(n),
            "requests": float(self._requests),
            "p50": pct(50),
            "p70": pct(70),
            "p95": pct(95),
            "p100": pct(100),
        }

    def provider_names(self) -> list[str]:
        """Configured generation providers, in preference order."""
        if self.pipeline is None or self.pipeline.router is None:
            return []
        return [p.name for p in self.pipeline.router.providers]

    def circuits(self) -> list[dict[str, Any]]:
        """Circuit-breaker snapshot per provider."""
        return [] if self.pipeline is None else self.pipeline.health()

    def stats(self) -> dict[str, Any]:
        """The ``/stats`` body."""
        return {
            "chunks": 0 if self.pipeline is None else self.pipeline.n_chunks,
            "strategy": (
                self.bundle.strategy if self.bundle is not None
                else self.settings.chunking_strategy
            ),
            "embedding_model": (
                self.bundle.embedder_spec if self.bundle is not None
                else self.settings.embedder_spec
            ),
            "index_ready": self.ready,
            "recent_latency": self.recent_latency(),
            "speculation": self.speculation.stats() if self.speculation is not None else {},
        }


def make_test_state(
    *,
    embedder: Any,
    hybrid: HybridIndex,
    store: ChunkStore,
    generators: Sequence[Generator] | None = None,
    settings: Settings | None = None,
) -> AppState:
    """Build an :class:`AppState` over an in-memory index.

    Lives in the shipped package rather than in the test file because the same
    assembly is what a notebook, a load test or a demo script needs, and a
    second copy of it would drift from this one.

    Args:
        embedder: Fitted query encoder.
        hybrid: Built hybrid index.
        store: Chunk store matching ``hybrid``.
        generators: Generation backends, fastest first.
        settings: Configuration.

    Returns:
        An :class:`AppState` whose :meth:`AppState.startup` will warm but not
        load from disk.
    """
    cfg = settings or get_settings()
    cache = SpeculationCache(
        max_size=cfg.speculation_cache_size, ttl_s=cfg.speculation_ttl_s
    )
    pipeline = RagPipeline(
        embedder=embedder,
        hybrid=hybrid,
        store=store,
        generator=list(generators or []) or None,
        settings=cfg,
        speculation=cache,
    )
    return AppState(cfg, pipeline=pipeline)
