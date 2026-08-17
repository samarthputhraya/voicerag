"""model2vec static embeddings: the production query encoder.

A model2vec "potion" model is a distilled *static* embedding table -- the
transformer has been collapsed into one vector per token, so encoding is a
gather over a lookup table plus a weighted mean. That has three properties this
system is built around:

* **Sub-millisecond queries on CPU.** No attention, no matmul over layers. A
  spoken question embeds in well under 1 ms, which is what makes speculative
  re-embedding on every ``transcript.partial`` affordable at all.
* **No torch.** The whole serving stack stays free of a 2 GB dependency; cold
  start is seconds, not minutes, and the container stays small.
* **Constant time in sequence length.** Latency does not spike on a long,
  rambling utterance -- important when the tail of the latency distribution is
  a graded deliverable.

The price is no contextual disambiguation. For MS MARCO-style factoid queries
the retrieval-tuned checkpoint recovers most of that gap, and any residual loss
is what the reranker and the hybrid BM25 leg are for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseEmbedder, EmbeddingBackendUnavailable

__all__ = ["StaticEmbedder", "DEFAULT_STATIC_MODEL", "SMALL_STATIC_MODEL"]

#: Retrieval-tuned distillation. Chosen over the general-purpose checkpoint
#: because it was distilled on a retrieval objective, which is exactly our task.
DEFAULT_STATIC_MODEL = "minishlab/potion-retrieval-32M"

#: Smaller, faster, lower quality. The right pick when RAM is the binding
#: constraint rather than nDCG.
SMALL_STATIC_MODEL = "minishlab/potion-base-8M"


class StaticEmbedder(BaseEmbedder):
    """Wraps a ``model2vec.StaticModel`` behind the project's embedder contract.

    Loading is **lazy**: constructing the object never touches the filesystem or
    the network. That keeps ``resolve_embedder`` cheap in config-parsing paths,
    lets tests construct the object on a machine with no weights, and defers the
    (potentially large) allocation until something actually encodes.

    Args:
        model_id: Hugging Face repo id *or* a local directory. Local paths are
            detected automatically, so a mounted weights volume needs no
            separate flag.
        batch_size: Texts per encode call during ingest. Static models are
            memory-bandwidth bound, so large batches help throughput and are
            harmless to single-query latency.
        normalize_in_model: Whether to ask model2vec to normalise. Left off by
            default because :class:`~voicerag.embed.base.BaseEmbedder`
            normalises anyway, and doing it twice is wasted work.
    """

    name = "static"

    def __init__(
        self,
        model_id: str = DEFAULT_STATIC_MODEL,
        *,
        batch_size: int = 512,
        normalize_in_model: bool = False,
    ) -> None:
        super().__init__(batch_size=batch_size)
        if not model_id or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        self.model_id = model_id.strip()
        self.normalize_in_model = bool(normalize_in_model)
        self._model: Any | None = None
        self._dim: int | None = None

    # ------------------------------------------------------------ construction

    @classmethod
    def from_path(cls, path: str | Path, **kwargs: Any) -> "StaticEmbedder":
        """Build an embedder from a locally downloaded model directory.

        Use this on air-gapped or rate-limited machines: download once with
        ``huggingface-cli download minishlab/potion-retrieval-32M --local-dir
        models/potion``, then point here.

        Args:
            path: Directory containing the model2vec files.
            **kwargs: Forwarded to :class:`StaticEmbedder`.

        Returns:
            A lazily-loading embedder bound to that directory.

        Raises:
            EmbeddingBackendUnavailable: If the directory does not exist. This
                is checked eagerly, unlike a repo id, because a wrong local
                path is a deployment mistake worth failing on immediately.
        """
        directory = Path(path).expanduser()
        if not directory.exists():
            raise EmbeddingBackendUnavailable(
                f"no static model directory at {directory}. Download one with: "
                f"huggingface-cli download {DEFAULT_STATIC_MODEL} --local-dir {directory}"
            )
        return cls(str(directory), **kwargs)

    @classmethod
    def from_model(cls, model: Any, **kwargs: Any) -> "StaticEmbedder":
        """Adopt an already-constructed ``StaticModel`` instance.

        Exists so a single loaded model can be shared between retrieval and the
        semantic/contextual chunking strategies instead of being held twice in
        RAM, and so tests can exercise the batching and normalisation plumbing
        with a stub.

        Args:
            model: Any object exposing ``encode(list[str]) -> np.ndarray``.
            **kwargs: Forwarded to :class:`StaticEmbedder`.

        Returns:
            An embedder that is already "loaded".

        Raises:
            TypeError: If ``model`` has no ``encode`` method.
        """
        if not hasattr(model, "encode"):
            raise TypeError(
                f"{type(model).__name__} has no .encode(); expected a StaticModel-like object"
            )
        kwargs.setdefault("model_id", getattr(model, "base_model_name", "in-memory"))
        embedder = cls(**kwargs)
        embedder._model = model
        return embedder

    # ---------------------------------------------------------------- public

    @property
    def dim(self) -> int:
        """Output dimensionality; triggers the lazy load on first access."""
        if self._dim is None:
            model = self._ensure_model()
            probe = np.asarray(model.encode(["dimension probe"]))
            self._dim = int(probe.reshape(1, -1).shape[-1])
        return self._dim

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying model has been materialised."""
        return self._model is not None

    # ------------------------------------------------------------- internals

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        return np.asarray(model.encode(texts), dtype=np.float32)

    def _ensure_model(self) -> Any:
        """Load the static model once, converting every failure into a fix-it.

        model2vec surfaces a missing package, a missing hub entry and a blocked
        network as three unrelated exception types. Retrieval code above us only
        cares "can I use this backend or must I fall back", so all of them are
        normalised into :class:`EmbeddingBackendUnavailable` with a message that
        names the actual remedy.
        """
        if self._model is not None:
            return self._model

        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - package is pinned
            raise EmbeddingBackendUnavailable(
                "model2vec is not installed; run `pip install model2vec` "
                "or use the offline 'lsa' embedder"
            ) from exc

        try:
            self._model = StaticModel.from_pretrained(
                self.model_id, normalize=self.normalize_in_model
            )
        except Exception as exc:
            raise EmbeddingBackendUnavailable(
                f"could not load static model {self.model_id!r} ({type(exc).__name__}: {exc}). "
                "If this machine has no network access, download the model elsewhere with "
                f"`huggingface-cli download {self.model_id} --local-dir <dir>` and use "
                "StaticEmbedder.from_path(<dir>), or fall back to the 'lsa' embedder."
            ) from exc
        return self._model
