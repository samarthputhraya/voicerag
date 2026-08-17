"""Dense ANN retrieval over L2-normalised embeddings.

Design note -- why HNSW and why we stopped optimising it
---------------------------------------------------------
Measured on this project's target hardware (2 vCPU), 200k vectors x dim 256,
``IndexHNSWFlat`` with ``METRIC_INNER_PRODUCT`` and ``efConstruction=80``:

===========  ========  ========  =========
``efSearch``  p50 (ms)  p70 (ms)  p100 (ms)
===========  ========  ========  =========
32            0.630     0.679     1.331
64            1.126     1.179     2.146
128           2.129     2.209     2.958
===========  ========  ========  =========

Build took 141.7 s and the resident set was ~205 MB.

Against a 200 ms end-to-end budget, even the slowest setting costs ~1 % of it.
The engineering conclusion that shapes the rest of this package is therefore:
*dense ANN search is free; spend the budget on embedding, fusion and generation
instead.* We default to ``ef_search=64`` -- near-exact recall at ~1 ms -- rather
than squeezing to 32, because the extra 0.5 ms buys recall we would otherwise
have to claw back with a reranker that costs 10-50x more.

The three knobs, and what each actually trades:

* ``m`` -- graph out-degree. Fixed at build time, dominates RAM
  (``~m * 2 * 4`` bytes/vector of links on top of the raw ``dim * 4`` bytes).
  Higher ``m`` raises the recall ceiling on hard, high-dimensional data.
  ``m=32`` is the standard sweet spot and what we measured.
* ``ef_construction`` -- candidate breadth while inserting. Pure build-time
  cost, permanently baked into graph quality. 80 is deliberately modest: the
  corpus is rebuilt on every ablation run, so a 2.4-minute build matters and
  the recall difference versus 200 was not visible at our ``ef_search``.
* ``ef_search`` -- candidate breadth per query. The only knob that is free to
  change after build, which is why :meth:`DenseIndex.set_ef` exists and why the
  ablation reports a recall/latency curve rather than a single number.

Inner product on L2-normalised vectors is exactly cosine similarity, so scores
land in ``[-1, 1]`` and are directly comparable across queries -- a property the
guardrail layer relies on when it decides whether retrieval was good enough to
answer at all.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import faiss
import numpy as np

__all__ = ["DenseIndex", "EfCurvePoint", "set_faiss_threads"]

_META_FILE = "dense_meta.json"
_INDEX_FILE = "dense.faiss"
_IDS_FILE = "dense_ids.json"


def set_faiss_threads(n: int) -> None:
    """Set faiss's OpenMP thread count for this process.

    Exposed as an explicit function rather than a constructor argument because
    it is genuinely process-global: hiding it inside ``DenseIndex.__init__``
    would let one index silently reconfigure another. Serving wants ``1`` (per-
    query latency is already sub-millisecond and threads only add contention
    when many requests are in flight); bulk index builds want all cores.

    Args:
        n: Number of OpenMP threads faiss may use.
    """
    faiss.omp_set_num_threads(int(n))


@dataclass(slots=True, frozen=True)
class EfCurvePoint:
    """One point on the recall/latency tradeoff curve.

    Attributes:
        ef_search: The ``efSearch`` value measured.
        recall: Mean recall@k against exact inner-product ground truth.
        p50_ms: Median single-query latency.
        p70_ms: 70th-percentile single-query latency.
        p100_ms: Worst observed single-query latency.
    """

    ef_search: int
    recall: float
    p50_ms: float
    p70_ms: float
    p100_ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ef_search": self.ef_search,
            "recall": round(self.recall, 4),
            "p50_ms": round(self.p50_ms, 3),
            "p70_ms": round(self.p70_ms, 3),
            "p100_ms": round(self.p100_ms, 3),
        }


class DenseIndex:
    """HNSW approximate nearest-neighbour index over normalised embeddings.

    The index owns a position -> ``chunk_id`` mapping because ``IndexHNSWFlat``
    cannot store external 64-bit ids itself. Wrapping it in ``IndexIDMap`` would
    work but adds a hash lookup per result and forces integer ids; our ids are
    content-addressed strings, so a plain list indexed by position is both
    faster and simpler.

    Query buffers are thread-local and preallocated. At ~1 ms per search, a
    fresh ``np.asarray`` + copy + normalise per query is a measurable fraction
    of the call, and this package is explicitly optimised to be allocation-light
    on the hot path. Thread-local (rather than a single shared buffer) keeps the
    index safe to query from the hybrid retriever's worker threads.

    Args:
        ef_search: Default query-time breadth. Overridable per call.

    Attributes:
        dim: Embedding dimensionality, or ``None`` before :meth:`build`.
        ids: Position-ordered chunk ids.
    """

    def __init__(self, *, ef_search: int = 64) -> None:
        self._index: faiss.IndexHNSWFlat | None = None
        self._ids: list[str] = []
        self._dim: int | None = None
        self._ef_search = int(ef_search)
        self._m = 0
        self._ef_construction = 0
        self._local = threading.local()

    # -- introspection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return (
            f"DenseIndex(n={len(self._ids)}, dim={self._dim}, m={self._m}, "
            f"ef_construction={self._ef_construction}, ef_search={self._ef_search})"
        )

    @property
    def dim(self) -> int | None:
        """Embedding dimensionality, or ``None`` if nothing has been built."""
        return self._dim

    @property
    def ids(self) -> list[str]:
        """Position-ordered chunk ids. Do not mutate."""
        return self._ids

    @property
    def ef_search(self) -> int:
        """Current default ``efSearch``."""
        return self._ef_search

    @property
    def is_empty(self) -> bool:
        """True when the index holds no vectors."""
        return not self._ids

    def config(self) -> dict[str, Any]:
        """Build parameters, for the ablation report and for save/load."""
        return {
            "n": len(self._ids),
            "dim": self._dim,
            "m": self._m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "metric": "inner_product",
        }

    # -- build ----------------------------------------------------------------

    def build(
        self,
        vectors: np.ndarray,
        ids: Sequence[str],
        *,
        m: int = 32,
        ef_construction: int = 80,
        normalize: bool = True,
    ) -> "DenseIndex":
        """Build the HNSW graph from a matrix of embeddings.

        Args:
            vectors: ``(n, dim)`` array. Cast to ``float32`` and, unless
                ``normalize`` is False, L2-normalised in a copy so inner product
                equals cosine similarity.
            ids: ``n`` chunk ids, positionally aligned with ``vectors``.
            m: HNSW out-degree. See the module docstring for the tradeoff.
            ef_construction: Build-time candidate breadth.
            normalize: Set False only when the caller guarantees unit norms;
                normalising twice is harmless but costs a full copy.

        Returns:
            ``self``, so building and searching can be chained.

        Raises:
            ValueError: If ``vectors`` is not 2-D or its length disagrees with
                ``ids``.
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vecs.shape}")
        if len(ids) != vecs.shape[0]:
            raise ValueError(
                f"ids/vectors length mismatch: {len(ids)} ids, {vecs.shape[0]} vectors"
            )

        self._dim = int(vecs.shape[1])
        self._ids = list(ids)
        self._m = int(m)
        self._ef_construction = int(ef_construction)
        self._local = threading.local()

        index = faiss.IndexHNSWFlat(self._dim, self._m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search

        if vecs.shape[0]:
            # faiss requires a contiguous float32 buffer it can read directly;
            # normalising in place on our own copy avoids a second allocation.
            data = np.ascontiguousarray(vecs)
            if normalize:
                data = data.copy() if data is vecs else data
                faiss.normalize_L2(data)
            index.add(data)

        self._index = index
        return self

    def set_ef(self, ef: int) -> None:
        """Change the default query-time ``efSearch``.

        The only recall/latency knob adjustable after build, which is what makes
        the tradeoff curve in :meth:`ef_curve` cheap to produce.

        Args:
            ef: New default breadth. Must be positive.

        Raises:
            ValueError: If ``ef`` is not positive.
        """
        if ef <= 0:
            raise ValueError(f"ef_search must be positive, got {ef}")
        self._ef_search = int(ef)
        if self._index is not None:
            self._index.hnsw.efSearch = self._ef_search

    # -- search ---------------------------------------------------------------

    def _query_buffer(self, n: int) -> np.ndarray:
        """Return a thread-local ``(n, dim)`` float32 scratch buffer."""
        buf = getattr(self._local, "buf", None)
        if buf is None or buf.shape[0] < n or buf.shape[1] != self._dim:
            buf = np.empty((max(n, 1), self._dim or 1), dtype=np.float32)
            self._local.buf = buf
        return buf[:n]

    def _prepare(self, q: np.ndarray, n_expected: int | None = None) -> np.ndarray:
        """Copy queries into the thread-local buffer and L2-normalise them."""
        arr = np.asarray(q, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"query must be 1-D or 2-D, got shape {arr.shape}")
        if self._dim is not None and arr.shape[1] != self._dim:
            raise ValueError(
                f"query dim {arr.shape[1]} != index dim {self._dim}"
            )
        if n_expected is not None and arr.shape[0] != n_expected:
            raise ValueError("query batch size mismatch")
        buf = self._query_buffer(arr.shape[0])
        buf[:] = arr
        faiss.normalize_L2(buf)
        return buf

    def _search_params(self, ef_search: int | None) -> Any:
        """Build per-call HNSW search parameters.

        Passing ``SearchParametersHNSW`` instead of mutating ``index.hnsw
        .efSearch`` keeps per-query overrides thread-safe: two concurrent
        callers asking for different ``ef`` values cannot clobber each other.
        """
        if ef_search is None:
            return None
        if ef_search <= 0:
            raise ValueError(f"ef_search must be positive, got {ef_search}")
        params = faiss.SearchParametersHNSW()
        params.efSearch = int(ef_search)
        return params

    def search(
        self,
        q: np.ndarray,
        k: int,
        ef_search: int | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve the ``k`` nearest chunks to one query vector.

        Args:
            q: Query embedding, shape ``(dim,)`` or ``(1, dim)``. Normalised
                internally, so callers may pass raw model output.
            k: Number of results wanted. Silently clamped to the index size.
            ef_search: Per-call breadth override. ``None`` uses the default.

        Returns:
            ``(chunk_id, cosine_similarity)`` pairs, best first. Empty list for
            an empty index or ``k <= 0``.
        """
        if self._index is None or not self._ids or k <= 0:
            return []
        k = min(int(k), len(self._ids))
        buf = self._prepare(q)
        params = self._search_params(ef_search)
        scores, labels = self._index.search(buf, k, params=params)
        row_s, row_l = scores[0], labels[0]
        out: list[tuple[str, float]] = []
        for pos in range(k):
            idx = int(row_l[pos])
            if idx < 0:  # faiss pads short result sets with -1
                break
            out.append((self._ids[idx], float(row_s[pos])))
        return out

    def search_batch(
        self,
        queries: np.ndarray,
        k: int,
        ef_search: int | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Retrieve for many queries at once.

        Batching lets faiss amortise graph traversal setup and use OpenMP across
        queries, which is why the benchmark harness uses this rather than a loop
        over :meth:`search`. Serving still uses :meth:`search` -- a live request
        has exactly one query and batching would only add latency.

        Args:
            queries: ``(n, dim)`` array of query embeddings.
            k: Results per query, clamped to the index size.
            ef_search: Per-call breadth override.

        Returns:
            One result list per query, in input order.
        """
        arr = np.asarray(queries, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self._index is None or not self._ids or k <= 0:
            return [[] for _ in range(arr.shape[0])]
        k = min(int(k), len(self._ids))
        buf = self._prepare(arr)
        params = self._search_params(ef_search)
        scores, labels = self._index.search(buf, k, params=params)
        results: list[list[tuple[str, float]]] = []
        for row_s, row_l in zip(scores, labels):
            hits: list[tuple[str, float]] = []
            for pos in range(k):
                idx = int(row_l[pos])
                if idx < 0:
                    break
                hits.append((self._ids[idx], float(row_s[pos])))
            results.append(hits)
        return results

    # -- evaluation -----------------------------------------------------------

    def reconstruct_all(self) -> np.ndarray:
        """Return the stored (normalised) vectors as an ``(n, dim)`` array.

        ``IndexHNSWFlat`` keeps the full vectors, so exact ground truth can be
        recomputed without the caller retaining a second copy of the matrix.
        """
        if self._index is None or not self._ids:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        return self._index.reconstruct_n(0, len(self._ids))

    def exact_search_batch(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Brute-force top-k positions, used as ANN ground truth.

        This is an offline analysis tool, not a serving path: it is ``O(n*dim)``
        per query and materialises the full score matrix.

        Args:
            queries: ``(n, dim)`` query embeddings; normalised internally.
            k: Neighbours per query.

        Returns:
            ``(n, k)`` array of index positions, best first.
        """
        if self._index is None or not self._ids or k <= 0:
            return np.zeros((0, 0), dtype=np.int64)
        k = min(int(k), len(self._ids))
        base = self.reconstruct_all()
        qs = np.asarray(queries, dtype=np.float32)
        if qs.ndim == 1:
            qs = qs.reshape(1, -1)
        qs = np.ascontiguousarray(qs.copy())
        faiss.normalize_L2(qs)
        sims = qs @ base.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(sims, part, axis=1).argsort(axis=1)[:, ::-1]
        return np.take_along_axis(part, order, axis=1)

    def ef_curve(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef_values: Sequence[int] = (16, 32, 64, 128, 256),
    ) -> list[EfCurvePoint]:
        """Measure recall and latency across ``efSearch`` values.

        Produces the tradeoff curve the write-up publishes. Recall is measured
        against exact inner-product ground truth computed by
        :meth:`exact_search_batch`, and latency is measured one query at a time
        because that is how the index is actually used in serving.

        Args:
            queries: ``(n, dim)`` evaluation queries.
            k: Cutoff for both recall and retrieval.
            ef_values: ``efSearch`` settings to sweep.

        Returns:
            One :class:`EfCurvePoint` per value, in the order given.
        """
        import time

        qs = np.asarray(queries, dtype=np.float32)
        if qs.ndim == 1:
            qs = qs.reshape(1, -1)
        if self._index is None or not self._ids or not len(qs):
            return []
        k = min(int(k), len(self._ids))
        truth = self.exact_search_batch(qs, k)
        gold = [set(int(x) for x in row) for row in truth]
        pos_of = {cid: i for i, cid in enumerate(self._ids)}

        points: list[EfCurvePoint] = []
        for ef in ef_values:
            hits = 0
            timings: list[float] = []
            for i in range(qs.shape[0]):
                t0 = time.perf_counter()
                res = self.search(qs[i], k, ef_search=ef)
                timings.append((time.perf_counter() - t0) * 1000.0)
                got = {pos_of[cid] for cid, _ in res}
                hits += len(got & gold[i])
            arr = np.asarray(timings)
            points.append(
                EfCurvePoint(
                    ef_search=int(ef),
                    recall=hits / float(k * qs.shape[0]),
                    p50_ms=float(np.percentile(arr, 50)),
                    p70_ms=float(np.percentile(arr, 70)),
                    p100_ms=float(arr.max()),
                )
            )
        return points

    # -- persistence ----------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Persist the faiss graph and the id mapping.

        Args:
            directory: Created if absent. Written files are ``dense.faiss``,
                ``dense_ids.json`` and ``dense_meta.json``.

        Returns:
            The directory written to.

        Raises:
            RuntimeError: If called before :meth:`build`.
        """
        if self._index is None:
            raise RuntimeError("cannot save a DenseIndex that has not been built")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILE))
        (path / _IDS_FILE).write_text(json.dumps(self._ids), encoding="utf-8")
        (path / _META_FILE).write_text(json.dumps(self.config()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "DenseIndex":
        """Restore an index written by :meth:`save`.

        Args:
            directory: Directory containing the three saved files.

        Returns:
            A ready-to-search :class:`DenseIndex`.

        Raises:
            FileNotFoundError: If any expected file is missing.
            ValueError: If the id mapping and the faiss index disagree in size,
                which means the two files came from different builds.
        """
        path = Path(directory)
        meta = json.loads((path / _META_FILE).read_text(encoding="utf-8"))
        ids = json.loads((path / _IDS_FILE).read_text(encoding="utf-8"))
        index = faiss.read_index(str(path / _INDEX_FILE))
        if index.ntotal != len(ids):
            raise ValueError(
                f"corrupt index: {index.ntotal} vectors but {len(ids)} ids"
            )
        obj = cls(ef_search=int(meta.get("ef_search", 64)))
        obj._index = index
        obj._ids = ids
        obj._dim = int(meta.get("dim") or index.d)
        obj._m = int(meta.get("m", 0))
        obj._ef_construction = int(meta.get("ef_construction", 0))
        index.hnsw.efSearch = obj._ef_search
        return obj
