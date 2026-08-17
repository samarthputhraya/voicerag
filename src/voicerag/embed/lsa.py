"""Hashing + truncated-SVD (LSA) embeddings: a real dense model, zero downloads.

Why this exists at all, given that we ship a neural static embedder:

* **Cold start.** A fresh clone with no network, no weights and no HF cache can
  still ingest a corpus, build an index and answer queries end to end. Nothing
  in the test suite or the chunking ablation is gated on a download.
* **An honest baseline row.** Every retrieval ablation needs a
  "no-neural-model" control. Hashing TF-IDF -> SVD is the classic one, and it
  is strong enough on MS MARCO-style keyword queries to make the neural gain
  meaningful rather than assumed.
* **Determinism.** Fully seeded, so a reported ablation number is reproducible
  bit for bit on the same machine.

Design notes that matter:

* Features are the **hashing trick** over word n-grams *and* character
  n-grams. There is no vocabulary, so ``fit`` never has to hold one and
  transform has no OOV path -- an important property for ASR output, where
  misrecognised tokens ("diabetis") are unseen by construction and character
  n-grams still align them with the right passages.
* Character n-grams are hashed with a **vectorised rolling polynomial hash**
  over the UTF-8 bytes. A Python loop over ``len(text) * (nmax - nmin + 1)``
  substrings is the obvious implementation and is roughly two orders of
  magnitude slower; corpus vectorisation would dominate ingest.
* Reduction is **randomised SVD** rather than an exact one: the term-document
  matrix is 30k+ columns wide and we only want a few hundred components, so
  the exact factorisation would be pure waste.
"""

from __future__ import annotations

import pickle
import re
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse

from .base import BaseEmbedder

__all__ = ["HashingLSAEmbedder", "DEFAULT_LSA_DIM", "DEFAULT_N_FEATURES"]

#: Matches the dimensionality we benchmarked FAISS HNSW at (0.63 ms p50 for
#: 200k vectors), so the LSA baseline and the neural path share index settings.
DEFAULT_LSA_DIM = 256

#: Hashed feature space. Power of two so the modulo is a bit-mask. 2**15 keeps
#: the SVD component matrix at ``dim * n_features * 4 B`` = 32 MB for dim 256,
#: which pickles and loads fast; collisions at this width are absorbed by the
#: SVD, which is itself a lossy projection.
DEFAULT_N_FEATURES = 1 << 15

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.ASCII)

# 64-bit mixing constants (splitmix64 / golden-ratio derived). uint64 maths in
# numpy wraps modulo 2**64, which is exactly the behaviour a hash wants.
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_CHAR_BASE = np.uint64(131)


class HashingLSAEmbedder(BaseEmbedder):
    """Fit-then-transform dense embeddings from hashed n-grams via truncated SVD.

    The pipeline is the textbook one, with the vocabulary replaced by a hash:

    1. Hash word n-grams and character n-grams into ``n_features`` buckets.
    2. Weight with sublinear TF and smoothed IDF, then L2-normalise rows.
    3. Project onto the top ``dim`` right singular vectors learned at fit time.
    4. L2-normalise again (done by :class:`~voicerag.embed.base.BaseEmbedder`).

    Step 2's IDF is learned on the fit corpus and frozen, so query and document
    vectors live in the same space -- re-estimating IDF per query batch is a
    classic and silent source of train/serve skew.

    Args:
        dim: Output dimensionality (number of SVD components).
        n_features: Size of the hashed feature space. Must be a power of two.
        word_ngrams: Inclusive ``(min, max)`` word n-gram range. ``(1, 2)``
            adds bigrams, which recover a little word order without the
            combinatorial blow-up of trigrams.
        char_ngrams: Inclusive ``(min, max)`` character n-gram range, or
            ``None`` to disable. Character grams are what make this robust to
            ASR spelling errors and to morphology.
        sublinear_tf: Apply ``1 + log(tf)`` instead of raw counts. A term
            repeated ten times is not ten times more topical.
        n_oversamples: Extra random projections in the randomised SVD; improves
            accuracy of the trailing components at negligible cost.
        n_iter: Power iterations in the randomised SVD. TF-IDF spectra decay
            slowly, so a couple of iterations measurably sharpen the subspace.
        seed: Seed for the random projection. Fixed seed -> identical model.
        batch_size: Default texts per transform batch.

    Raises:
        ValueError: If ``n_features`` is not a power of two or any range is
            malformed.
    """

    name = "lsa"

    def __init__(
        self,
        dim: int = DEFAULT_LSA_DIM,
        *,
        n_features: int = DEFAULT_N_FEATURES,
        word_ngrams: tuple[int, int] = (1, 2),
        char_ngrams: tuple[int, int] | None = (3, 5),
        sublinear_tf: bool = True,
        n_oversamples: int = 10,
        n_iter: int = 4,
        seed: int = 0,
        batch_size: int = 256,
    ) -> None:
        super().__init__(batch_size=batch_size)
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_features < 2 or (n_features & (n_features - 1)) != 0:
            raise ValueError(
                f"n_features must be a power of two >= 2, got {n_features}"
            )
        _check_range("word_ngrams", word_ngrams, floor=1)
        if char_ngrams is not None:
            _check_range("char_ngrams", char_ngrams, floor=1)

        self._dim = int(dim)
        self.n_features = int(n_features)
        self.word_ngrams = (int(word_ngrams[0]), int(word_ngrams[1]))
        self.char_ngrams = (
            None if char_ngrams is None else (int(char_ngrams[0]), int(char_ngrams[1]))
        )
        self.sublinear_tf = bool(sublinear_tf)
        self.n_oversamples = int(n_oversamples)
        self.n_iter = int(n_iter)
        self.seed = int(seed)

        self._mask = np.uint64(self.n_features - 1)
        self.idf_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.n_docs_fit_: int = 0

    # ---------------------------------------------------------------- public

    @property
    def dim(self) -> int:
        """Number of SVD components, i.e. the output dimensionality."""
        return self._dim

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` (or :meth:`load`) has produced a projection."""
        return self.components_ is not None

    def fit(
        self,
        corpus: Sequence[str],
        *,
        sample_size: int | None = None,
    ) -> "HashingLSAEmbedder":
        """Learn IDF weights and the SVD projection from a corpus.

        Args:
            corpus: Documents (passages or chunks) to fit on. Order does not
                affect the result beyond floating-point summation order.
            sample_size: If given and smaller than the corpus, fit on a
                deterministic random subsample of this many documents. Fitting
                on a sample is the intended path for large corpora: the
                projection converges long before 200k passages, while the
                sparse matrix build is linear in corpus size.

        Returns:
            ``self``, so ``embedder = HashingLSAEmbedder().fit(docs)`` works.

        Raises:
            ValueError: If the corpus is empty, or has fewer documents than
                ``dim`` (a rank-``dim`` subspace cannot be estimated from
                fewer than ``dim`` samples).
        """
        docs = [d if isinstance(d, str) else str(d) for d in corpus]
        if not docs:
            raise ValueError("cannot fit HashingLSAEmbedder on an empty corpus")

        rng = np.random.default_rng(self.seed)
        if sample_size is not None and 0 < sample_size < len(docs):
            picks = rng.choice(len(docs), size=int(sample_size), replace=False)
            docs = [docs[i] for i in np.sort(picks)]

        if len(docs) < self._dim:
            raise ValueError(
                f"need at least dim={self._dim} documents to fit a rank-{self._dim} "
                f"projection, got {len(docs)}; lower `dim` or supply more text"
            )

        counts = self._count_matrix(docs)
        df = np.diff(counts.tocsc().indptr).astype(np.float64)
        n = float(counts.shape[0])
        # Smoothed IDF (the +1s act as a virtual document containing every
        # feature), so an unseen bucket can never produce a division by zero
        # at query time.
        self.idf_ = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
        self.n_docs_fit_ = len(docs)

        weighted = self._weight(counts)
        components, singular = _randomised_svd(
            weighted,
            n_components=self._dim,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            rng=rng,
        )
        self.components_ = np.ascontiguousarray(components, dtype=np.float32)
        self.singular_values_ = singular.astype(np.float32)
        return self

    def fit_transform(
        self, corpus: Sequence[str], *, sample_size: int | None = None
    ) -> np.ndarray:
        """Fit on ``corpus`` and return its embeddings.

        Args:
            corpus: Documents to fit on and encode.
            sample_size: See :meth:`fit`. The returned matrix always covers the
                full ``corpus``, even when the fit used a subsample.

        Returns:
            L2-normalised ``float32`` array of shape ``(len(corpus), dim)``.
        """
        self.fit(corpus, sample_size=sample_size)
        return self.encode(list(corpus))

    def explained_variance_ratio(self) -> float:
        """Fraction of the fit matrix's squared Frobenius mass the SVD retains.

        Cheap to compute exactly here: rows of the weighted matrix are
        L2-normalised, so its total energy is exactly the number of fit
        documents and the ratio is ``sum(s**2) / n_docs``. Useful for choosing
        ``dim`` -- the value is monotone in ``dim`` and its knee is a
        defensible place to stop.

        Returns:
            A value in ``[0, 1]``; ``0.0`` if the model is unfitted.
        """
        if self.singular_values_ is None or not self.n_docs_fit_:
            return 0.0
        kept = float(np.sum(self.singular_values_.astype(np.float64) ** 2))
        return min(1.0, kept / float(self.n_docs_fit_))

    def save(self, path: str | Path) -> Path:
        """Pickle the fitted model to disk.

        Args:
            path: Destination file. Parent directories are created.

        Returns:
            The resolved path written.

        Raises:
            RuntimeError: If the model is not fitted -- saving an unfitted
                model is always a mistake and would fail confusingly on load.
        """
        self._require_fitted("save")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "HashingLSAEmbedder":
        """Load a model previously written by :meth:`save`.

        Args:
            path: File to read.

        Returns:
            The restored embedder.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            TypeError: If the pickle does not contain a ``HashingLSAEmbedder``.
        """
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"no LSA model at {src}")
        with src.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{src} contains {type(obj).__name__}, not {cls.__name__}")
        return obj

    # ------------------------------------------------------------- internals

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        self._require_fitted("encode")
        assert self.components_ is not None  # narrowed by _require_fitted
        weighted = self._weight(self._count_matrix(texts))
        return weighted @ self.components_.T

    def _require_fitted(self, action: str) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"HashingLSAEmbedder must be fitted before {action}(): call "
                "fit(corpus) on your passages, or load() a saved model"
            )

    def _weight(self, counts: sparse.csr_matrix) -> sparse.csr_matrix:
        """Apply sublinear TF, frozen IDF and L2 row normalisation."""
        assert self.idf_ is not None
        out = counts.astype(np.float32)
        if self.sublinear_tf and out.nnz:
            np.log(out.data, out=out.data)
            out.data += 1.0
        if out.nnz:
            out.data *= self.idf_[out.indices]
        norms = np.sqrt(np.asarray(out.multiply(out).sum(axis=1))).ravel()
        norms[norms <= 0.0] = 1.0
        return sparse.diags(1.0 / norms, dtype=np.float32) @ out

    def _count_matrix(self, texts: Sequence[str]) -> sparse.csr_matrix:
        """Build a raw hashed-count CSR matrix of shape ``(len(texts), F)``."""
        indices: list[np.ndarray] = []
        data: list[np.ndarray] = []
        indptr = np.zeros(len(texts) + 1, dtype=np.int64)
        for i, text in enumerate(texts):
            buckets = self._hash_buckets(text)
            if buckets.size:
                uniq, cnt = np.unique(buckets, return_counts=True)
                indices.append(uniq.astype(np.int32))
                data.append(cnt.astype(np.float32))
                indptr[i + 1] = indptr[i] + uniq.size
            else:
                indptr[i + 1] = indptr[i]
        idx = (
            np.concatenate(indices)
            if indices
            else np.zeros(0, dtype=np.int32)
        )
        val = np.concatenate(data) if data else np.zeros(0, dtype=np.float32)
        return sparse.csr_matrix(
            (val, idx, indptr), shape=(len(texts), self.n_features), dtype=np.float32
        )

    def _hash_buckets(self, text: str) -> np.ndarray:
        """Hash one document's word and character n-grams into bucket ids.

        Returns:
            ``uint64`` array of bucket indices, with repeats (the caller counts
            them). Empty when the text has no usable content.
        """
        lowered = text.lower()
        parts: list[np.ndarray] = []

        tokens = _WORD_RE.findall(lowered)
        if tokens:
            wmin, wmax = self.word_ngrams
            word_hashes: list[int] = []
            for n in range(wmin, min(wmax, len(tokens)) + 1):
                if n == 1:
                    word_hashes.extend(zlib.crc32(t.encode("utf-8"), 1) for t in tokens)
                else:
                    word_hashes.extend(
                        zlib.crc32(" ".join(tokens[i : i + n]).encode("utf-8"), n)
                        for i in range(len(tokens) - n + 1)
                    )
            if word_hashes:
                parts.append(_finalise(np.asarray(word_hashes, dtype=np.uint64)))

        if self.char_ngrams is not None:
            raw = np.frombuffer(lowered.encode("utf-8", "ignore"), dtype=np.uint8)
            if raw.size:
                buf = raw.astype(np.uint64) + np.uint64(1)
                cmin, cmax = self.char_ngrams
                for n in range(cmin, min(cmax, buf.size) + 1):
                    parts.append(_finalise(_char_gram_hashes(buf, n)))

        if not parts:
            return np.zeros(0, dtype=np.uint64)
        return np.concatenate(parts) & self._mask


def _check_range(label: str, rng: tuple[int, int], *, floor: int) -> None:
    """Validate an inclusive ``(min, max)`` n-gram range."""
    lo, hi = rng
    if lo < floor or hi < lo:
        raise ValueError(f"{label} must satisfy {floor} <= min <= max, got {rng}")


def _char_gram_hashes(buf: np.ndarray, n: int) -> np.ndarray:
    """Vectorised rolling polynomial hash of every length-``n`` byte window.

    Evaluated as ``sum(byte[k] * BASE**k) mod 2**64`` with the powers folded in
    by a single matrix-vector product, which is why this is fast: one BLAS-free
    ``uint64`` dot per n-gram length instead of a Python loop per substring.
    Reduction modulo ``2**64`` is a ring homomorphism, so pre-reducing the
    powers is exact, and unsigned overflow in numpy wraps by definition.

    Args:
        buf: ``uint64`` view of the document bytes (already offset by 1 so a
            NUL byte cannot be confused with padding).
        n: Window length.

    Returns:
        ``uint64`` array of ``len(buf) - n + 1`` hashes, with ``n`` mixed in so
        that 3-grams and 4-grams do not systematically collide.
    """
    # Polynomial hashing relies on wraparound at 2**64, which is well-defined
    # here but which numpy reports as an overflow. Silence it locally rather
    # than globally: an unexpected overflow anywhere else should still surface.
    with np.errstate(over="ignore"):
        windows = np.lib.stride_tricks.sliding_window_view(buf, n)
        powers = np.ones(n, dtype=np.uint64)
        for k in range(1, n):
            powers[k] = powers[k - 1] * _CHAR_BASE
        hashes = (windows * powers).sum(axis=1, dtype=np.uint64)
        return hashes ^ (np.uint64(n) * _MIX_B)


def _finalise(hashes: np.ndarray) -> np.ndarray:
    """Avalanche step so low bits are usable as bucket indices.

    CRC32 and polynomial hashes have poor low-bit dispersion; masking them
    straight into buckets clusters features. Two splitmix64-style multiply-xor
    rounds fix that for the price of a few nanoseconds per gram.
    """
    h = hashes.astype(np.uint64, copy=False)
    h = (h ^ (h >> np.uint64(30))) * _MIX_A
    h = (h ^ (h >> np.uint64(27))) * _MIX_B
    return h ^ (h >> np.uint64(31))


def _randomised_svd(
    mat: sparse.csr_matrix,
    *,
    n_components: int,
    n_oversamples: int,
    n_iter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated SVD by randomised range finding (Halko, Martinsson, Tropp).

    Projects the matrix onto a small random subspace, refines that subspace
    with ``n_iter`` power iterations (re-orthonormalising each time, otherwise
    float32 rounding collapses the basis onto the leading singular vector), and
    takes an exact SVD of the resulting tiny ``k x n_features`` matrix.

    Args:
        mat: Sparse ``(n_docs, n_features)`` weighted term matrix.
        n_components: Components to keep.
        n_oversamples: Extra random directions beyond ``n_components``.
        n_iter: Power iterations.
        rng: Seeded generator; the only source of randomness in the fit.

    Returns:
        ``(components, singular_values)`` where ``components`` has shape
        ``(n_components, n_features)`` with rows in descending singular-value
        order and a deterministic sign convention.
    """
    n_rows, n_cols = mat.shape
    size = min(n_components + n_oversamples, n_rows, n_cols)
    proj = rng.standard_normal(size=(n_cols, size)).astype(np.float32)

    basis, _ = np.linalg.qr(mat @ proj)
    for _ in range(n_iter):
        basis, _ = np.linalg.qr(mat.T @ basis)
        basis, _ = np.linalg.qr(mat @ basis)

    small = np.asarray(basis.T @ mat, dtype=np.float32)
    _, singular, right = np.linalg.svd(small, full_matrices=False)
    keep = min(n_components, right.shape[0])
    components = right[:keep]
    singular = singular[:keep]

    if keep < n_components:  # pathologically low-rank corpus: pad with zeros
        components = np.vstack(
            [components, np.zeros((n_components - keep, n_cols), dtype=np.float32)]
        )
        singular = np.concatenate(
            [singular, np.zeros(n_components - keep, dtype=np.float32)]
        )

    return _sign_flip(components), singular


def _sign_flip(components: np.ndarray) -> np.ndarray:
    """Fix the SVD's sign ambiguity so refits are byte-comparable.

    ``(u, v)`` and ``(-u, -v)`` are equally valid singular vector pairs; LAPACK
    picks either depending on rounding. Forcing the largest-magnitude entry of
    each component to be positive makes the fitted model reproducible, which
    matters because we assert on it in tests and pickle it to disk.
    """
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), pivots])
    signs[signs == 0] = 1.0
    return components * signs[:, None]
