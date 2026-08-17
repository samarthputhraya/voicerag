"""Lexical BM25 retrieval.

Why keep a sparse index at all when the dense one is this fast? Because voice
queries are full of things embeddings quietly smooth away: product codes, proper
nouns, numbers, and words the embedding model never saw. On MS MARCO the two
retrievers fail on *different* queries, which is exactly the condition under
which fusion helps (see :mod:`voicerag.index.hybrid`). Sparse retrieval also
gives the guardrail layer an independent signal -- when BM25 and the dense index
disagree completely, that disagreement is itself evidence the corpus may not
contain the answer.

Implementation choices worth knowing about:

* **bm25s, not rank_bm25.** ``rank_bm25`` scores every document in Python on
  every query, which is ``O(n)`` interpreted work. ``bm25s`` precomputes the
  full term-document score matrix as a sparse CSC array, so a query is a handful
  of column slices plus a top-k selection in numpy.
* **PyStemmer.** Snowball stemming collapses ``running``/``ran``/``runs``,
  materially helping recall on spoken questions, and costs microseconds.
* **Memory-mapped load.** :meth:`SparseIndex.load` defaults to ``mmap=True`` so
  the score arrays are paged in on demand rather than read up front. On a
  200k-passage corpus that turns a multi-hundred-megabyte read into a near-
  instant open, which matters because this service is expected to cold-start
  quickly and a mapped index also lets several worker processes share one copy
  of the data in page cache.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import bm25s
import numpy as np

__all__ = ["SparseIndex"]

_IDS_FILE = "sparse_ids.json"
_CONFIG_FILE = "sparse_config.json"

#: Identical to bm25s' default ``token_pattern``. Duplicated here (rather than
#: imported) so that index-time and query-time tokenisation are provably the
#: same thing -- ``test_index.py`` asserts parity against ``bm25s.tokenize``.
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def _load_stopwords(spec: str | Sequence[str]) -> frozenset[str]:
    """Resolve a stopword spec the same way bm25s does."""
    from bm25s.tokenization import _infer_stopwords

    return frozenset(_infer_stopwords(spec))


class SparseIndex:
    """BM25 index over chunk texts, with the same API shape as ``DenseIndex``.

    Args:
        k1: BM25 term-frequency saturation.
        b: BM25 length normalisation.
        method: bm25s scoring variant. ``"lucene"`` matches the Lucene/Elastic
            formulation, which is what published MS MARCO BM25 baselines use --
            worth keeping so our numbers are comparable to the literature.
        stemmer: Snowball language name, or ``None`` to disable stemming.
        stopwords: bm25s stopword spec (``"en"``, ``"english"``, a list, or
            ``[]`` for none).

    Attributes:
        ids: Position-ordered chunk ids.
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        method: str = "lucene",
        stemmer: str | None = "english",
        stopwords: str | Sequence[str] = "en",
    ) -> None:
        self._k1 = float(k1)
        self._b = float(b)
        self._method = method
        self._stemmer_name = stemmer
        self._stopwords_spec = stopwords
        self._stopwords = _load_stopwords(stopwords)
        self._stemmer = self._make_stemmer(stemmer)
        self._bm25: bm25s.BM25 | None = None
        self._ids: list[str] = []

    @staticmethod
    def _make_stemmer(name: str | None) -> Any:
        """Instantiate a PyStemmer stemmer, or ``None`` when disabled."""
        if not name:
            return None
        import Stemmer

        return Stemmer.Stemmer(name)

    # -- introspection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return (
            f"SparseIndex(n={len(self._ids)}, method={self._method!r}, "
            f"k1={self._k1}, b={self._b}, stemmer={self._stemmer_name!r})"
        )

    @property
    def ids(self) -> list[str]:
        """Position-ordered chunk ids. Do not mutate."""
        return self._ids

    @property
    def is_empty(self) -> bool:
        """True when the index holds no documents."""
        return not self._ids

    def config(self) -> dict[str, Any]:
        """Build parameters, for the ablation report and for save/load."""
        return {
            "n": len(self._ids),
            "k1": self._k1,
            "b": self._b,
            "method": self._method,
            "stemmer": self._stemmer_name,
            "stopwords": (
                self._stopwords_spec
                if isinstance(self._stopwords_spec, str)
                else list(self._stopwords_spec)
            ),
        }

    # -- tokenisation ---------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Tokenise one string exactly as the index was built.

        Reimplemented rather than calling ``bm25s.tokenize`` on the hot path
        because that function allocates a vocabulary dict, a progress bar and
        several intermediate lists per call -- overhead that is invisible at
        ingest but is a real share of a sub-millisecond query. The pipeline
        (lowercase, regex split, drop stopwords, stem) is byte-for-byte the same
        and is pinned by a parity test.

        Args:
            text: Raw query or document text.

        Returns:
            Stemmed, stopword-filtered tokens. Empty list for empty input.
        """
        toks = [t for t in _TOKEN_RE.findall(text.lower()) if t not in self._stopwords]
        if not toks or self._stemmer is None:
            return toks
        return self._stemmer.stemWords(toks)

    # -- build ----------------------------------------------------------------

    def build(self, texts: Iterable[str], ids: Sequence[str]) -> "SparseIndex":
        """Tokenise a corpus and build the BM25 score matrix.

        Args:
            texts: Document texts, positionally aligned with ``ids``.
            ids: Chunk ids.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``texts`` and ``ids`` differ in length.
        """
        corpus = list(texts)
        if len(corpus) != len(ids):
            raise ValueError(
                f"ids/texts length mismatch: {len(ids)} ids, {len(corpus)} texts"
            )
        self._ids = list(ids)
        if not corpus:
            self._bm25 = None
            return self

        tokens = bm25s.tokenize(
            corpus,
            stopwords=self._stopwords_spec,
            stemmer=self._stemmer,
            show_progress=False,
        )
        bm25 = bm25s.BM25(k1=self._k1, b=self._b, method=self._method)
        bm25.index(tokens, show_progress=False)
        self._bm25 = bm25
        return self

    # -- search ---------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int,
        *,
        drop_zero: bool = True,
    ) -> list[tuple[str, float]]:
        """Retrieve the ``k`` best-matching chunks for a query string.

        Args:
            query: Raw query text; tokenised with :meth:`tokenize`.
            k: Number of results wanted, clamped to the corpus size.
            drop_zero: Discard results scoring exactly 0. A zero BM25 score
                means the document shares no query term at all, so it is padding
                rather than a match -- and passing padding downstream would
                corrupt the guardrail layer's abstention decision, which reads
                the sparse score as evidence.

        Returns:
            ``(chunk_id, bm25_score)`` pairs, best first. Empty list when the
            index is empty, ``k <= 0``, or the query has no in-vocabulary terms.
        """
        if self._bm25 is None or not self._ids or k <= 0:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        k = min(int(k), len(self._ids))
        # n_threads=1: a single query is far too small to amortise the cost of
        # dispatching to a thread pool, and the hybrid retriever is already
        # running this call on its own worker thread.
        docs, scores = self._bm25.retrieve(
            [tokens], k=k, show_progress=False, n_threads=1
        )
        out: list[tuple[str, float]] = []
        for pos in range(k):
            score = float(scores[0][pos])
            if drop_zero and score <= 0.0:
                break
            out.append((self._ids[int(docs[0][pos])], score))
        return out

    def search_batch(
        self,
        queries: Sequence[str],
        k: int,
        *,
        drop_zero: bool = True,
        n_threads: int = 0,
    ) -> list[list[tuple[str, float]]]:
        """Retrieve for many queries at once, for benchmarking and evaluation.

        Args:
            queries: Raw query strings.
            k: Results per query, clamped to the corpus size.
            drop_zero: See :meth:`search`.
            n_threads: bm25s worker threads; ``0`` lets bm25s decide.

        Returns:
            One result list per query, in input order.
        """
        if self._bm25 is None or not self._ids or k <= 0 or not len(queries):
            return [[] for _ in queries]
        k = min(int(k), len(self._ids))
        tokenised = [self.tokenize(q) for q in queries]
        # bm25s cannot represent a query with no terms, so those are held out
        # and re-inserted as empty results afterwards.
        keep = [i for i, t in enumerate(tokenised) if t]
        results: list[list[tuple[str, float]]] = [[] for _ in queries]
        if not keep:
            return results
        docs, scores = self._bm25.retrieve(
            [tokenised[i] for i in keep],
            k=k,
            show_progress=False,
            n_threads=n_threads,
        )
        for row, qi in enumerate(keep):
            hits: list[tuple[str, float]] = []
            for pos in range(k):
                score = float(scores[row][pos])
                if drop_zero and score <= 0.0:
                    break
                hits.append((self._ids[int(docs[row][pos])], score))
            results[qi] = hits
        return results

    # -- persistence ----------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Persist the BM25 arrays, the vocabulary and the id mapping.

        Args:
            directory: Created if absent.

        Returns:
            The directory written to.

        Raises:
            RuntimeError: If called before :meth:`build`.
        """
        if self._bm25 is None:
            raise RuntimeError("cannot save a SparseIndex that has not been built")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(path), show_progress=False)
        (path / _IDS_FILE).write_text(json.dumps(self._ids), encoding="utf-8")
        (path / _CONFIG_FILE).write_text(json.dumps(self.config()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path, *, mmap: bool = True) -> "SparseIndex":
        """Restore an index written by :meth:`save`.

        Args:
            directory: Directory containing the saved files.
            mmap: Memory-map the score arrays instead of reading them into RAM.
                Default True: it makes cold start effectively independent of
                corpus size and lets multiple workers share one physical copy.
                Pass False for a long-lived batch job that will touch the whole
                matrix anyway and would rather pay the read once up front.

        Returns:
            A ready-to-search :class:`SparseIndex`.

        Raises:
            ValueError: If the id mapping disagrees with the stored corpus size.
        """
        path = Path(directory)
        cfg = json.loads((path / _CONFIG_FILE).read_text(encoding="utf-8"))
        ids = json.loads((path / _IDS_FILE).read_text(encoding="utf-8"))
        obj = cls(
            k1=float(cfg.get("k1", 1.5)),
            b=float(cfg.get("b", 0.75)),
            method=cfg.get("method", "lucene"),
            stemmer=cfg.get("stemmer", "english"),
            stopwords=cfg.get("stopwords", "en"),
        )
        if int(cfg.get("n", len(ids))) != len(ids):
            raise ValueError("corrupt index: id count disagrees with saved config")
        obj._bm25 = bm25s.BM25.load(str(path), mmap=mmap, show_progress=False)
        obj._ids = ids
        return obj
