"""MSMARCO-XI loading, deduplication and qrels construction.

The dataset is ``ai4bharat/MSMARCO-XI``: MS MARCO translated into 24 Indic
languages, where every row keeps the **original human-written English passages**
alongside the translated query. That pairing is what lets one English index
serve a Hindi voice demo against identical ground truth.

The full repository is 55.6 GB, which we never download. We pin a single
validation shard (``validation/hinval.parquet``, 440 MB) and stream it, so
ingest starts producing documents in seconds and RAM stays flat.

Row schema (10 columns, verified against the HF datasets-server)::

    source_lang   str                     e.g. "eng_Latn"
    target_lang   str                     FLORES-200 code, e.g. "hin_Deva"
    meta          struct                  free-form provenance
    Answer        str                     answer in the target language
    query_id      int64
    query_type    str                     DESCRIPTION|NUMERIC|ENTITY|LOCATION|PERSON
    passages      struct
        English_passages     list<str>    original MS MARCO English text
        Translated_passages  list<str>
        is_selected          list<int64>  0/1, index-aligned with the passages
    Eng_Query     str
    Eng_Answer    str
    query         str                     query in the target language

**The invariant this module exists to protect.** ``is_selected == 1`` marks the
human-judged gold passage, and the same passage text recurs across queries --
MS MARCO's candidate pools overlap heavily. If the corpus is deduplicated by
content but the qrels still reference per-row positions, a gold label can point
at a passage that no longer exists under that identity, and every retrieval
metric silently becomes wrong-but-plausible. So the document id **is** the
content hash: dedup and qrels are computed from the same function, and a
duplicated gold passage necessarily resolves to the surviving copy.
:func:`passage_id` is that function, and ``tests/test_eval.py`` proves the
invariant holds.

Rows whose passages are *all* ``is_selected == 0`` carry
``Answer == "No Answer Present."``. Those are not noise -- they are a free,
human-labelled set of *the corpus cannot answer this*, which is exactly what the
abstention guardrail needs to be evaluated against
(:mod:`eval.abstention_eval`).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

__all__ = [
    "DEFAULT_REPO_ID",
    "DEFAULT_SHARD",
    "NO_ANSWER_SENTINEL",
    "Corpus",
    "DatasetUnavailable",
    "Query",
    "abstention_set",
    "build_corpus",
    "dump_jsonl",
    "load_jsonl",
    "load_msmarco_xi",
    "passage_id",
    "synthetic_corpus",
    "synthetic_rows",
    "to_documents",
    "to_qrels",
    "to_queries",
]

#: Exact Hugging Face repository id. Getting this wrong is the single most
#: common way to end up evaluating on the wrong corpus.
DEFAULT_REPO_ID = "ai4bharat/MSMARCO-XI"

#: The one shard we pin. 461,888,616 bytes, 97,941 validation rows, Hindi
#: (``hin_Deva``) queries over the original English passages.
DEFAULT_SHARD = "validation/hinval.parquet"

#: Verbatim marker MS MARCO uses when no candidate passage answers the query.
NO_ANSWER_SENTINEL = "No Answer Present."

_WS = re.compile(r"\s+")


class DatasetUnavailable(RuntimeError):
    """The real dataset could not be loaded on this machine.

    Raised instead of the underlying ``ImportError``/network error so callers
    can fall back to :func:`synthetic_rows` deliberately rather than by
    catching everything. The message always names the concrete fix, because
    this is the first error a fresh clone hits.
    """


# --- identity -----------------------------------------------------------------


def passage_id(text: str) -> str:
    """Content-addressed id for one passage.

    Args:
        text: Raw passage text, exactly as it appears in the shard.

    Returns:
        ``"p:"`` followed by 16 hex characters of BLAKE2b over the
        whitespace-normalised, case-folded text.

    Normalisation before hashing is deliberate. MS MARCO's candidate pools
    contain the same passage with different leading whitespace and occasional
    case drift from the crawl; treating those as distinct documents would
    inflate the corpus, split a gold label across two ids and quietly depress
    every recall number. The stored :class:`~voicerag.chunking.base.Document`
    keeps the first-seen verbatim text -- normalisation affects identity only,
    never what gets indexed or cited.

    64 bits of digest over a corpus of ~2e5 passages gives a collision
    probability around 1e-9, which is far below the noise floor of the metrics
    it protects.
    """
    norm = _WS.sub(" ", text).strip().casefold()
    return "p:" + hashlib.blake2b(norm.encode("utf-8"), digest_size=8).hexdigest()


# --- typed views over a row ---------------------------------------------------


@dataclass(slots=True, frozen=True)
class Query:
    """One evaluation query in both languages, with its gold answer.

    Attributes:
        query_id: MS MARCO query id; the key of the qrels mapping.
        eng_query: Original English query. Used for English retrieval eval.
        indic_query: Same query in ``target_lang``. Drives the cross-lingual
            demo: it is asked by voice, and retrieval still runs over English
            passages.
        target_lang: FLORES-200 code of ``indic_query``.
        query_type: MS MARCO's own type label. Kept because per-type breakdowns
            are where retrieval regressions actually show up (NUMERIC queries
            behave nothing like DESCRIPTION ones).
        answer: Gold answer in the target language.
        eng_answer: Gold answer in English.
        answerable: False when the annotators found no passage that answers it.
    """

    query_id: int
    eng_query: str
    indic_query: str = ""
    target_lang: str = ""
    query_type: str = ""
    answer: str = ""
    eng_answer: str = ""
    answerable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Corpus:
    """Everything one pass over the shard produces.

    Attributes:
        documents: Deduplicated passages, in first-seen order.
        qrels: ``query_id -> {passage_id, ...}`` for gold passages only.
        queries: Answerable queries, aligned with ``qrels``.
        unanswerable: Queries the annotators marked ``No Answer Present.``.
            Held separately because they are a labelled *negative* set for the
            abstention gate, not evaluation queries for retrieval.
        n_rows: Rows consumed.
        n_duplicates: Passage occurrences collapsed by deduplication. Reported
            because it is large (MS MARCO pools overlap) and it is the number
            that makes the dedup/qrels invariant matter.
    """

    documents: list[Any] = field(default_factory=list)
    qrels: dict[int, set[str]] = field(default_factory=dict)
    queries: list[Query] = field(default_factory=list)
    unanswerable: list[Query] = field(default_factory=list)
    n_rows: int = 0
    n_duplicates: int = 0

    def __len__(self) -> int:
        return len(self.documents)

    def stats(self) -> dict[str, Any]:
        """Summary for the report header and the ingest manifest."""
        gold = [len(v) for v in self.qrels.values()]
        return {
            "n_rows": self.n_rows,
            "n_documents": len(self.documents),
            "n_duplicates_collapsed": self.n_duplicates,
            "n_queries": len(self.queries),
            "n_unanswerable": len(self.unanswerable),
            "mean_gold_per_query": round(sum(gold) / len(gold), 3) if gold else 0.0,
            "mean_doc_chars": (
                round(sum(len(d.text) for d in self.documents) / len(self.documents), 1)
                if self.documents
                else 0.0
            ),
        }


# --- row accessors ------------------------------------------------------------


def _passages(row: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    """Extract ``(english_passages, is_selected)`` from a row, defensively.

    The struct column arrives as a plain dict from ``datasets`` and as a dict of
    lists from parquet; both are handled. A row whose two lists disagree in
    length is truncated to the shorter one rather than raising: a single
    malformed row in a 98k-row shard should not abort an ingest, and the
    truncation cannot invent a label because ``zip`` pairs by position.
    """
    block = row.get("passages") or {}
    texts = list(block.get("English_passages") or [])
    flags = [int(x) for x in (block.get("is_selected") or [])]
    if len(texts) != len(flags):
        n = min(len(texts), len(flags))
        texts, flags = texts[:n], flags[:n]
    return texts, flags


def _row_query(row: Mapping[str, Any]) -> Query:
    _, flags = _passages(row)
    answer = str(row.get("Answer") or "")
    eng_answer = str(row.get("Eng_Answer") or "")
    answerable = any(flags) and NO_ANSWER_SENTINEL not in (answer, eng_answer)
    return Query(
        query_id=int(row.get("query_id", -1)),
        eng_query=str(row.get("Eng_Query") or ""),
        indic_query=str(row.get("query") or ""),
        target_lang=str(row.get("target_lang") or ""),
        query_type=str(row.get("query_type") or ""),
        answer=answer,
        eng_answer=eng_answer,
        answerable=answerable,
    )


def _document(doc_id: str, text: str, row: Mapping[str, Any], ordinal: int) -> Any:
    """Build a :class:`~voicerag.chunking.base.Document` for one passage.

    Imported here rather than at module scope so that ``eval.dataset`` stays
    importable (and the fixture generator usable) even in a context where the
    chunking package's dependencies are not installed.

    The title is taken from the row's ``meta`` struct when it carries one. It is
    deliberately **never** derived from the query: a title containing the query
    text would leak the label straight into the embedded text and make the
    metadata-aware chunking strategy look brilliant for entirely fake reasons.
    """
    from voicerag.chunking.base import Document

    meta = row.get("meta") if isinstance(row.get("meta"), Mapping) else {}
    return Document(
        doc_id=doc_id,
        text=text,
        title=str(meta.get("title") or ""),
        url=str(meta.get("url") or ""),
        meta={
            "source": "msmarco-xi",
            "lang": "eng_Latn",
            "first_seen_row": ordinal,
        },
    )


# --- public conversions -------------------------------------------------------


def to_documents(rows: Iterable[Mapping[str, Any]]) -> Iterator[Any]:
    """Yield deduplicated :class:`Document` objects from raw rows.

    Args:
        rows: Raw MSMARCO-XI rows. Consumed once; pass a list if you also need
            :func:`to_qrels` and :func:`to_queries`, or use :func:`build_corpus`
            which does all three in a single streaming pass.

    Yields:
        One ``Document`` per distinct passage, ``doc_id`` = :func:`passage_id`,
        in first-seen order. Empty and whitespace-only passages are skipped --
        they carry no signal, cannot be gold, and would otherwise become
        zero-vector index entries that match nothing.
    """
    seen: set[str] = set()
    for ordinal, row in enumerate(rows):
        texts, _ = _passages(row)
        for text in texts:
            if not text or not text.strip():
                continue
            doc_id = passage_id(text)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            yield _document(doc_id, text, row, ordinal)


def to_qrels(rows: Iterable[Mapping[str, Any]]) -> dict[int, set[str]]:
    """Build ``query_id -> {gold passage_id}`` from ``is_selected``.

    Args:
        rows: Raw rows. Consumed once.

    Returns:
        A mapping containing only queries with at least one gold passage.
        Queries with none are absent by construction (see
        :func:`abstention_set`), so a caller iterating this mapping can never
        accidentally score an unanswerable query as a retrieval failure.

    The values are :func:`passage_id` hashes -- the *same* identity
    deduplication uses -- which is what makes gold labels survive dedup.
    """
    qrels: dict[int, set[str]] = {}
    for row in rows:
        texts, flags = _passages(row)
        gold = {
            passage_id(t)
            for t, f in zip(texts, flags)
            if f == 1 and t and t.strip()
        }
        if gold:
            qrels.setdefault(int(row.get("query_id", -1)), set()).update(gold)
    return qrels


def to_queries(rows: Iterable[Mapping[str, Any]]) -> list[Query]:
    """Materialise every row as a :class:`Query`, answerable or not."""
    return [_row_query(row) for row in rows]


def abstention_set(rows: Iterable[Mapping[str, Any]]) -> list[Query]:
    """Queries whose candidate passages are *all* judged non-relevant.

    Args:
        rows: Raw rows. Consumed once.

    Returns:
        The human-labelled "corpus cannot answer this" set. These are real
        annotator judgements, not hand-written adversarial examples, which is
        the entire reason the abstention numbers in the report are worth
        anything.
    """
    return [q for q in (_row_query(r) for r in rows) if not q.answerable]


def build_corpus(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_queries: int | None = None,
) -> Corpus:
    """Documents, qrels, queries and the abstention set in one streaming pass.

    Args:
        rows: Raw rows from :func:`load_msmarco_xi`, :func:`load_jsonl` or
            :func:`synthetic_rows`.
        max_queries: Stop after this many *answerable* queries. Unanswerable
            rows encountered before the limit are still kept, because they cost
            nothing and the abstention set is the scarcer resource.

    Returns:
        A populated :class:`Corpus`.

    One pass matters: the shard is streamed, so a second pass means a second
    440 MB read (or, under ``streaming=True``, a second HTTP range scan).
    """
    corpus = Corpus()
    seen: set[str] = set()
    for ordinal, row in enumerate(rows):
        corpus.n_rows += 1
        texts, flags = _passages(row)
        for text in texts:
            if not text or not text.strip():
                continue
            doc_id = passage_id(text)
            if doc_id in seen:
                corpus.n_duplicates += 1
                continue
            seen.add(doc_id)
            corpus.documents.append(_document(doc_id, text, row, ordinal))

        query = _row_query(row)
        gold = {
            passage_id(t) for t, f in zip(texts, flags) if f == 1 and t and t.strip()
        }
        if gold and query.answerable:
            corpus.qrels.setdefault(query.query_id, set()).update(gold)
            corpus.queries.append(query)
        else:
            corpus.unanswerable.append(query)

        if max_queries is not None and len(corpus.queries) >= max_queries:
            break
    return corpus


# --- loaders ------------------------------------------------------------------


def load_msmarco_xi(
    shard: str = DEFAULT_SHARD,
    *,
    limit: int | None = 20_000,
    streaming: bool = True,
    repo_id: str = DEFAULT_REPO_ID,
    split: str = "validation",
) -> Iterator[dict[str, Any]]:
    """Stream rows from one pinned shard of MSMARCO-XI.

    Args:
        shard: Path of the parquet file inside the repo. Defaults to
            :data:`DEFAULT_SHARD`; the split name is derived from it.
        limit: Stop after this many rows. ``None`` reads the whole shard
            (97,941 rows, ~1M passages) -- rarely what you want.
        streaming: Keep ``True``. With ``False``, ``datasets`` materialises the
            shard to the local cache first, which is 440 MB of disk and minutes
            of wall clock before the first row appears.
        repo_id: Overridable for a local copy of the same layout.
        split: Split key the loader registers ``shard`` under.

    Yields:
        Raw rows as plain dicts, schema exactly as documented in this module.

    Raises:
        DatasetUnavailable: If ``datasets`` is not installed or the hub cannot
            be reached. The message states the fix; callers that must work
            offline should catch this and use :func:`synthetic_rows`.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise DatasetUnavailable(
            "the `datasets` package is required to read MSMARCO-XI; "
            "`pip install datasets`, or use eval.dataset.synthetic_rows() for "
            "an offline run"
        ) from exc

    try:
        ds = load_dataset(
            repo_id,
            data_files={split: shard},
            split=split,
            streaming=streaming,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise DatasetUnavailable(
            f"could not open {repo_id}:{shard} ({type(exc).__name__}: {exc}); "
            "check network access to huggingface.co and HF_TOKEN if the repo is gated"
        ) from exc

    rows: Iterable[Any] = ds if limit is None else islice(ds, limit)
    for row in rows:
        yield dict(row)


def dump_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write rows to (optionally gzipped) JSONL and return the path.

    Caching a slice of the shard locally turns a repeated ablation from a
    network operation into a disk read, which is what makes ``--resume`` in
    :mod:`scripts.ingest` meaningful.

    Args:
        rows: Raw rows.
        path: Destination. ``.gz`` suffix enables compression.

    Returns:
        The written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "wt", encoding="utf-8") as fh:  # type: ignore[operator]
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def load_jsonl(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Read rows previously written by :func:`dump_jsonl`.

    Args:
        path: Source file, ``.jsonl`` or ``.jsonl.gz``.
        limit: Stop after this many rows.

    Yields:
        Raw rows.
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


# --- synthetic fixture --------------------------------------------------------

#: Topic templates for the offline fixture. Each is a (subject, fact-template,
#: filler) family whose vocabulary is distinctive enough that a gold passage is
#: findable, while sibling topics share sentence structure -- which is what
#: makes the distractors *hard* rather than trivially separable.
_TOPICS: tuple[tuple[str, str, str, str], ...] = (
    (
        "the boiling point of {subject}",
        "The boiling point of {subject} is {value} degrees celsius at standard "
        "atmospheric pressure.",
        "{value} degrees celsius",
        "Chemists measure this property in a calibrated apparatus. The value "
        "shifts with altitude because ambient pressure falls. Industrial "
        "handling guides quote the same figure for {subject}.",
    ),
    (
        "when was {subject} founded",
        "{subject} was founded in {value} by a group of regional merchants.",
        "{value}",
        "Its early charter covered trade along the river. Later records list "
        "warehouses, a mint and a customs house. Historians treat the founding "
        "date of {subject} as well attested.",
    ),
    (
        "how tall is {subject}",
        "{subject} rises {value} metres above the surrounding plain.",
        "{value} metres",
        "Surveyors re-measured it with differential GPS. Earlier barometric "
        "estimates were several metres lower. Guidebooks now print the "
        "corrected height of {subject}.",
    ),
    (
        "what is the population of {subject}",
        "The population of {subject} was recorded as {value} in the most recent "
        "census.",
        "{value}",
        "The count includes seasonal residents of the outer wards. Growth has "
        "been driven by the port expansion. Planners expect {subject} to grow "
        "further this decade.",
    ),
    (
        "who discovered {subject}",
        "{subject} was discovered by the astronomer Halden Prue in {value}.",
        "Halden Prue",
        "The discovery was confirmed by two independent observatories. Earlier "
        "sightings had been dismissed as instrument noise. The naming of "
        "{subject} followed the usual convention.",
    ),
)

_SUBJECT_STEMS: tuple[str, ...] = (
    "Kelvarn", "Torrisel", "Mundaka", "Zephyrine", "Bactroline", "Olmsheda",
    "Quenmar", "Vasterly", "Hollimund", "Prasadine", "Nerivault", "Cambrisk",
    "Dolmarine", "Espirith", "Farnholt", "Gravenmoor", "Hesperil", "Ivorspan",
    "Jandrelic", "Kaustrine", "Lorbanite", "Marasenth", "Nyxholme", "Oberquay",
)

_QUERY_TYPES: tuple[str, ...] = ("NUMERIC", "ENTITY", "LOCATION", "PERSON", "DESCRIPTION")

#: Appended to every passage so a document reliably exceeds one fixed window
#: (120 words) and therefore splits into several chunks. Without it, every
#: strategy would emit exactly one chunk per passage on this fixture and the
#: chunking ablation would compare six identical configurations -- an offline
#: fixture that cannot distinguish the things it exists to test is worthless.
#:
#: The sentences are topic-neutral boilerplate on purpose: they lengthen the
#: document without adding query terms, so they cannot leak relevance. There
#: are several variants rather than one because a single shared tail would make
#: hundreds of chunks byte-identical, which is not what a real corpus looks like
#: and would depress every sentence-level strategy for an artefactual reason.
_TAILS: tuple[str, ...] = (
    "Reference works differ on how the measurement should be reported, and the "
    "committee that reviews such entries meets twice a year to reconcile them. "
    "Local archives hold correspondence from the surveyors who compiled the "
    "earliest tables, much of it never published. A later edition added "
    "footnotes describing the instruments then in use, which were recalibrated "
    "after a dispute over their accuracy. Readers consulting older printings "
    "should note that the figures there were rounded differently, and that at "
    "least one plate was reset between impressions without any note being made.",
    "The entry has been revised three times since it was first compiled, each "
    "revision noted in the margin by the editor responsible. Correspondence "
    "between the compilers survives in a provincial library and has been "
    "digitised only in part. Two of the original tables were lost in a flood "
    "and were reconstructed from a printer's proof. Scholars generally treat "
    "the reconstructed portions as reliable but not authoritative, pending a "
    "comparison against the surviving proof sheets held in the same collection.",
    "A standing panel reviews submissions of this kind before they are added to "
    "the register, and its minutes are published annually. Earlier compilers "
    "worked without any such review, which explains several inconsistencies "
    "between the first and second editions. The panel has since issued guidance "
    "on rounding, units and the citation of primary sources. Entries that "
    "predate the guidance are being revisited in order of publication, a process "
    "the panel expects to finish only after several further sitting seasons.",
    "Documentation for this record was assembled from municipal filings, a "
    "surveyor's notebook and two contemporary newspaper accounts. Where the "
    "sources disagree the register prefers the filing, on the grounds that it "
    "was made closest to the event. A note in the appendix lists the "
    "discrepancies for readers who need them. The register is reissued every "
    "few years with corrections supplied by its readers, whose submissions are "
    "acknowledged in the front matter and retained in the editorial file.",
)


def _passage(fact: str, filler: str, variant: int) -> str:
    """Assemble one candidate passage: the fact, its filler, and a neutral tail."""
    return " ".join((fact, filler, _TAILS[variant % len(_TAILS)]))


#: Stems reserved for the subjects of *unanswerable* queries. They appear in no
#: passage anywhere in the corpus, which is the defining property of a query the
#: corpus cannot answer: nothing lexically or semantically close exists to
#: retrieve. Drawing those subjects from the shared pool instead would leave a
#: strongly-matching passage about the same entity sitting in the index under a
#: different fact type, and the abstention set would be measuring something else
#: entirely.
_ABSENT_STEMS: tuple[str, ...] = (
    "Zolquint", "Vermadral", "Xanthorpe", "Ulvarine",
    "Threnody", "Sarnovex", "Requiline", "Pellucid",
)


def _subject(rng: random.Random, i: int) -> str:
    """Deterministic multi-token subject, unique per index."""
    stem = _SUBJECT_STEMS[i % len(_SUBJECT_STEMS)]
    suffix = ("Prime", "Basin", "Reach", "Hollow", "Ridge")[(i // len(_SUBJECT_STEMS)) % 5]
    return f"{stem} {suffix} {i}"


def _absent_subject(i: int) -> str:
    """Subject guaranteed never to occur in any generated passage."""
    stem = _ABSENT_STEMS[i % len(_ABSENT_STEMS)]
    suffix = ("Waste", "Shelf", "Verge", "Cleft")[(i // len(_ABSENT_STEMS)) % 4]
    return f"{stem} {suffix} {900 + i}"


def synthetic_rows(
    n_queries: int = 60,
    *,
    n_passages: int = 8,
    unanswerable_rate: float = 0.25,
    duplicate_rate: float = 0.3,
    seed: int = 0,
    target_lang: str = "hin_Deva",
) -> list[dict[str, Any]]:
    """Generate rows with the exact MSMARCO-XI schema, with no downloads.

    This is what makes every test, ``scripts/smoke.py`` and a full ablation
    smoke-run work on a machine with no network. It is a *fixture*, not a
    benchmark: absolute scores on it mean nothing, but relative comparisons
    between chunking strategies and the correctness of every code path are
    exercised for real.

    Design choices that make it non-trivial:

    * Distractor passages come from the **same topic template** with a different
      subject, so BM25 cannot win on template words alone and dense retrieval
      cannot win on topic alone -- the discriminating token is the subject.
    * ``duplicate_rate`` re-injects gold passages verbatim into other rows'
      candidate pools, reproducing MS MARCO's pool overlap and therefore
      exercising the dedup/qrels invariant rather than assuming it.
    * ``unanswerable_rate`` rows have every ``is_selected == 0`` and
      ``Answer == "No Answer Present."``, mirroring the real labelled
      abstention set.

    Args:
        n_queries: Number of rows to emit.
        n_passages: Candidate passages per row.
        unanswerable_rate: Fraction of rows with no gold passage.
        duplicate_rate: Probability that a row's pool gets a verbatim copy of
            an earlier gold passage appended.
        seed: RNG seed. The output is a pure function of this.
        target_lang: FLORES-200 code written into the rows; the "translated"
            query is a marked transliteration stand-in, since a real translation
            is not available offline.

    Returns:
        A list of row dicts, ready for :func:`build_corpus`.

    Raises:
        ValueError: If ``n_queries`` or ``n_passages`` is not positive, or a
            rate falls outside ``[0, 1]``.
    """
    if n_queries < 1:
        raise ValueError(f"n_queries must be >= 1, got {n_queries}")
    if n_passages < 1:
        raise ValueError(f"n_passages must be >= 1, got {n_passages}")
    for label, rate in (("unanswerable_rate", unanswerable_rate), ("duplicate_rate", duplicate_rate)):
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"{label} must be in [0, 1], got {rate}")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    gold_pool: list[str] = []

    for i in range(n_queries):
        q_tmpl, fact_tmpl, answer_tmpl, filler_tmpl = _TOPICS[i % len(_TOPICS)]
        value = str(rng.randint(101, 9999))
        unanswerable = rng.random() < unanswerable_rate
        # An unanswerable query asks about an entity that occurs nowhere in the
        # corpus; an answerable one asks about the entity its gold passage
        # describes.
        subject = _absent_subject(i) if unanswerable else _subject(rng, i)

        gold_text = _passage(
            fact_tmpl.format(subject=subject, value=value),
            filler_tmpl.format(subject=subject, value=value),
            i,
        )

        passages: list[str] = []
        flags: list[int] = []
        for j in range(n_passages - 1):
            # Hard negative: same template, different subject and value.
            d_tmpl_idx = (i + j + 1) % len(_TOPICS)
            _, d_fact, _, d_filler = _TOPICS[d_tmpl_idx]
            d_subject = _subject(rng, (i * 7 + j * 13 + 101) % (len(_SUBJECT_STEMS) * 5))
            d_value = str(rng.randint(101, 9999))
            passages.append(
                _passage(
                    d_fact.format(subject=d_subject, value=d_value),
                    d_filler.format(subject=d_subject, value=d_value),
                    i * 31 + j,
                )
            )
            flags.append(0)

        if unanswerable:
            # No gold: the pool is all distractors, exactly like the real
            # "No Answer Present." rows.
            other = _subject(rng, i + 997)
            passages.append(
                _passage(
                    fact_tmpl.format(subject=other, value=value),
                    filler_tmpl.format(subject=other, value=value),
                    i + 2,
                )
            )
            flags.append(0)
        else:
            insert_at = rng.randrange(len(passages) + 1)
            passages.insert(insert_at, gold_text)
            flags.insert(insert_at, 1)
            gold_pool.append(gold_text)

        if gold_pool and rng.random() < duplicate_rate:
            # Pool overlap: an earlier gold passage reappears verbatim here as a
            # non-relevant candidate. Deduplication must collapse it, and the
            # earlier query's label must still resolve.
            passages.append(rng.choice(gold_pool))
            flags.append(0)

        eng_query = q_tmpl.format(subject=subject)
        answer = (
            NO_ANSWER_SENTINEL
            if unanswerable
            else answer_tmpl.format(subject=subject, value=value)
        )
        rows.append(
            {
                "source_lang": "eng_Latn",
                "target_lang": target_lang,
                "meta": {"synthetic": True, "topic": i % len(_TOPICS)},
                "Answer": answer,
                "query_id": 10_000 + i,
                "query_type": _QUERY_TYPES[i % len(_QUERY_TYPES)],
                "passages": {
                    "English_passages": passages,
                    "Translated_passages": [f"[{target_lang}] {p}" for p in passages],
                    "is_selected": flags,
                },
                "Eng_Query": eng_query,
                "Eng_Answer": answer,
                "query": f"[{target_lang}] {eng_query}",
            }
        )
    return rows


def synthetic_corpus(
    n_queries: int = 60,
    **kwargs: Any,
) -> Corpus:
    """Convenience wrapper: :func:`synthetic_rows` straight into a :class:`Corpus`."""
    return build_corpus(synthetic_rows(n_queries, **kwargs))
