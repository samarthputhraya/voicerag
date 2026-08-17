# VoiceRAG

**Speak a question. Get an answer grounded in MS MARCO, with citations and a receipt for every millisecond.**

Submission for **HH Goa 2026 — Shortlisting Task 2: Voice-Enabled RAG System**.

> Live demo: `<URL>` · Repo: `<URL>` · Demo video: `<URL>` · Process video: `<URL>`

---

## The latency claim, stated exactly

The brief defines the target precisely, and we quote it rather than paraphrase:

> *"The full process — chunking + vector DB retrieval + everything through to
> final output — should complete in under **200ms**."*

Two things follow from that sentence, and both drive the engineering.

**It ends at *final* output, not first token.** The usual voice-assistant metric
is time-to-first-token; this is stricter. We measure to the **last** token of the
complete grounded answer. At Groq's throughput every output token costs roughly a
millisecond, so **answer length is a hard engineering constraint here, not a
stylistic preference** — the prompt enforces terse, factoid-shaped answers and
generation is capped. This suits the corpus exactly: MS MARCO's gold answers
*are* short factoid answers, so brevity costs us nothing in quality.

**It begins at chunking/retrieval, not at the microphone.** The brief's own
pipeline diagram lists `Voice input → Speech-to-text → Chunking/Retrieval →
Answer generation` as four stages, and scopes the 200 ms sentence to the
retrieval-and-generation portion. We measure that quantity — and publish
everything else beside it, unmassaged, so nothing is hidden by the framing:

| Metric | What it covers | Target |
|---|---|---|
| **Pipeline latency** ← *the 200 ms claim* | Transcript in → **final** answer token out. Guardrails, embedding, hybrid retrieval, fusion, abstention, prompt build, full generation. | **< 200 ms** |
| Time to first token | Transcript in → first token. Reported because it is what a user perceives. | reported |
| Speech latency | End of speech (VAD endpoint) → final transcript. Sarvam, measured client-side. | reported |
| Client network | Browser ↔ API round trip. Geography, not engineering. | reported |
| Wall clock | End of speech → last token painted. What a human actually experiences. | reported |

`eval/latency.py` emits all five, per stage, at P50/P70/P90/P100.

Percentiles use the **nearest-rank** method (`ceil(p/100 × n)`), stated here
because P70 is unusual enough that the interpolation choice materially moves the
number. The frontend HUD uses the identical definition, so the live demo and the
report can never disagree.

<!-- BENCHMARK_TABLE_START -->
*Populated by `python scripts/bench_latency.py`.*
<!-- BENCHMARK_TABLE_END -->

---

## What makes this fast

Three decisions do most of the work.

### 1. Nothing on the query path crosses a network except the LLM

Query embedding runs **in process** on a [model2vec](https://github.com/MinishLab/model2vec)
static model — a token-embedding lookup, no transformer forward pass, no torch.
A hosted embedding API would cost 100–300 ms of round trip; this costs
microseconds. The vector index is in-process faiss HNSW, not a hosted vector DB.

Measured on 2 vCPU, 200k vectors × dim 256, `efConstruction=80`:

| `efSearch` | P50 | P70 | P100 |
|---:|---:|---:|---:|
| 32 | 0.630 ms | 0.679 ms | 1.331 ms |
| 64 | 1.126 ms | 1.179 ms | 2.146 ms |
| 128 | 2.129 ms | 2.209 ms | 2.958 ms |

Retrieval is **effectively free**. The budget belongs to the LLM, so that is
where we spent the optimisation effort.

### 2. Speculative retrieval on partial transcripts

Sarvam's realtime WebSocket emits partial transcripts while you are still
talking. On every partial that gains ≥3 tokens, we fire a **cancellable**
embed + ANN search keyed by a hash of the partial. When the final transcript
arrives, if it is within cosine 0.98 of the last speculated partial, we reuse the
cached result and skip retrieval entirely.

Most short questions are fully determined before the endpoint fires, so
post-endpoint retrieval cost collapses toward zero at P50 — the work happened
during the silence at the end of your sentence. Hit rate and milliseconds saved
are reported per request in the API response and shown live in the HUD.

### 3. The audio never touches our server

Sarvam is hosted in India; Groq is hosted in the US. Rather than compromise, we
split the path. The **browser** streams PCM directly to Sarvam over WebSocket
(India → India, ~15–40 ms), and only the finished transcript — a few dozen
bytes — crosses to our API, which is **co-located with Groq in US-West**. Each
leg is short even though the endpoints are twelve time zones apart.

The Sarvam key never reaches the browser: `POST /stt/token` mints a short-lived
credential server-side.

> **The single largest latency win in this project is not in the model layer.**
> The standard browser VAD library waits **1400 ms** of silence by default before
> deciding you stopped speaking. We run it at 260 ms. That one constant is worth
> more than every other optimisation here combined, and it is invisible unless
> you go looking. See `VAD_OPTS` in `web/app/page.tsx`.

---

## Retrieval that is actually engineered

Six chunking strategies, implemented behind one interface and **ablated against
real relevance judgements** — not chosen by vibes.

| Strategy | What it does | Why it might win |
|---|---|---|
| `fixed` | 120-word windows, 24-word overlap | The control. An ablation without a baseline proves nothing. |
| `recursive` | Descends paragraph → line → sentence → clause | Breaks on natural boundaries instead of mid-clause. |
| `sentence_window` | Indexes 1 sentence, serves a 7-sentence window | Small units embed sharply; the generator still gets context. Usually strongest on short factoid queries — which is exactly the MS MARCO distribution. |
| `metadata` | Fixed geometry + title/section prefix **on the embedding only** | "It was signed in 1919" is unretrievable alone. The prefix restores the referent. Cited text stays verbatim. |
| `semantic` | Boundaries at per-document similarity percentiles | A fixed cosine threshold that works on encyclopedic prose over-splits conversational text. A percentile adapts per document. |
| `contextual` | Prepends a situating blurb before embedding | Recovers the discourse context a chunk loses in isolation. Composes as an overlay over any splitter. |

Retrieval is **hybrid**: dense HNSW and BM25 run concurrently and fuse with
Reciprocal Rank Fusion. Component scores and ranks survive fusion, because the
abstention logic needs them (see below).

<!-- ABLATION_TABLE_START -->
*Populated by `python scripts/run_ablation.py`. Reports Recall@1/5/10, MRR@10,
nDCG@10, chunk count, index build time and mean query latency per strategy.*
<!-- ABLATION_TABLE_END -->

---

## Guardrails that know when *not* to answer

Abstention is evaluated against **real labels, not hand-written examples.**

MS MARCO contains queries whose candidate passages are *all* marked
non-relevant — the human annotator's verdict is `"No Answer Present."` Those
rows survive into MSMARCO-XI, giving us a genuine labelled set of
*the corpus cannot answer this*. We report precision/recall/F1 on it.

Three stages, ordered by cost:

**Input guard** (~0.1 ms, before retrieval) — filler/silence transcripts,
prompt-injection heuristics, inappropriate content. Catching a dud transcript
here avoids a pointless retrieval *and* a pointless LLM call.

**Abstention** (~0.2 ms, after retrieval, before generation) — decides from
retrieval signals alone whether the corpus can answer:

- `max_score` — top hit below an absolute floor means nothing relevant exists
- `score_gap` — a confident retrieval separates top-1 from top-2; a flat distribution means the retriever is guessing
- `score_entropy` — high entropy over the top-k softmax means no clear winner
- `agreement` — did dense *and* sparse surface the same chunk? Disagreement is a strong out-of-domain signal, and it is free, because fusion already computed both ranks

Thresholds are **calibrated from data**, not hardcoded. When this fires we
return early and never call the LLM — which is why abstention is also a latency
feature.

**Grounding verification** (concurrent with streaming) — sentence-level claim
extraction, lexical entailment against retrieved chunks, exact checking of
numbers and dates (the things that hallucinate most and are cheapest to verify),
and citation validation. Runs *alongside* generation, so it costs no wall-clock.

---

## The orchestration harness

Not a prompt and a hope. `src/voicerag/harness/`:

- **Deadline** — one monotonic budget threaded through every stage. A retry that cannot finish before the deadline is *not attempted*, because under a latency SLO the question is never "how many attempts" but "how much time is left".
- **RetryPolicy** — exponential backoff with **full jitter**. Fixed backoff makes every client that failed during an outage retry in lockstep and recreate the herd that caused it.
- **Error taxonomy** — callers classify their own failures. A 401 is `PermanentError` and is never retried; guessing from exception type is how harnesses end up retrying auth failures forever.
- **CircuitBreaker** — closed → open → half-open. A dead provider is skipped outright rather than adding its full timeout to every request. A failed probe re-opens.
- **first_healthy** — ordered provider fallback (Groq → Gemini) that skips open circuits.
- **Trace** — every published latency number originates here. Spans may overlap, deliberately: `total_ms` is the envelope, `critical_path_ms` is the union of busy intervals, and the gap between them is exactly what the concurrency bought.

---

## Architecture

```
Browser (Goa)                          API (US-West, co-located with Groq)
─────────────                          ──────────────────────────────────
 mic
  ↓ Silero VAD (redemption 260ms)
  ↓ 16kHz PCM
  ├──────── wss ────────► Sarvam saaras:v3-realtime   (India → India, ~15-40ms)
  │                          │
  │  ◄── transcript.partial ─┤
  ├── POST /speculate ──────────────────► warm retrieval cache (cancellable)
  │  ◄── transcript.final ───┘
  └── POST /ask/stream ─────────────────► input guard      ~0.1ms
                                          embed (in-proc)  ~0.5ms
                                          dense ∥ sparse   ~2ms   ← usually cached
                                          RRF fusion       ~0.1ms
                                          abstention       ~0.2ms
                                          ├─ abstain → return, LLM never called
                                          └─ prompt → Groq TTFT  ~120ms
                                             ↓ grounding check runs concurrently
       ◄──── SSE token deltas, then final frame with full trace ────┘
```

---

## Quickstart

```bash
git clone <repo> && cd voicerag
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # add SARVAM_API_KEY and GROQ_API_KEY

# Build an index. ~200k passages from a single 440MB shard, not the 55.6GB dump.
python scripts/ingest.py --limit 20000 --strategy sentence_window --out data/index

# Prove it works, offline, with no API keys:
python scripts/smoke.py

# Reproduce our numbers:
python scripts/run_ablation.py --out reports/ablation.md
python scripts/bench_latency.py --iterations 200 --out reports/latency.json

uvicorn voicerag.api.main:app --reload      # API  :8000
cd web && npm install && npm run dev        # UI   :3000
```

### Dataset

`ai4bharat/MSMARCO-XI` is 55.6 GB. We never download it.

We pin **one 440 MB validation shard** and stream it. Each row carries the
original human-written English MS MARCO passages *and* their Indic translation
of the same query, plus `is_selected` — the human relevance judgement, which
becomes our qrels. Passages are deduplicated by content hash, and the qrels
reference that hash, so gold labels survive deduplication.

That last detail sounds pedantic. Get it wrong and every retrieval metric you
report is silently invalid. There is a test for it.

### Cross-lingual mode

Because each row pairs an **Indic query** with **English passages** for the same
`query_id`, the same index serves both an English and a Hindi voice demo against
identical ground truth. Ask in Hindi, retrieve in English, answer grounded.
Switch languages in the UI.

---

## Tests

```bash
pytest -q
```

Every test runs **offline** with no network and no model downloads — the
`lsa` embedder backend is a real corpus-trained LSA model that needs no weights,
which keeps the whole suite and a full ablation runnable anywhere.

---

## Repository layout

```
src/voicerag/
  chunking/     six strategies + registry, offset-preserving
  embed/        static (model2vec) · lsa (zero-download) · onnx, all pluggable
  index/        dense HNSW · BM25 · RRF hybrid · chunk store
  guardrails/   input · abstention · grounding · policy
  harness/      trace · deadline · retry · circuit breaker · fallback
  stt/          sarvam · elevenlabs · speculative driver
  generate/     groq · gemini · prompt · router
  pipeline.py   the orchestrated request path
  api/          FastAPI: /ask /ask/stream /speculate /stt/token /healthz /stats
eval/           dataset · metrics · ablation · latency · abstention_eval
web/            Next.js voice UI with live latency HUD
deploy/         Dockerfile · render.yaml
```

## License and attribution

Built on MS MARCO via `ai4bharat/MSMARCO-XI`. MS MARCO is licensed for
non-commercial research use; this submission is a hackathon entry and inherits
those terms. Speech recognition by Sarvam AI. Inference by Groq.
