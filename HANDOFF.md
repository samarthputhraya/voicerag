# VoiceRAG — session handoff

Paste this whole file into a new chat. It is written to be read cold.

**Submission:** HH Goa 2026, Shortlisting Task 2 (Voice-Enabled RAG).
**Deadline:** 22 Aug 2026. ~1000 teams, top ~50 qualify.
**Repo:** https://github.com/samarthputhraya/voicerag (public, 11 commits)
**Machine:** Windows 11, Python 3.12 in `.venv`, ~15.4 GB RAM (often <3 GB free),
slow/flaky network, no Docker, portable Node at
`C:\Users\samar\AppData\Local\nodejs-portable\node-v24.19.0-win-x64`.

---

## 1. THE OPEN PROBLEM — read this first

**The user reports the microphone still does not work. Typing works.** This is
the single blocking issue and it has survived three rounds of fixes.

**Critical constraint on whoever picks this up:** the previous session had **no
microphone and no browser automation**. Every voice fix was reasoned from
library source and the wire protocol, never observed working end to end in a
real browser. If you also cannot drive a real mic, say so early rather than
shipping another unverified fix.

### What IS verified working (measured, not assumed)

- The relay itself: `ws://127.0.0.1:8000/stt/stream` connects in **8 ms** and
  returns real transcripts when fed synthesized speech from a Python client.
- Sarvam STT, TTS and translate all return 200 with a live key.
- Full voice→answer chain, driven from Python: **~1.0–1.6 s** after speech ends,
  in Hindi, Bengali and Tamil.

So: **backend voice works. The browser leg is what fails.**

### Already ruled out (checked in `web/node_modules`, do not redo)

- `onFrameProcessed` **does** pass the `Float32Array` frame — not the bug.
- `@ricky0123/vad-web` resolves to the **top-level `onnxruntime-web` 1.27.0**;
  there is no nested copy with different WASM filenames.
- Every asset the VAD requests returns **200** from the dev server:
  `vad.worklet.bundle.min.js`, `silero_vad_v5.onnx`, `silero_vad_legacy.onnx`,
  `ort-wasm-simd-threaded.wasm`, `ort-wasm-simd-threaded.mjs`.
  (`ort-wasm.wasm` / `ort-wasm-simd.wasm` 404, but 1.27 never requests them.)

### The prime suspects, in order

1. **Open the browser devtools console and network tab.** Nobody has done this.
   The answer is almost certainly sitting there. `vad.errored` is now surfaced
   in the UI as of the last commit, so a load failure should name itself.
2. **`vad-web` defaults are `baseAssetPath: "./"` and `onnxWASMBasePath: "./"`
   — relative.** At `localhost:3000/` that resolves to `/` and works, but it is
   fragile. Try passing them explicitly as `"/"` in `VAD_OPTS`
   (`web/app/page.tsx`).
3. **Threaded WASM needs cross-origin isolation.** `ort-wasm-simd-threaded.wasm`
   wants `SharedArrayBuffer`, which needs COOP/COEP headers Next does not send
   by default. ORT normally falls back to single-threaded, but if it does not,
   the model never initialises. Try setting
   `ort.env.wasm.numThreads = 1` before VAD init, or add the COOP/COEP headers
   in `next.config.mjs`.
4. **Microphone permission / secure origin.** `localhost` is a secure origin so
   this *should* be fine, but confirm the browser actually prompted.
5. **`onSpeechEnd` fallback (added last commit, unverified).** The VAD hands the
   complete utterance to `onSpeechEnd`; the code now resends it in full when the
   frame stream is known to have dropped audio. If streaming is the problem,
   consider making this the *only* path (simpler, loses partial transcripts and
   speculative retrieval, but removes the race entirely).

---

## 2. What the system is

Voice-enabled RAG over `ai4bharat/MSMARCO-XI`. Speak or type a question → STT →
hybrid retrieval → guardrails → LLM → grounded, cited answer, spoken back, with
a per-stage latency receipt.

**Serving now:** `data/index` — **956,128 chunks from 950,526 passages**,
`recursive` chunking, `static:minishlab/potion-base-8M` (dim 256), 2.6 GB on
disk. Built from the full 97,941-row Hindi validation shard in 234 s.

### Run it

```powershell
# API  (loads ~2.6GB index, takes ~45s)
$env:PYTHONPATH="C:\Users\samar\OneDrive\Documents\voicerag\src"
.\.venv\Scripts\python.exe -m uvicorn voicerag.api.main:app --host 127.0.0.1 --port 8000

# Frontend
$env:Path="C:\Users\samar\AppData\Local\nodejs-portable\node-v24.19.0-win-x64;$env:Path"
cd web; npm run dev      # localhost:3000

.\.venv\Scripts\python.exe -m pytest -q          # 496 pass, 1 skipped
```

### Endpoints

`POST /ask` · `POST /ask/stream` (SSE) · `WS /stt/stream` (relay) ·
`POST /speak` (TTS) · `GET /examples` · `GET /healthz` · `GET /stats` ·
`POST /speculate`

---

## 3. Requirement status

| # | Requirement | State |
|---|---|---|
| 1 | Sarvam **or** ElevenLabs STT | Both implemented. Sarvam via server-side relay. **Browser leg broken.** |
| 2 | "Vast" chunking | 6 strategies + real ablation that *changed the build*. Done. |
| 3 | <200 ms | P50 141.6 / P70 142.9 / **P100 157.3 ms** with modelled decode. Retrieval path 7.3 ms P50 fully measured. |
| 4 | P50/P70/P100 | `reports/latency.md`, nearest-rank, same definition in Python and the HUD. Done. |
| 5 | Harness | Error taxonomy, jittered retries, circuit breakers, Groq→OpenAI fallback. Done. |
| 6 | Guardrails | 4 stages. 9/9 adversarial probes refused. Done. |
| — | Public GitHub repo | Done. |
| — | **Live working link** | **NOT DONE.** Deploy chain fixed but never run. |
| — | **2 videos** | **NOT DONE.** |

---

## 4. Key measurements (all real, reproduce with the commands in the repo)

**Latency**, 200 warm runs from a 2,000-query pool, `reports/latency.md`:

| Stage | P50 | P100 |
|---|---:|---:|
| embed | 0.2 ms | 1.0 ms |
| retrieve | 6.8 ms | 22.6 ms |
| guards + prompt | ~0.3 ms | ~1 ms |
| generate *(modelled)* | 134 ms | 149.6 ms |

Real Groq from India: **0.5–4.4 s wall clock** (~14 ms of Groq compute; the rest
is trans-Pacific RTT). Serving budget is therefore `BUDGET_TOTAL_MS=8000`. The
200 ms figure is a benchmark target `bench_latency.py` passes explicitly.

**Chunking ablation** (`reports/ablation.md`), 19,878 passages / 400 queries:
`recursive` 0.5628 nDCG@10 > metadata > contextual > fixed > semantic >
`sentence_window` 0.5133. The index originally shipped with the **worst** one;
the ablation is why it was rebuilt.

**Serving-scale retrieval** (196k-passage index, 400 queries): R@10 **0.7202**,
nDCG@10 0.4321. *Not* the 0.9070 from the reduced-scale comparison — both are in
the README, labelled.

**Groq free tier: 8,000 tokens/min** ≈ 4 RAG questions/minute. This is a live
demo risk. `OPENAI_API_KEY` is wired and tested as fallback but **the user has
not yet supplied the key**.

---

## 5. Things that were broken and are now fixed (do not regress these)

1. **`asyncio.sleep()` vs Windows' 15.6 ms timer tick** inflated a 3.4 ms
   simulated token stream to 211 ms. `eval/latency.py` now uses deadline-based
   sleeping. Accurate to ±0.03 ms.
2. **Sarvam accepts no `token` query param and mints no ephemeral credential.**
   The browser was sending a self-signed capability Sarvam ignores. Fixed with
   `WS /stt/stream`, a server-side relay.
3. **`lsa:256` embed cost 20.3 ms P50** and blew the retrieval budget. Swapped
   for `static:minishlab/potion-base-8M` at 0.2 ms — **~320×**. Also removed the
   45-minute SVD fit; rebuilds now take 93 s–234 s.
4. **System prompt banned reading comprehension** ("never infer beyond what is
   written"), so the model refused questions whose answers were in the retrieved
   passages. Rewritten.
5. **Abstention prior rules were authored before any index existed.** Measured:
   `top1_agree` separation **0.000**, `rel_gap` **negative**, `max_score` is
   RRF-bounded noise, and `dense_max` — the only real signal — was thresholded
   at 0.30 against a live range of 0.54–0.86, so it never fired. Recalibrated to
   3 rules; balanced accuracy **0.499 → 0.912**.
6. **Grounding tokenised with `[a-z0-9]+`**, so a fabricated Devanagari answer
   tokenised to `[]` and was certified `grounded=True, score=1.0`. The
   hallucination check *inverted* on the corpus's own scripts.
7. **HUD summed overlapping trace spans** — showed **1621 ms** for a 557 ms
   request, in red. Now 4.16 ms retrieval path, LLM reported separately.
8. **`.env.example` shipped `BUDGET_TOTAL_MS=200`** while the README says
   `cp .env.example .env` — every answer truncated after one word.
9. **No TTS existed at all.** Added `POST /speak` (Sarvam `bulbul:v2`).
10. **Test fixture asserted `dense_max≈0.28`** while the real embedder produces
    **0.73** — thresholds tuned to reality looked broken in tests.

---

## 6. The multilingual angle (a genuine differentiator)

**Do not try to embed Indic text.** Measured: `potion-base-8M` holds 70
Devanagari tokens, all bare single characters. `कॉर्पोरेशन क्या है?` tokenises
to `['[UNK]','क','##य','##ा','ह','?']`. Cross-lingual alignment is 1-in-4
(chance). Hindi R@10 against the live index is **0.000**.

**What works instead:** Sarvam's realtime socket accepts `mode=translate`,
returning **English** text for speech in 11 Indian languages — so Devanagari
never reaches the embedder. Measured *faster* than transcribing (945 ms vs
1006 ms). Verified end to end in Hindi, Bengali, Tamil.

---

## 7. Dataset facts (checked, do not re-download blindly)

`ai4bharat/MSMARCO-XI` is **55.62 GB / 30 files**. 14 validation shards
(97,941 rows each) + 13 train shards. **No test split.**

**The important one:** `English_passages` is **byte-identical across all 14
validation shards.** The other 6.1 GB adds only translated query/answer text —
**zero new corpus passages.** Downloading more languages will not improve a
single answer.

Local: `data/raw/validation/hinval.parquet` (461.9 MB, all 97,941 rows) — this
is the entire English validation corpus.

The repo's own README lists filenames that don't exist (`gutrain`, `orval`,
`.jsonl`) and its loader script is broken. Use `scripts/ingest.py --parquet` or
`--download` (both added this session; the `datasets` streaming path hangs for
tens of minutes).

---

## 8. What to do next, in priority order

1. **Fix the microphone.** Devtools console first. See §1.
2. **Deploy the live link** — a required deliverable, currently missing. All six
   known blockers are fixed (`.dockerignore`, `model2vec` in requirements, env
   vars un-prefixed, `INDEX_URL`/`INDEX_ROWS` build args, correct baked model,
   uvloop flags dropped). Never actually built — no Docker locally. Script ready
   at `deploy/huggingface/push_space.ps1` (HF Space, free, 16 GB RAM — Render's
   `starter` is 512 MB and would OOM on a ~2.6 GB index; consider rebuilding at
   `--limit 20000` for a smaller deploy).
3. **Get the OpenAI key in** — Groq's 8k tok/min will break a live demo.
4. **`MMAP_SPARSE=false`** — `retrieve.sparse` was **98 ms** on the larger index
   (memory-mapped BM25 faulting). This is now the dominant retrieval cost.
5. **Videos** (2 required).
6. Remaining audit items: guardrail bypasses that evade the `how do I` anchor
   (e.g. *"steps to synthesise…"*), a per-IP rate limiter before the link goes
   public, and citations rendering as `0.033` (correct RRF values, bounded to
   `[1/61, 2/61]`, but they read as "3% match" on camera).

---

## 9. Honest assessment

The engineering is strong and unusually well-measured — six chunking strategies
with an ablation that *changed the build*, a guardrail chain with a published
negative result, a resilience harness with real failover, and a latency story
with the modelled and measured parts clearly separated. 496 tests pass.

What is weak: **the demo has never been seen working by a human through a
browser microphone**, and two required deliverables (live link, videos) do not
exist. A judge sees the demo, not the test suite. Fix the mic, ship the link,
film it — in that order.

A recurring theme worth carrying forward: on this project, *the docs were
repeatedly more optimistic than the code*. Several README claims, several
config comments, and one test fixture all asserted things the system did not do.
Every number in the README is now traceable to an artifact in `reports/`. Keep
it that way — it is the strongest thing this submission has.
