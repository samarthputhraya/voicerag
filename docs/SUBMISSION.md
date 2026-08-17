# Submission playbook — HH Goa 2026, Task 2

**Deadline: Saturday 22 August 2026, 11:59 PM IST.**
**No resubmissions.** The form is a one-shot. Everything below gets tested
before anyone opens it.

Form: https://forms.gle/MNvCjcv23Hn2Eeu58

---

## Deliverables checklist

| # | Deliverable | Owner | Status |
|---|---|---|---|
| 1 | Submission form filled | — | ☐ |
| 2 | GitHub repo, public | — | ☐ |
| 3 | Live working link | — | ☐ |
| 4 | Video 1 — team/process, 90s | — | ☐ |
| 5 | Video 2 — demo, end to end | — | ☐ |
| 6 | Both videos on Instagram ×3 members | all | ☐ |
| 7 | Both videos on X ×3 members | all | ☐ |
| 8 | Both videos on LinkedIn ×3 members | all | ☐ |
| 9 | ≥1 Instagram account public | — | ☐ |
| 10 | `#RAGInGoa` on every post | all | ☐ |

That is **9 social posts** (3 members × 3 platforms), each carrying both videos
and the hashtag. Miss one and the submission is non-compliant on a technicality.
Do them together, in one sitting, with a shared checklist open.

---

## Video 1 — Team / process (90 seconds, hard cap)

**The brief says process, not product.** Most teams will misread this and submit
a second demo. Showing how you actually worked is free differentiation.

Do not narrate features. Show the work.

| Time | Shot | Voiceover |
|---|---|---|
| 0:00–0:08 | Whiteboard/paper with the latency budget written as a column of numbers, someone crossing one out and writing a smaller one | "Two hundred milliseconds, end to end. We started by writing down where every one of them goes." |
| 0:08–0:22 | Screen: the six chunking strategy files in the editor, scrolling | "Six chunking strategies, not one. We didn't pick a favourite — we built all six and made them compete." |
| 0:22–0:36 | Screen: ablation table printing in a terminal, rows filling in | "Then we measured. Recall, MRR, nDCG, against real human relevance judgements from the dataset." |
| 0:36–0:50 | Two people at a laptop, one pointing at a number; cut to a VAD constant being changed from 1400 to 260 | "Our biggest win wasn't a model. It was a browser setting that waits 1.4 seconds before deciding you've stopped talking. We found it on day two." |
| 0:50–1:04 | Screen: tests running, green | "Every strategy has tests. Every guardrail has tests. The whole suite runs offline, so it runs anywhere." |
| 1:04–1:18 | Split screen: someone speaking into a laptop, latency HUD bars painting | "We instrumented every stage, so the demo reports the same numbers the benchmark does. No cherry-picked runs." |
| 1:18–1:30 | Three of you, to camera, brief | "Three of us, nine days. See you in Goa." |

**Shooting notes**

- Real screens, real terminals. Do not mock anything.
- Show at least one thing that *failed* and got fixed — a red test going green, a bad number improving. Process videos that show only success read as staged.
- Phone footage is fine. Audio quality matters more than video quality: use a phone close to the speaker, not the laptop mic.
- Cut hard at 90 seconds. Going over is the easiest disqualification on the list.

---

## Video 2 — Demo, end to end

No stated length; **aim for 2:00–2:30.** Judges watch many of these. Front-load
the proof.

| Time | Beat | What to show |
|---|---|---|
| 0:00–0:10 | Cold open — no intro, no logo | Speak a question into the mic. Answer streams back with citations. Latency HUD lands under 200ms. Say nothing until it's done. |
| 0:10–0:25 | Name what just happened | "Voice in, grounded answer out. That was 180 milliseconds from transcript to final token — measured, not estimated." |
| 0:25–0:45 | The HUD | Point at the waterfall: guard, embed, dense ∥ sparse, fusion, abstention, generation. Show P50 / P70 / P100 accumulating across several questions. |
| 0:45–1:05 | **Ask something the corpus cannot answer** | Show it *decline*, with the reason and the retrieval signals that triggered it. Say: "The brief asked for a system that knows when not to answer. This is that." |
| 1:05–1:25 | Speculative retrieval | Ask a longer question. Point at the speculative-search counter incrementing while you are still speaking, then the hit indicator on the final. "Retrieval finished before I stopped talking." |
| 1:25–1:45 | Cross-lingual | Switch to Hindi. Ask in Hindi, retrieve from English passages, answer grounded. "Same index. Same ground truth." |
| 1:45–2:05 | The ablation table | On screen briefly. "Six chunking strategies, scored against human relevance judgements. We shipped the one that won, and we can show you why." |
| 2:05–2:20 | Close | Repo URL and live URL on screen, held for 3 full seconds so they're readable on a phone. |

**Rules for this shoot**

- **One continuous take for the opening question.** No cuts between speaking and the answer arriving — a cut there looks like editing out a wait.
- Show a **failure or refusal**. A demo where everything works looks rehearsed; a demo that shows the system correctly refusing looks engineered.
- Have the HUD visible for the entire video.
- Do a real run right before recording. Cold starts are the classic demo killer — the service must be warm.

---

## Social copy

Each member posts **both videos** to **Instagram, X, LinkedIn** with `#RAGInGoa`.
Vary the wording per person — three identical posts read as spam and reflect badly.

### X — version A

> We built a voice RAG pipeline that answers in under 200ms — transcript to final token.
>
> Six chunking strategies, ablated against real human relevance judgements. Hybrid dense+BM25 retrieval. Guardrails that decline when the corpus can't answer.
>
> Biggest speedup wasn't a model. It was a VAD constant.
>
> #RAGInGoa

### X — version B

> Speak a question → grounded answer with citations, in <200ms.
>
> The trick: we fire retrieval on *partial* transcripts. By the time you stop talking, the search is already done.
>
> Built for @247pmstudio HH Goa 2026.
>
> #RAGInGoa

### X — version C

> Most RAG demos pick one chunking strategy and hope.
>
> We built six, scored them on Recall@k / MRR / nDCG against human labels, and shipped the winner.
>
> Voice in, cited answer out, under 200ms.
>
> #RAGInGoa

### LinkedIn (adapt per person)

> **Voice-enabled RAG, under 200 milliseconds.**
>
> For HH Goa 2026's shortlisting task we built a pipeline that takes a spoken question and returns an answer grounded in MS MARCO, with citations — transcript to final token in under 200ms, measured at P50/P70/P100 across real queries, not a single lucky run.
>
> Three things we're proud of:
>
> • **Six chunking strategies, not one.** Fixed, recursive, sentence-window, metadata-aware, semantic and contextual — all implemented, then ablated against the dataset's own human relevance judgements. We can show you why the winner won.
>
> • **Speculative retrieval.** Search fires on partial transcripts while you're still speaking, so retrieval is usually finished before you stop.
>
> • **It knows when not to answer.** Abstention is driven by retrieval signals — score gap, entropy, dense/sparse agreement — and evaluated against genuinely labelled unanswerable queries from the dataset, not hand-written examples.
>
> The single biggest latency win wasn't in the model layer. The standard browser voice-activity library waits 1.4 seconds of silence before deciding you've stopped talking. We run it at 260ms.
>
> Repo and demo below. Built with [teammates].
>
> #RAGInGoa

### Instagram (caption — keep it short, the video carries it)

> Speak a question. Get a cited answer. Under 200ms. ⚡
>
> Six chunking strategies, benchmarked against real human relevance labels. Retrieval that starts before you finish talking. And guardrails that say "I don't know" when the corpus genuinely doesn't.
>
> Built for HH Goa 2026 🌴
>
> #RAGInGoa

**Reminder:** at least one Instagram account must be **public**. Check this
before posting, not after.

---

## Submission form — prepared answers

Draft these before opening the form. No resubmissions.

**What did you build?**
> A voice-enabled RAG system. A spoken question is transcribed by Sarvam's realtime WebSocket STT, retrieved against a hybrid dense + BM25 index over MS MARCO passages from ai4bharat/MSMARCO-XI, and answered by a grounded generation step with inline citations. Transcript to final answer token completes in under 200ms, measured at P50/P70/P100 across real queries.

**Chunking approach**
> Six strategies implemented behind one interface and ablated against the dataset's own relevance judgements: fixed-size with overlap (control), recursive separator-hierarchy, sentence-window (retrieve small / read big), metadata-aware prefixing, semantic percentile-boundary, and contextual enrichment. Selection was made on measured Recall@1/5/10, MRR@10 and nDCG@10 rather than by preference. Retrieval is hybrid — dense HNSW and BM25 run concurrently and fuse via Reciprocal Rank Fusion, with component ranks preserved for the abstention logic.

**Latency numbers**
> P50 / P70 / P100: `<fill from reports/latency.json>`. Measured across `<n>` real queries with warmup runs excluded. Per-stage breakdown published in the repo. Percentiles use the nearest-rank method. Speech-to-text and client network latency are measured and reported separately, matching the brief's own pipeline decomposition.

**Harness**
> A structured orchestration layer: a single monotonic Deadline threaded through every stage so a retry that cannot finish in the remaining budget is never attempted; exponential backoff with full jitter; an explicit transient/permanent error taxonomy so non-retryable failures are not retried; per-provider circuit breakers with half-open probing; and ordered provider fallback. Every stage is traced, and every latency number we publish comes from those traces.

**Guardrails**
> Three stages ordered by cost. Input guard (sub-millisecond): filler/silence transcripts, prompt-injection heuristics, inappropriate content. Abstention (post-retrieval, pre-generation): combines max score, top1–top2 gap, top-k entropy and dense/sparse agreement, with thresholds calibrated from data. When it fires we return early and never call the LLM. Grounding verification (concurrent with streaming): sentence-level claim extraction, lexical entailment against retrieved chunks, exact verification of numbers and dates, and citation validation. Abstention is evaluated against genuinely labelled unanswerable queries — MS MARCO rows where every candidate passage is marked non-relevant — reporting precision/recall/F1.

---

## Plan, 16–22 August

| Day | Focus | Done when |
|---|---|---|
| **Sun 16** | Core build complete. Sarvam + Groq keys obtained. | Test suite green |
| **Mon 17** | Ingest the real shard. Build the index. First real end-to-end run. | A spoken question returns a real cited answer |
| **Tue 18** | Run the ablation. Tune thresholds. Calibrate abstention on the labelled set. | `reports/ablation.md` exists with real numbers |
| **Wed 19** | Deploy API to Render, UI to Vercel. Measure production latency. | Live URL works from a phone on mobile data |
| **Thu 20** | Record both videos. Reshoot anything weak. | Both videos exported |
| **Fri 21** | Fill README with final numbers. Full dry run of every link. | Every checklist item ☑ except posting |
| **Sat 22** | Post to 9 social targets. Submit the form. **Early.** | Submitted by 6 PM IST, not 11:58 |

**Do not submit at 11:58 PM.** The form is one-shot and traffic spikes at the
deadline. Target Saturday afternoon.

---

## Pre-submission checks

Run these before touching the form.

- [ ] Live URL loads in a **private window** (catches "works because I'm logged in")
- [ ] Live URL works on a **phone, on mobile data** (catches localhost and CORS)
- [ ] Microphone permission prompt appears and works on that phone
- [ ] Cold-start a fresh request — first-request latency is acceptable
- [ ] Repo is **public**; clone it to a clean folder and follow your own README from scratch
- [ ] No API keys committed — `git log -p | grep -iE "sk-|api_key|subscription"`
- [ ] `.env.example` present, `.env` **not** committed
- [ ] Both video links are public and playable in a private window
- [ ] All 9 social posts live, each with both videos and `#RAGInGoa`
- [ ] At least one Instagram account confirmed public
- [ ] README latency and ablation tables filled with **real** numbers, no placeholders
- [ ] Every claim in the README is reproducible by a stranger running your scripts
