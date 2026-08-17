> # ⚠ SUPERSEDED
>
> This run measured the **first** index built: 622,699 chunks, `sentence_window`
> chunking, `lsa:256` embeddings. Neither is what ships.
>
> It is kept, rather than deleted, because it is the evidence for two claims the
> README makes about *why* the build changed:
>
> * `lsa:256` costs **20.3 ms P50 / 53.2 ms P100** per query encode, against
>   0.2 / 1.0 ms for `static:minishlab/potion-base-8M`. That tail alone pushed
>   retrieval past its slice of the budget and the verdict below is **P100
>   215.8 ms — a FAIL**.
> * `sentence_window` produced 3× the chunks of `recursive` for worse recall.
>
> The shipped numbers are in [`latency.md`](latency.md): P50 141.6 / P70 142.9 /
> **P100 157.3 ms — PASS**.

### Latency

Target: **< 200 ms** from transcript in to the **final** answer token (the brief's wording), measured over 200 warm runs across 2000 distinct queries.

> **Generation is SIMULATED in this run.** No LLM credentials were available, so decode timing came from `groq/openai/gpt-oss-20b (vendor-published estimate, NOT measured here)`. Retrieval, guardrail and prompt numbers are real measurements; the generation and total rows are a model of the provider, not an observation of it.

| Series | n | mean | P50 | P70 | P90 | P95 | P99 | P100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Pipeline (answered)** - transcript in -> final token out | 45 | 181.2 | 177.9 | 181.3 | 198.2 | 206.3 | 215.8 | 215.8 |
| Pipeline incl. abstentions | 200 | 77.7 | 46.4 | 62.8 | 178.5 | 182.9 | 206.3 | 215.8 |
| Time to first token | 45 | 167.7 | 164.3 | 167.3 | 189.3 | 192.5 | 201.7 | 201.7 |
| Cold (warmup runs, excluded above) | 5 | 43.6 | 41.7 | 48.4 | 51.2 | 51.2 | 51.2 | 51.2 |

#### Per-stage breakdown (warm, answered runs)

| Series | n | mean | P50 | P70 | P90 | P95 | P99 | P100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generate.total | 45 | 134.9 | 134.1 | 134.1 | 134.9 | 138.0 | 150.6 | 150.6 |
| generate.ttft | 45 | 121.3 | 120.0 | 120.0 | 120.1 | 120.9 | 150.5 | 150.5 |
| retrieve | 45 | 23.5 | 22.4 | 24.1 | 28.3 | 32.8 | 53.4 | 53.4 |
| embed | 45 | 22.4 | 20.3 | 22.9 | 27.3 | 37.5 | 53.2 | 53.2 |
| guard.abstention | 45 | 0.2 | 0.2 | 0.2 | 0.3 | 0.4 | 0.7 | 0.7 |
| guard.input | 45 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 |
| prompt | 45 | 0.1 | 0.0 | 0.1 | 0.1 | 0.1 | 0.3 | 0.3 |

_Stage times come from the same `Trace` spans the API returns and the HUD renders. Spans may overlap; they are not expected to sum to the pipeline total._

**Verdict:** P100 = 215.8 ms >= 200 ms target (P50 = 177.9 ms, P70 = 181.3 ms).
