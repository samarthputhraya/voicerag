### Latency

Target: **< 200 ms** from transcript in to the **final** answer token (the brief's wording), measured over 200 warm runs across 2000 distinct queries.

> **Generation is SIMULATED in this run.** No LLM credentials were available, so decode timing came from `groq/openai/gpt-oss-20b (vendor-published estimate, NOT measured here)`. Retrieval, guardrail and prompt numbers are real measurements; the generation and total rows are a model of the provider, not an observation of it.

| Series | n | mean | P50 | P70 | P90 | P95 | P99 | P100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Pipeline (answered)** - transcript in -> final token out | 67 | 142.2 | 141.6 | 142.9 | 145.5 | 148.0 | 157.3 | 157.3 |
| Pipeline incl. abstentions | 200 | 53.3 | 8.8 | 138.9 | 142.9 | 144.2 | 149.9 | 157.3 |
| Time to first token | 67 | 128.1 | 127.4 | 128.8 | 131.1 | 134.0 | 143.3 | 143.3 |
| Cold (warmup runs, excluded above) | 5 | 63.2 | 14.3 | 141.5 | 144.1 | 144.1 | 144.1 | 144.1 |

#### Per-stage breakdown (warm, answered runs)

| Series | n | mean | P50 | P70 | P90 | P95 | P99 | P100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generate.total | 67 | 134.3 | 134.0 | 134.0 | 134.1 | 134.3 | 149.6 | 149.6 |
| generate.ttft | 67 | 120.2 | 120.0 | 120.0 | 120.0 | 120.1 | 128.4 | 128.4 |
| retrieve | 67 | 7.3 | 6.8 | 8.2 | 10.1 | 11.6 | 22.6 | 22.6 |
| embed | 67 | 0.3 | 0.2 | 0.3 | 0.4 | 0.5 | 1.0 | 1.0 |
| guard.abstention | 67 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 | 0.4 | 0.4 |
| guard.input | 67 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2 |
| prompt | 67 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

_Stage times come from the same `Trace` spans the API returns and the HUD renders. Spans may overlap; they are not expected to sum to the pipeline total._

**Verdict:** P100 = 157.3 ms < 200 ms target (P50 = 141.6 ms, P70 = 142.9 ms).
