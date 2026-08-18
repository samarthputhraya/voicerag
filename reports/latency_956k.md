### Latency

> ⚠️ **NOT the shipped configuration. This is the measurement that decided which index ships.**
>
> We built a 956,128-chunk index over the full 950,526-passage validation shard —
> 4.8x the 197,511 chunks the system serves — to find out whether more corpus was
> worth more latency. It is not. Against the same 2,000-query pool, the same
> nearest-rank definition and the same modelled decoder, the larger index misses
> the brief's 200 ms bar:
>
> | Index | chunks | retrieve P50 | retrieve P100 | pipeline P100 | verdict |
> |---|---:|---:|---:|---:|---|
> | **shipped** (`reports/latency.md`) | 197,511 | 6.8 ms | 22.6 ms | **157.3 ms** | passes |
> | this run | 956,128 | 51.9 ms | 84.2 ms | **283.3 ms** | fails |
>
> The cost is BM25, not the vector index. Measured separately on the 956k build:
> dense HNSW search is **0.82 ms P50 / 1.49 ms P100**, while BM25 over the same
> chunks is **12.9 ms P50 / 73.4 ms P100** — and the spread tracks posting-list
> length, not query length ("how to print an excel sheet" costs 44 ms; "how to
> make hat on graduation cake pops", two words longer, costs 8.8 ms). Preloading
> the BM25 arrays (`MMAP_SPARSE=false`) removes a further ~100 ms of first-touch
> page faults and is now the default, but it does not close the gap.
>
> So the shipped index is the smaller one, on purpose, and this file is the
> evidence for that choice rather than a result we are hiding.



Target: **< 200 ms** from transcript in to the **final** answer token (the brief's wording), measured over 200 warm runs across 2000 distinct queries.

> **Generation is SIMULATED in this run.** The decoder was pinned to the simulated profile with --force-simulated, so the run is reproducible and consumes no provider quota. Decode timing came from `groq/openai/gpt-oss-20b (vendor-published estimate, NOT measured here)`. Retrieval, guardrail and prompt numbers are real measurements; the generation and total rows are a model of the provider, not an observation of it.

| Series | n | mean | P50 | P70 | P90 | P95 | P99 | P100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Pipeline (answered)** - transcript in -> final token out | 78 | 209.8 | 212.6 | 225.4 | 244.2 | 266.3 | 283.3 | 283.3 |
| Pipeline incl. abstentions | 200 | 125.5 | 100.2 | 194.4 | 229.5 | 241.6 | 273.7 | 283.3 |
| Time to first token | 78 | 195.8 | 198.5 | 211.3 | 230.2 | 252.2 | 269.3 | 269.3 |
| Cold (warmup runs, excluded above) | 5 | 158.2 | 160.8 | 223.1 | 242.9 | 242.9 | 242.9 | 242.9 |

#### Per-stage breakdown (warm, answered runs)

| Series | n | mean | P50 | P70 | P90 | P95 | P99 | P100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generate.total | 78 | 134.1 | 134.1 | 134.1 | 134.2 | 134.2 | 135.1 | 135.1 |
| generate.ttft | 78 | 120.1 | 120.0 | 120.0 | 120.1 | 120.1 | 120.6 | 120.6 |
| retrieve | 78 | 73.5 | 76.9 | 89.7 | 107.6 | 128.8 | 147.8 | 147.8 |
| embed | 78 | 0.7 | 0.7 | 0.8 | 0.9 | 1.0 | 1.4 | 1.4 |
| prompt | 78 | 0.7 | 0.1 | 0.2 | 0.3 | 0.3 | 38.7 | 38.7 |
| guard.abstention | 78 | 0.4 | 0.3 | 0.5 | 0.5 | 0.7 | 0.8 | 0.8 |
| guard.input | 78 | 0.2 | 0.2 | 0.2 | 0.3 | 0.4 | 0.5 | 0.5 |

_Stage times come from the same `Trace` spans the API returns and the HUD renders. Spans may overlap; they are not expected to sum to the pipeline total._

**Verdict:** P100 = 283.3 ms >= 200 ms target (P50 = 212.6 ms, P70 = 225.4 ms).
