### Chunking ablation

Corpus: 19,878 deduplicated passages from 2,000 rows (validation/hinval.parquet); 400 queries, 109 duplicate passages collapsed.  
Embedder: `static:minishlab/potion-base-8M` (dim 256), fitted once on the passage corpus and shared by every row so the table isolates chunking.  
Retrieval: hybrid HNSW + BM25, fusion `rrf`, efSearch 64, query field `eng_query`. Relevance is judged on passages: chunk hits are mapped back to their source passage id before scoring.  

| Strategy | Chunks | Mean chars | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | Chunk ms | Embed ms | Build ms | Index MB | Query p50 | Query p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recursive | 19998 | 311.0 | 0.2455 | 0.7228 | 0.9070 | 0.4572 | 0.5628 | 61.4 | 1299.4 | 1573.1 | 54.9 | 1.1290 | 1.9470 |
| metadata | 20062 | 311.2 | 0.2430 | 0.7228 | 0.9024 | 0.4542 | 0.5590 | 384.9 | 1291.0 | 1640.2 | 55.2 | 1.2150 | 1.8490 |
| contextual | 19998 | 311.0 | 0.2280 | 0.7103 | 0.8987 | 0.4526 | 0.5567 | 292.5 | 1815.0 | 1711.3 | 62.9 | 1.1810 | 2.0480 |
| fixed | 20062 | 311.2 | 0.2405 | 0.7203 | 0.9024 | 0.4512 | 0.5566 | 374.4 | 2439.6 | 1528.0 | 54.7 | 1.2080 | 2.0840 |
| semantic | 32933 | 188.5 | 0.2201 | 0.7045 | 0.8699 | 0.4387 | 0.5372 | 8932.6 | 1328.8 | 2484.8 | 76.2 | 1.6990 | 2.7250 |
| sentence_window | 63475 | 97.3 | 0.2276 | 0.6757 | 0.8023 | 0.4298 | 0.5133 | 402.1 | 1891.3 | 4029.0 | 141.1 | 1.9800 | 3.5210 |

**Why each row exists**

- `recursive` -- Descends a separator hierarchy so chunks break on paragraphs and sentences rather than mid-clause.
- `metadata` -- Fixed geometry plus a title/section prefix on the embedded text only, restoring referents that bare passages lose.
- `contextual` -- Overlay that prepends a situating blurb (title + lead) to each chunk before embedding, without altering cited text.
- `fixed` -- Control. Fixed 120-word windows, 24-word overlap. The naive split every other strategy is measured against.
- `semantic` -- Boundaries placed at per-document similarity percentiles, so cuts land where the topic actually changes.
- `sentence_window` -- Retrieve small, read big: indexes one sentence, serves a 7-sentence window to the generator.

### Fusion ablation (chunking fixed at `sentence_window`)

`dense` and `sparse` are single-run controls, not fusion methods: if RRF does not beat both, the hybrid retriever is not earning its complexity.  

| Strategy | Chunks | Mean chars | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | Chunk ms | Embed ms | Build ms | Index MB | Query p50 | Query p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minmax | 63475 | 97.3 | 0.2438 | 0.6794 | 0.8111 | 0.4407 | 0.5234 | 442.4 | 2131.3 | 4350.7 | 141.1 | 1.9700 | 3.0620 |
| zscore | 63475 | 97.3 | 0.2413 | 0.6782 | 0.8111 | 0.4408 | 0.5232 | 442.4 | 2131.3 | 4350.7 | 141.1 | 2.0460 | 3.1150 |
| rrf | 63475 | 97.3 | 0.2301 | 0.6757 | 0.8023 | 0.4314 | 0.5145 | 442.4 | 2131.3 | 4350.7 | 141.1 | 1.9510 | 2.8380 |
| sparse | 63475 | 97.3 | 0.2476 | 0.6632 | 0.7628 | 0.4207 | 0.4972 | 442.4 | 2131.3 | 4350.7 | 141.1 | 2.0540 | 3.1090 |
| dense | 63475 | 97.3 | 0.2151 | 0.6503 | 0.7511 | 0.3991 | 0.4794 | 442.4 | 2131.3 | 4350.7 | 141.1 | 2.0400 | 2.9960 |

**Why each row exists**

- `minmax` -- Weighted sum of min-max normalised scores: keeps within-run margins, but one outlier rescales the whole run.
- `zscore` -- Weighted sum of z-scored runs: margin-preserving and outlier-tolerant, assumes roughly symmetric score distributions.
- `rrf` -- Reciprocal Rank Fusion: rank-only, so a run with an uncalibrated score scale cannot dominate.
- `sparse` -- Control: BM25 only. Isolates what the dense run contributes.
- `dense` -- Control: HNSW only. Isolates what the sparse run contributes.

### Abstention (MS MARCO 'No Answer Present.' labels)

Retriever: `sentence_window` chunking, hybrid `rrf` fusion, identical to the ablation above.  
Labels are MS MARCO's own: a query is unanswerable when every candidate passage is `is_selected == 0` and the gold answer is `"No Answer Present."`.  

781 unanswerable and 400 answerable queries; decision threshold 0.500 (prior rules). Positive class = *should abstain*.

| Precision | Recall | F1 | Accuracy | Balanced acc. | False-abstention rate |
|---:|---:|---:|---:|---:|---:|
| 0.636 | 0.009 | 0.018 | 0.341 | 0.499 | 0.010 |

**Confusion matrix**

| | predicted answer | predicted abstain |
|---|---:|---:|
| **gold answerable** | 396 | 4 |
| **gold unanswerable** | 774 | 7 |

**Most confident mistakes**

- `why does my knee hurt on and off` -- answered an unanswerable query at confidence 0.12
- `glyc root word meaning` -- answered an unanswerable query at confidence 0.14
- `how long does a physician have to sign an order for observation` -- answered an unanswerable query at confidence 0.14
- `how long does rough hair shrink` -- answered an unanswerable query at confidence 0.16
- `difference between a celsius and pounds` -- answered an unanswerable query at confidence 0.16

**Calibrated on a train split, scored on held-out data**

| Gate | Precision | Recall | F1 | False-abstention |
|---|---:|---:|---:|---:|
| prior thresholds | 1.000 | 0.010 | 0.019 | 0.000 |
| calibrated (threshold 0.463) | 0.666 | 0.933 | 0.777 | 0.912 |

_709 training / 472 held-out examples._

**Threshold sweep (prior gate)**

| Threshold | Precision | Recall | F1 | Accuracy | False-abstention |
|---:|---:|---:|---:|---:|---:|
| 0.100 | 0.661 | 1.000 | 0.796 | 0.661 | 1.000 |
| 0.204 | 0.657 | 0.942 | 0.774 | 0.636 | 0.963 |
| 0.225 | 0.644 | 0.827 | 0.724 | 0.583 | 0.892 |
| 0.249 | 0.634 | 0.718 | 0.673 | 0.539 | 0.810 |
| 0.270 | 0.640 | 0.629 | 0.634 | 0.521 | 0.690 |
| 0.282 | 0.633 | 0.526 | 0.575 | 0.485 | 0.595 |
| 0.290 | 0.635 | 0.433 | 0.515 | 0.461 | 0.485 |
| 0.297 | 0.656 | 0.347 | 0.454 | 0.448 | 0.355 |
| 0.300 | 0.677 | 0.301 | 0.417 | 0.443 | 0.280 |
| 0.305 | 0.665 | 0.201 | 0.309 | 0.405 | 0.198 |
| 0.311 | 0.706 | 0.108 | 0.187 | 0.380 | 0.087 |
| 0.400 | 0.636 | 0.009 | 0.018 | 0.341 | 0.010 |
| 0.600 | 0.636 | 0.009 | 0.018 | 0.341 | 0.010 |
| 0.800 | 0.000 | 0.000 | 0.000 | 0.339 | 0.000 |
