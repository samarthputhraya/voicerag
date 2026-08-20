### Abstention (MS MARCO 'No Answer Present.' labels)

> Measured on the pre-ablation `sentence_window` build, not the shipped `recursive` index. The finding — the retrieval-signal gate barely separates answerable from unanswerable — is independently confirmed on the shipped 197,511-chunk index by the gate phase of `reports/answer_quality.json`: refusal rates 12.0% vs 16.8%, medians 0.188 vs 0.251, over 250 queries per class.

Retriever: `sentence_window` chunking, hybrid `rrf` fusion, identical to `reports/ablation.md`.  
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
