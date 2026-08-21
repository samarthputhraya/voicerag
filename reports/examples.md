# Verified example questions

Every question below was run through the **whole** pipeline -- retrieval,
abstention gate, generation and grounding -- against the index this
deployment serves, and answered. They are what `GET /examples` offers.

The rejected list matters more than the kept one. Those questions are
labelled `answerable: True` by MS MARCO and are refused anyway, because
the labelled gold answer does not address what the question asks. A chip
is a promise; these are the promises this corpus cannot keep.

- index: `C:\Users\samar\OneDrive\Documents\voicerag\data\index_20k` (197,511 chunks, recursive, `static:minishlab/potion-base-8M`)
- candidates tried: 30
- answered: 22 · declined: 8

## Published

| question | grounding | citations |
|---|---:|---:|
| How long for cantaloupe to mature? | 1.00 | 1 |
| Do indians eat rice? | 1.00 | 2 |
| How much is it per day when locked up in county jail? | 1.00 | 1 |
| What is bayern munich? | 1.00 | 1 |
| What is a corporation? | 0.90 | 2 |
| How to print an excel sheet? | 0.89 | 1 |
| How long does it take cracked ribs to heal? | 0.88 | 1 |
| How do i pit cherries? | 0.86 | 2 |
| How effective are sassy water? | 0.84 | 3 |
| What is basal in dna? | 0.83 | 1 |

## Rejected — labelled answerable, declined anyway

| question | what the system said |
|---|---|
| How fast does an eagle travel? | The indexed passages have nothing on “fast eagle travel”, so I'd be guessing: only 0% of the top passages were found by  |
| What is barter system and its problems? | I drafted an answer but 1 statement(s) weren't supported by the retrieved passages, so I won't state them. The first was |
| How long should you carb cycle? | The indexed passages have nothing on “long carb cycle”, so I'd be guessing: the best passage only reaches 0.54 similarit |
| How long is shoulder recovery? | The model read the retrieved passages and judged them insufficient, so it declined to answer rather than guess. |
| How far is philadelphia from lancaster pa? | The model read the retrieved passages and judged them insufficient, so it declined to answer rather than guess. |
| How much supervisor pay rate for cvs warehouse? | The model read the retrieved passages and judged them insufficient, so it declined to answer rather than guess. |
| What is battery life f? | The model read the retrieved passages and judged them insufficient, so it declined to answer rather than guess. |
| What is bay crest partners? | The indexed passages have nothing on “bay crest partners”, so I'd be guessing: the best passage only reaches 0.43 simila |

## Re-verified against the live deployment, 21 Aug 2026

Every question above was re-run against <https://voicerag-demo.duckdns.org>.
**All 10 published questions still answer**, grounding 0.83–1.00. **Six of the
eight rejections still decline. Two now answer:**

| question | then | now |
|---|---|---|
| How long should you carb cycle? | declined, best passage 0.54 | **answers** — "Carb cycle 3 days low, 1 day high. [1]" |
| What is bay crest partners? | declined, best passage 0.43 | **answers**, grounding 1.00, with a cited passage |

This is recorded rather than quietly edited, because the drift is more
interesting than the table.

Note which ones moved. Both were **retrieval-gate** declines — the mechanism
that reads similarity and retriever agreement, and the one that ought to be
deterministic for a fixed index. The four declines that came from the *model*
reading the passages and judging them insufficient all held. That is the
opposite of what we would have predicted, and we are not claiming a cause we
have not established: the chunk count in this deployment matches the one this
file was generated against, so a plain rebuild does not explain it.

What follows for anyone relying on this file: **a decline listed here is
evidence, not a guarantee.** If you need a question that reliably refuses — for
a demo, or a regression test — re-run it first. `How fast does an eagle travel?`
was re-verified 5/5 on this date, refusing with the same wording every time.

