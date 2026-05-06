# Discarded-answer analysis — results_subprobs-vanilla120-r1-1187s-chunk0

**Question:** for each subtest, did the decoder ever produce the correct grid as one of its candidates? If yes, was it ranked in the top-2 (the attempts we submit) or discarded below that?

- Subtests with inference output: **6**
- Correct answer in top-2 (= we submitted it): **0/6**
- Correct answer anywhere in candidate list: **0/6**
- Right answer DISCARDED (model produced it, selection algo ranked it >2): **0/6**

## Per-subtest detail

| subkey | aug files | candidates | unique grids | correct rank | in top-2 | anywhere | discarded |
|---|---:|---:|---:|---:|---|---|---|
| `13e47133_0` | 3 | 3 | 3 | — |  |  |  |
| `142ca369_0` | 6 | 6 | 5 | — |  |  |  |
| `142ca369_1` | 13 | 14 | 8 | — |  |  |  |
| `16de56c4_1` | 2 | 2 | 2 | — |  |  |  |
| `195c6913_0` | 1 | 1 | 1 | — |  |  |  |
| `195c6913_1` | 2 | 2 | 1 | — |  |  |  |

## Interpretation

No correct answers found in any candidate list. Pure capability/budget failures, not selection.

Candidate-count breadth is also informative: a subtest with only 1–3 unique grids was under-explored (decode budget hit, max_score threshold too tight, or DFS branches died early). Subtests with 12+ unique grids that still missed are pure capability failures — the model explored a wide search but couldn't find the answer.