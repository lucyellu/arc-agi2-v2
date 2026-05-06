# Discarded-answer analysis — results_subprobs-chunk2-vanilla120-r2-1691s

**Question:** for each subtest, did the decoder ever produce the correct grid as one of its candidates? If yes, was it ranked in the top-2 (the attempts we submit) or discarded below that?

- Subtests with inference output: **8**
- Correct answer in top-2 (= we submitted it): **2/8**
- Correct answer anywhere in candidate list: **2/8**
- Right answer DISCARDED (model produced it, selection algo ranked it >2): **0/8**

## Per-subtest detail

| subkey | aug files | candidates | unique grids | correct rank | in top-2 | anywhere | discarded |
|---|---:|---:|---:|---:|---|---|---|
| `221dfab4_0` | 15 | 19 | 6 | — |  |  |  |
| `221dfab4_1` | 13 | 16 | 7 | — |  |  |  |
| `247ef758_0` | 2 | 2 | 2 | — |  |  |  |
| `247ef758_1` | 7 | 9 | 7 | — |  |  |  |
| `269e22fb_0` | 3 | 3 | 2 | — |  |  |  |
| `269e22fb_1` | 1 | 1 | 1 | — |  |  |  |
| `2d0172a1_0` | 13 | 17 | 8 | 1 | ✓ | ✓ |  |
| `2d0172a1_1` | 9 | 15 | 10 | 1 | ✓ | ✓ |  |

## Interpretation

All 2 correct answers found by the model also made it into the top-2 submitted. **The selection algorithm is not the bottleneck** — when we miss, the model never produced the right grid.

Candidate-count breadth is also informative: a subtest with only 1–3 unique grids was under-explored (decode budget hit, max_score threshold too tight, or DFS branches died early). Subtests with 12+ unique grids that still missed are pure capability failures — the model explored a wide search but couldn't find the answer.