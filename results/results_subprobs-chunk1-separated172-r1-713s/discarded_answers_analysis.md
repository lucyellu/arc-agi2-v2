# Discarded-answer analysis — results_subprobs-chunk1-separated172-r1-713s

**Question:** for each subtest, did the decoder ever produce the correct grid as one of its candidates? If yes, was it ranked in the top-2 (the attempts we submit) or discarded below that?

- Subtests with inference output: **7**
- Correct answer in top-2 (= we submitted it): **1/7**
- Correct answer anywhere in candidate list: **1/7**
- Right answer DISCARDED (model produced it, selection algo ranked it >2): **0/7**

## Per-subtest detail

| subkey | aug files | candidates | unique grids | correct rank | in top-2 | anywhere | discarded |
|---|---:|---:|---:|---:|---|---|---|
| `10844438_0` | 6 | 6 | 3 | — |  |  |  |
| `2d34f9a6_0` | 16 | 17 | 6 | — |  |  |  |
| `34c37912_0` | 11 | 14 | 14 | — |  |  |  |
| `5307dcfa_0` | 16 | 23 | 6 | — |  |  |  |
| `667d5e11_0` | 5 | 6 | 4 | — |  |  |  |
| `e583ed11_0` | 15 | 17 | 7 | 1 | ✓ | ✓ |  |
| `faca0f18_0` | 2 | 2 | 2 | — |  |  |  |

## Interpretation

All 1 correct answers found by the model also made it into the top-2 submitted. **The selection algorithm is not the bottleneck** — when we miss, the model never produced the right grid.

Candidate-count breadth is also informative: a subtest with only 1–3 unique grids was under-explored (decode budget hit, max_score threshold too tight, or DFS branches died early). Subtests with 12+ unique grids that still missed are pure capability failures — the model explored a wide search but couldn't find the answer.