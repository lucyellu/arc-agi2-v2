# Discarded-answer analysis — results_subprobs-chunk2-separated172-r2-1135s

**Question:** for each subtest, did the decoder ever produce the correct grid as one of its candidates? If yes, was it ranked in the top-2 (the attempts we submit) or discarded below that?

- Subtests with inference output: **8**
- Correct answer in top-2 (= we submitted it): **2/8**
- Correct answer anywhere in candidate list: **2/8**
- Right answer DISCARDED (model produced it, selection algo ranked it >2): **0/8**

## Per-subtest detail

| subkey | aug files | candidates | unique grids | correct rank | in top-2 | anywhere | discarded |
|---|---:|---:|---:|---:|---|---|---|
| `366c2ebb_0` | 4 | 4 | 2 | — |  |  |  |
| `50e6cde5_0` | 1 | 1 | 1 | — |  |  |  |
| `632bd3ac_0` | 11 | 15 | 11 | 1 | ✓ | ✓ |  |
| `7bc65847_0` | 15 | 19 | 9 | 1 | ✓ | ✓ |  |
| `84ba9721_0` | 4 | 5 | 4 | — |  |  |  |
| `a2a27f40_0` | 3 | 4 | 4 | — |  |  |  |
| `d624eb16_0` | 16 | 20 | 7 | — |  |  |  |
| `ff3108dc_0` | 9 | 12 | 7 | — |  |  |  |

## Interpretation

All 2 correct answers found by the model also made it into the top-2 submitted. **The selection algorithm is not the bottleneck** — when we miss, the model never produced the right grid.

Candidate-count breadth is also informative: a subtest with only 1–3 unique grids was under-explored (decode budget hit, max_score threshold too tight, or DFS branches died early). Subtests with 12+ unique grids that still missed are pure capability failures — the model explored a wide search but couldn't find the answer.