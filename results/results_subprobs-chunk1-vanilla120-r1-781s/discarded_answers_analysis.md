# Discarded-answer analysis — results_subprobs-chunk1-vanilla120-r1-781s

**Question:** for each subtest, did the decoder ever produce the correct grid as one of its candidates? If yes, was it ranked in the top-2 (the attempts we submit) or discarded below that?

- Subtests with inference output: **7**
- Correct answer in top-2 (= we submitted it): **2/7**
- Correct answer anywhere in candidate list: **3/7**
- Right answer DISCARDED (model produced it, selection algo ranked it >2): **1/7**

## Per-subtest detail

| subkey | aug files | candidates | unique grids | correct rank | in top-2 | anywhere | discarded |
|---|---:|---:|---:|---:|---|---|---|
| `1ae2feb7_0` | 2 | 2 | 1 | — |  |  |  |
| `1ae2feb7_1` | 16 | 27 | 7 | 7 |  | ✓ | YES |
| `1ae2feb7_2` | 14 | 16 | 7 | 1 | ✓ | ✓ |  |
| `20270e3b_0` | 11 | 16 | 13 | — |  |  |  |
| `20270e3b_1` | 15 | 16 | 4 | 2 | ✓ | ✓ |  |
| `20a9e565_0` | 1 | 1 | 1 | — |  |  |  |
| `20a9e565_1` | 6 | 6 | 4 | — |  |  |  |

## Interpretation

**1 subtest(s) had the right answer hidden in the candidate list.** Improving the selection algorithm could potentially recover these.

Candidate-count breadth is also informative: a subtest with only 1–3 unique grids was under-explored (decode budget hit, max_score threshold too tight, or DFS branches died early). Subtests with 12+ unique grids that still missed are pure capability failures — the model explored a wide search but couldn't find the answer.