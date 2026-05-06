# agi-2 Audit For agi2-v2

Audit date: 2026-05-05

## What Was Kept

- `data/arc-prize-2026-arc-agi-2/`: core AGI-2 data only.
- `scripts/arc_loader.py`: useful dataset, augmentation, and Qwen formatting code.
- `scripts/arc_decoder.py`: compact voting and dedupe code for bz2/pickle inference outputs.
- `scripts/constraint_filter.py`: strongest cheap heuristic in the repo. Keep applying it before attempt selection.
- `scripts/score_submissions.py`: retained, made root-relative, and fixed to score all 172 eval subtests instead of only the first test output per puzzle.
- `scripts/viz_arc.py`, `tools/`: retained for quick human inspection.
- `scripts/infer_qwen.py`: retained for prompt-based Qwen runs.
- `scripts/infer_trm.py`: retained as a scaffold, but external TRM code/checkpoints must be restored.
- `notebooks/reference/nvarc-2025-bf16-faster-ttt-via-early-stopping.ipynb`: clean NVARC-style base.
- `notebooks/reference/v13-multitask-fast.ipynb`: best confirmed leaderboard reference from inherited notes.

## What Was Left Behind

- `results_best/`: mostly historical run outputs, compiled caches, placeholder submissions, and stale copies of scripts.
- `notebooks/v20*`, `v21*`, `v22_chain*`: useful as history, but not clean foundations.
- `data/arc-prize-2026-arc-agi-3/`: not needed for a fresh AGI-2 workspace and included wheels/git data.
- Kaggle push/build helper scripts: old notebook packaging scripts tied to the previous iteration.
- `.docx` planning artifacts and archive notes.

## Main Findings

- The repo was not a git repository, so all work was copied into a new sibling folder without mutating the original.
- The compact reusable code is concentrated in `scripts/`, not in the run archives.
- The old state notes say the clean NVARC base and v13 are the most reliable references.
- The v19-v22 feedback/warmup line is not a good base: inherited notes show crashes, blank outputs, and likely pipeline fragility.
- The inherited scorer undercounted multi-test puzzles; v2 reports both subtest accuracy and fully solved puzzles.
- `infer_trm.py` referenced NVARC/TRM paths that were not present in the copied project. v2 now makes those external requirements explicit via `external/` and environment variables.

## Recommended Fresh Attack

1. Build a delta-format experiment. Generate changed-cell lists, reconstruct full grids, and score against eval.
2. Add a program synthesis path. Generate small Python transforms, keep candidates that pass all train pairs, and ensemble their predictions with Qwen/TRM.
3. Use `constraint_filter.py` as a universal reranker across every method.
4. Keep every experiment reproducible: inputs, outputs, score report, and a short note in `experiments/`.

## Risk Notes

- Qwen prompt inference is usable, but it is not the same as the NVARC TTT approach that produced the strongest historical score.
- TRM cannot run until external code and checkpoints are restored.
- Local scoring uses the 120-puzzle evaluation set only. Kaggle leaderboard uses a separate test set.
