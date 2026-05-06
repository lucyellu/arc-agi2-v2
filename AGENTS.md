# AGENTS.md

Guidance for Codex when working in this fresh ARC-AGI-2 workspace.

## Project Shape

This repo is a cleaned AGI-2 restart. Treat `scripts/` as reusable infrastructure and `docs/history/` as inherited context, not gospel.

Useful retained pieces:

- `scripts/score_submissions.py`: local eval scorer for the 120-puzzle evaluation set.
- `scripts/constraint_filter.py`: low-cost color-count and size-change constraint scoring.
- `scripts/arc_loader.py`: ARC dataset loading, augmentation, and Qwen ChatML formatting.
- `scripts/arc_decoder.py`: bz2/pickle attempt dedupe and selection scoring.
- `scripts/infer_qwen.py`: prompt-based Qwen runner.
- `scripts/infer_trm.py`: TRM runner scaffold. Requires external TRM code/checkpoints.
- `tools/`: local submission viewer.

Do not build new work on the old feedback-loop notebooks from v19-v22 unless the user explicitly asks. The inherited state notes mark those as broken or inconclusive.

## Run Commands

Smoke test:

```powershell
python tests/test_smoke.py
```

Score a submission:

```powershell
python scripts/make_copy_baseline.py
python scripts/score_submissions.py results/submissions/latest/copy_eval_submission.json
```

Run Qwen:

```powershell
cd scripts
python infer_qwen.py `
  --challenges ../data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json `
  --output ../results/submissions/latest/qwen_submission.json `
  --attempts 3
```

Submission viewer lives in the separate `lucyellu/arc-agi2-viewer` repo.

## Direction

Prefer fresh, inspectable experiments:

1. Delta-format outputs before full-grid outputs.
2. Program synthesis candidates as an ensemble source.
3. Constraint filtering and scoring on every generated attempt.
4. Small eval subsets before GPU-heavy full runs.

Keep generated artifacts in `results/` and heavyweight model assets in `external/`.
