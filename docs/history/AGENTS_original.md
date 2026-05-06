# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

An ARC-AGI-2 (2026) competition workspace. It adapts the NVARC team's winning 2025 solution — which combines a fine-tuned Qwen3-4B (ARChitects) with a Tiny Recursive Model (TRM) — to the 2026 evaluation set.

## Running things

**Score a submission against the eval set (120 puzzles):**
```bash
cd scripts
python score_submissions.py                          # auto-detect all local submissions
python score_submissions.py path/to/sub1.json ...   # explicit files
```

**Run Qwen3 inference (ARChitects model):**
```bash
cd scripts
python infer_qwen.py \
  --challenges ../data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json \
  --output ../results/submissions/latest/qwen_submission.json \
  [--lora ../nvarc/models/architects_lora] \
  [--ids 00576224 007bbfb7] \
  [--quant auto|4bit|8bit|none] \
  [--attempts 2]
```

**Run TRM inference (test-time fine-tuning):**
```bash
cd scripts
python infer_trm.py \
  --challenges ../data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json \
  --output ../results/submission.json \
  [--num-aug 8 --epochs 200]   # quick
  # competition: --num-aug 128 --epochs 4000
```

**Launch the results viewer:**
```
START_VIEWER.bat          # Windows: starts viewer_server.py and opens Chrome
python viewer_server.py   # then open http://localhost:8080/run_viewer.html
```

## Architecture

```
scripts/          Core pipeline scripts
  arc_loader.py       ArcDataset + QwenFormatter — loads puzzles, applies augmentations,
                      formats grids as ChatML for Qwen3
  arc_solver.py       Qwen3 LoRA fine-tuning via Unsloth (UnslothFixedTrainer patches
  arc_solver_rev3.py  a view-tensor bug in Unsloth's DDP gradient scaling)
  infer_qwen.py       Qwen3 inference — auto-detects VRAM and selects 4-bit/8-bit/full precision
  infer_trm.py        TRM test-time fine-tuning runner (spawns the TRM eval script)
  arc_decoder.py      ArcDecoder — loads bz2+pickle inference outputs, deduplicates guesses
                      by solution hash, applies selection algorithms to pick best 2 attempts
  constraint_filter.py  Pre-filter: derives color-delta constraints from training examples
                        (zero/increase/decrease/inconsistent per color), eliminates attempts
                        that violate them before submission
  complexity_analyzer.py  Puzzle difficulty scoring (grid area, colors, size-change penalty)
  score_submissions.py    Local eval scorer with per-difficulty-level breakdown
  arc_counter.py      Produces color-count matrices for debugging constraint filtering
  viz_arc.py          Grid visualization utilities

nvarc/            NVARC 2025 solution code (upstream reference)
  models/             Pre-downloaded checkpoints: ARChitects LoRA, TRM (arc_v2_public step_723914)
  ARChitects/         Fine-tuning config for Qwen3-4B
  TRM/                Tiny Recursive Model scripts
  SDG/                Synthetic data generation pipeline

data/
  arc-prize-2026-arc-agi-2/   2026 evaluation challenges + solutions (120 puzzles)
  arc-prize-2026-arc-agi-3/   Additional puzzle data

results/
  submissions/latest/   Most recent inference outputs (picked up by viewer and scorer)
  results_*s/           Timestamped bz2+pickle output directories from past runs
```

## Key design patterns

**Grid format:** 2D lists of ints 0–9. Serialized for LLMs as plain ASCII rows with no separator, e.g. `0123\n4567`. `QwenFormatter` wraps these in ChatML: `<|im_start|>user\n{grid}<|im_end|><|im_start|>assistant\n{grid}<|im_end|>`.

**Augmentation pipeline** (`ArcDataset.augment`): transpose + 3 rotations + random color permutations. `n=8` for quick runs, `n=128` for competition.

**Inference output format:** bz2+pickle files per puzzle, each containing a list of dicts with `solution` (numpy array), `beam_score`, and `score_aug`. `ArcDecoder` groups by solution hash and applies two selection algorithms:
- `score_full_probmul_3`: weighted by beam score + augmentation score (primary)
- `score_kgmon`: count-based voting (secondary)

**Constraint filtering** (`constraint_filter.py`): runs before selecting attempts. Analyzes color cell-count deltas across training examples to find hard constraints, then rejects attempts that violate them. Applied in both `infer_qwen.py` and `infer_trm.py`.

**Eval vs test split:** `score_submissions.py` only scores the 120-puzzle eval set. Kaggle's leaderboard uses a separate 240-puzzle test set.
