# ARC-AGI-2 (ARC Prize 2026)

My workspace for the [ARC Prize 2026](https://arcprize.org/competitions/2026/arc-agi-2) Kaggle
competition: abstract-reasoning puzzles where a model sees a few input/output grid pairs and must
infer the rule.

**Approach:** Qwen3-4B with test-time training (TTT), building on NVARC, the 2025 winning approach.
Best public leaderboard score: **28.19%**, above NVARC's 27.64%.

**Current experiment:** does splitting multi-test puzzles into single-test puzzles, each with its
own fresh fine-tune, recover subtests the grouped approach misses? So far (25 subtests): grouped
4/25, separated 3/25. No case yet where splitting solves something grouped missed.

**Findings along the way:**
- The inherited scorer undercounted multi-test puzzles. It now scores all 172 eval subtests.
- Answer selection is sometimes the bottleneck: on `1ae2feb7_1` the right answer was generated but
  ranked 7th and discarded.

Live status: [`STATE.md`](STATE.md). Open questions and next experiments: [`PLAN.md`](PLAN.md).
Viewer: [arc-agi2-viewer](https://github.com/lucyellu/arc-agi2-viewer) ([live](https://arc-agi2-viewer.netlify.app/)).

## Workspace

Fresh ARC-AGI-2 workspace distilled from `agi-2`.

This folder keeps the useful spine of the old project:

- ARC-AGI-2 2026 train/eval/test JSON data
- local scoring, visualization, and viewer tools
- Qwen prompt/inference runner
- TRM runner scaffold with explicit external paths
- NVARC reference notebooks that were known useful
- audit notes on what to keep, avoid, and try next

It deliberately leaves out cached Unsloth files, old run directories, AGI-3 wheels, broken feedback-loop notebooks, and placeholder submissions.

## Quick Start

From the repo root:

```powershell
python tests/test_smoke.py
python scripts/make_copy_baseline.py
python scripts/score_submissions.py results/submissions/latest/copy_eval_submission.json
```

Run Qwen inference:

```powershell
cd scripts
python infer_qwen.py `
  --challenges ../data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json `
  --output ../results/submissions/latest/qwen_submission.json `
  --attempts 3 `
  --quant auto
```

Submission viewer lives in the separate `lucyellu/arc-agi2-viewer` repo.

## External Assets

Large model/code assets are intentionally not copied. Put them under `external/`, or set env vars:

- `AGI2_TRM_DIR`: TinyRecursiveModels checkout
- `AGI2_TRM_CKPT_V2`: TRM v2 checkpoint
- `AGI2_TRM_CKPT_V1`: TRM v1 checkpoint

Suggested defaults:

```text
external/TinyRecursiveModels/
external/models/trm_arc_prize_verification/arc_v2_public/step_723914/
external/models/architects_lora/
```

## What To Build Next

The cleanest next attack is not another feedback loop. Start with:

1. Delta-output experiment: ask the model for changed cells only, then reconstruct.
2. Program synthesis ensemble: generate Python transforms that fit train pairs.
3. Only after that, revisit new LoRA training or stitched-grid synthetic data.

See `docs/AUDIT.md` for the project audit and `docs/history/state_2026-05-01.md` for inherited run history.
