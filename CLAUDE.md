# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active investigation — read first

For current work in flight, read `STATE.md` and `PLAN.md` in the repo root before doing anything else. Those track the active isolation experiment (multi-test grouped-vs-separated NVARC), the running Kaggle kernels, recent findings, and the priority-ordered next steps. Memory entries (auto-loaded via `MEMORY.md`) cover recurring Kaggle gotchas (model attachment quirks, unsloth re-import bug, log-fetch UTF-8 workaround) — don't relearn them.

The architecture documentation below is from the initial workspace setup and remains accurate for the long-running parts (data shape, scoring, libraries). When in doubt about what's *currently active*, defer to `STATE.md`.

## Workspace Purpose

`agi2-v2` is a clean ARC-AGI-2 (2026) experimentation workspace, distilled from the older `agi-2` project. The repo intentionally drops historical run dumps, broken feedback-loop notebooks, AGI-3 wheels, and placeholder submissions; what remains is the reusable spine. See `docs/AUDIT.md` for what was kept vs. left behind, and `README.md` / `AGENTS.md` for the high-level pitch.

## Common Commands (PowerShell, run from repo root)

```powershell
# Smoke test — verifies eval data shape (120 puzzles / 172 subtests), constraint filter, scorer
python tests/test_smoke.py

# Build the trivial copy-input baseline submission
python scripts/make_copy_baseline.py

# Score one or more submissions against the local eval set
python scripts/score_submissions.py results/submissions/latest/copy_eval_submission.json
python scripts/score_submissions.py        # auto-discovers under results/

# Run Qwen3 prompt-based inference (auto-detects GPU tier, falls back to CPU)
cd scripts
python infer_qwen.py `
  --challenges ../data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json `
  --output ../results/submissions/latest/qwen_submission.json `
  --attempts 3 --quant auto

# Local submission viewer — separate repo: lucyellu/arc-agi2-viewer
```

Single-test invocation (the smoke tests use `unittest`):

```powershell
python -m unittest tests.test_smoke.SmokeTest.test_eval_data_loads
```

Optional dependency groups are declared in `pyproject.toml`:

- `pip install -e .[qwen]` → torch, transformers, peft, bitsandbytes
- `pip install -e .[trm]`  → torch, hydra-core, omegaconf, pydantic, tqdm, coolname, adam-atan2-pytorch
- `pip install -e .[dev]`  → pytest

## Architecture

### Data flow

ARC-AGI-2 challenges/solutions JSON lives under `data/arc-prize-2026-arc-agi-2/`. The eval set is **120 puzzles / 172 subtests** (multi-test puzzles count multiple subtests). Kaggle's leaderboard test set (240 puzzles) is *not* in the repo — every local score number is the eval-set score. `score_submissions.py` is fixed to count all subtests, not just the first per puzzle, so its numbers will not match the original `agi-2` scorer.

A submission is `{ puzzle_id: [ {"attempt_1": grid, "attempt_2": grid}, ... per test ] }`. `_attempt_pair()` in `score_submissions.py` also accepts the list shape `[grid1, grid2]`. A subtest is correct if either attempt matches the solution.

### Inference pipelines

Two independent runners write submissions in the same shape:

- **`scripts/infer_qwen.py`** — prompt-based Qwen3 runner. Auto-tiers quantization by VRAM (`<6GB→4bit`, `6–14GB→8bit`, `14GB+→bf16/fp16`), supports optional `--lora`, and reranks attempts through `constraint_filter`.
- **`scripts/infer_trm.py`** — TinyRecursiveModel runner. Generates a self-contained eval script and shells out to it. **Requires external assets** — `external/TinyRecursiveModels/` and the `arc_v2_public` / `arc_v1_public` checkpoints — overridable via `AGI2_TRM_DIR`, `AGI2_TRM_CKPT_V2`, `AGI2_TRM_CKPT_V1`. Without those, the runner can't execute.

`scripts/arc_solver.py` and `arc_solver_rev3.py` are heavyweight Unsloth-based test-time-training solvers carried over for reference; they pull `unsloth`, `peft`, and large model state and are not on the smoke-test path.

### Reusable scoring/heuristic library

The compact reusable code lives in `scripts/`:

- **`arc_loader.py`** — `ArcDataset`, `QwenFormatter` (ChatML with optional `COLOR_LEGEND`), augmentation/permutation utilities.
- **`constraint_filter.py`** — cheap heuristic reranker. Counts color deltas across train pairs; if the direction is consistent, attempts violating it are penalized. `rank_attempts()` is the entry point — used inside both inference runners and the smoke test. Treat this as a universal reranker for any new method.
- **`arc_decoder.py`** — bz2/pickle attempt aggregation, dedupe, and beam/aug score combinators (`score_full_probmul_3`, `getter_kgmon`).
- **`score_submissions.py`** — multi-submission scorer with rank-based decile complexity levels (L1–L10) and side-by-side comparison. `calc_complexity()` is the formula; `assign_levels()` produces the deciles.
- **`complexity_analyzer.py`**, **`arc_counter.py`**, **`analyze_run.py`** — analysis helpers for runs and the puzzle pool.

### Viewer

The submission viewer (HTTP server + HTML pages) lives in a separate repo: `lucyellu/arc-agi2-viewer`. Point it at this repo's `results/submissions/latest/` directory.

### Results layout

```
results/
  submissions/latest/   # active submissions; viewer auto-loads from here
  inference_outputs/    # bz2/pickle attempt dumps (decoder consumes these)
  worker_markers/       # multi-worker coordination
  results_*/            # historical run folders (auto-discovered by scorer)
```

`results/kaggle_multitest_failure_*` are frozen snapshots from the inherited multitest budget bug — read-only forensics, don't build on them.

### External assets and history

- `external/` — heavy assets (TRM code, checkpoints, LoRAs). Gitignored. See `external/README.md`.
- `notebooks/reference/` — `nvarc-2025-bf16-faster-ttt-via-early-stopping.ipynb` (29.72 LB; we already ported the `EarlyStopOnLowLoss` callback from it into `tools/_solver_patches.py`). Reference only — don't extend in place. `v13-multitask-fast.ipynb` was moved to `notebooks/archive/` after recurring error reports.
- `notebooks/archive/` — failed/superseded notebooks. See `notebooks/archive/README.md`. Read-only.
- `docs/history/state_2026-05-01.md` — inherited session notes; useful for context, but date-stamped. Cross-check before acting on anything in there.

## Project-specific guidance

- The v19–v22 feedback-loop / warmup notebook line is recorded as broken in the audit. Don't extend it unless explicitly asked.
- The recommended fresh attack order from `docs/AUDIT.md`: (1) delta-format outputs, (2) program-synthesis ensemble, (3) constraint filtering applied to every method, (4) small eval subsets before any GPU-heavy run.
- Generated artifacts go under `results/`; heavyweight model assets under `external/`. The `.gitignore` already excludes `*.bz2`, `*.pkl`, `*.pt`, `*.safetensors`, `external/models/`, `external/TinyRecursiveModels/`, and `results/**/*.json` (with carve-outs for READMEs and `.gitkeep`).
- This is a Windows / PowerShell environment — examples in `README.md` use backtick line continuations, not backslashes.
