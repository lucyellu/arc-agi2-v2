# State — agi2-v2

Live status. Update when something completes or changes status. Sister doc: `PLAN.md` (open questions, next experiments).

## Current investigation

**Question:** Does isolating multi-test ARC puzzles into single-test pseudo-puzzles (each with its own fresh TTT fine-tune) recover any subtests that vanilla grouped NVARC misses?

**Why it might:** vanilla TTT shares one fine-tune across all of a multi-test puzzle's subtests, plus shares the per-puzzle 1200 s decode cap. Isolation gives each subtest its own fine-tune and its own full budget.

**Why it might not:** if vanilla failures on multi-test puzzles are capability-bound rather than budget-bound, isolation gains nothing. Discarded-answer analysis already confirmed the decoder selection algo is *not* the bottleneck — when we miss, the model never produced the right grid.

## Pending Kaggle runs (in flight as of 2026-05-06 PM)

Both running on v1 of new slugs. Same chunk 0 content (4 multi-test originals: `13e47133`, `142ca369`, `16de56c4`, `195c6913`). Vanilla produces 8 subtest results (grouped processes whole puzzle); separated produces 8 separated-puzzle results (one per subtest).

| Kernel | What it tests | Expected wall clock |
|---|---|---|
| `lucyellu/subprobs-vanilla120-r1` | Vanilla grouped baseline on those 4 multi originals | ~16–22 min |
| `lucyellu/subprobs-separated172-r1` | Separated isolation on those 4 originals' 8 subtests | ~25–35 min |

### Pickup commands when they complete

```powershell
PYTHONIOENCODING=utf-8 kaggle kernels status lucyellu/subprobs-vanilla120-r1
PYTHONIOENCODING=utf-8 kaggle kernels status lucyellu/subprobs-separated172-r1

# Package each completed run:
python tools/save_kernel_run.py lucyellu/subprobs-vanilla120-r1 <Xs> chunk0
python tools/save_kernel_run.py lucyellu/subprobs-separated172-r1 <Xs> chunk0
```

### Sanity checks to run after results land

1. **Mirror model loaded:** log should print `Using mirror model: /kaggle/input/datasets/lucyellu/qwen3-4b-grids15-sft139-mirror` AND worker subprocesses should print `[model] using AGI2_MODEL_PATH override: ...`.
2. **EarlyStopOnLowLoss firing:** if any puzzle is "easy", expect `[Early stop] step N loss=...` lines in the worker output.
3. **Vanilla determinism cross-check:** vanilla chunk 0 should produce the same `13e47133`, `142ca369` results as the OLD `vanilla-chunked` chunk 0/1 runs (saved in `results/results_vanilla-chunked-1415.9s-chunk0-all/` and `results/results_vanilla-chunked-1297.97s-chunk1-mirror/`). Differences would indicate the new chunking or early-stop changed inference outcome — worth investigating.
4. **Pairing analysis (the actual experiment):** for each of the 8 multi-test subtests in chunk 0, compare vanilla grouped result vs separated result. Look for:
   - Both wrong → capability-bound
   - Vanilla wrong, separated correct → **isolation helped** (the result we're hunting)
   - Vanilla correct, separated wrong → isolation hurt (regression)
   - Both correct → no signal (vanilla solved it already)

## Active artifacts

### Kaggle kernels (lucyellu/)

| slug | purpose | status |
|---|---|---|
| `subprobs-vanilla120-r1` | Vanilla NVARC (grouped) on 120-eval, multi-only (49 originals, 13 chunks of 4). EarlyStopOnLowLoss TTT, chunk-on-originals, mirror model. | v1 RUNNING chunk 0 |
| `subprobs-separated172-r1` | Vanilla NVARC on separated 172, multi-only (101 separated subtests of 49 multi originals, 13 chunks). | v1 RUNNING chunk 0 |
| `kaggle-v1-n8-r64-a16-nvarc-32-64-base` | Working 26.94 LB reference. Don't modify. | reference |
| `vanilla-chunked` | OLD slug, superseded by `subprobs-vanilla120-r1`. Old chunk 0+1 results still attached. Safe to delete via web UI. | orphaned |
| `separated172-all` | OLD slug, superseded by `subprobs-separated172-r1`. Old chunk 0 results still attached. Safe to delete. | orphaned |

### Kaggle datasets (lucyellu/)

- `arc-agi-2-separated-eval-172` v2 — 172-puzzle separated eval set with **random hex IDs** (seed=2602) mimicking the 120-eval ID format. Same input/output pairs as the original puzzles. Built by `scripts/build_separated_eval.py`.
- `qwen3-4b-grids15-sft139-mirror` — mirror of sorokin's 4B weights (~7 GB), as a Dataset. Resolver in `arc_solver.py` finds it automatically once attached.

### Local notebooks (active in `notebooks/`)

- `notebooks/subprobs-separated172-r1.ipynb` — chunked separated runner. Push folder: `notebooks/subprobs-separated172-r1-push/`. Kaggle slug: `lucyellu/subprobs-separated172-r1`.
- `notebooks/subprobs-vanilla120-r1.ipynb` — chunked vanilla baseline. Push folder: `notebooks/subprobs-vanilla120-r1-push/`. Kaggle slug: `lucyellu/subprobs-vanilla120-r1`.
- `notebooks/kaggle-v1-n8-r64-a16-nvarc-32-64-base.ipynb` — 26.94 LB working base. **Build dependency** — `_build_*_nb.py` reads `arc_loader.py` / `arc_decoder.py` / `arc_solver.py` cells from this. Don't move.

Everything else is in `notebooks/archive/` (gitignored) — see `notebooks/archive/README.md`.

### Build / tooling

- `tools/_solver_patches.py` — shared arc_solver patches: `apply_path_override`, `apply_model_resolver`, `apply_early_stop`. Both build scripts import from here.
- `tools/_chunking.py` — shared `compute_chunk_mappings(chunk_size=4, multi_only=True)`. Both build scripts use it.
- `tools/_build_subprobs_separated172_r1_nb.py` and `tools/_build_subprobs_vanilla120_r1_nb.py` — produce the active notebooks. **Always edit these, never hand-edit the .ipynb JSON.** After rebuilding, also `cp` the active `.ipynb` into the matching `-push/` folder so `kaggle kernels push` ships the latest content.
- `tools/dry_run_notebook.py` — local exec of any active notebook with subprocess + GPU phases stubbed; ~5 s. **Run before every push.**
- `tools/save_kernel_run.py <owner>/<slug> <duration> <descriptor>` — packages a finished Kaggle run into `results/results_<slug>-<duration>-<descriptor>/`. Handles cp1252 emoji bug, pulls notebook source + log + outputs.
- `tools/_fetch_log3.py` — UTF-8 workaround for the Kaggle CLI cp1252 bug on the unsloth 🦥 emoji.
- `tools/_dump_kaggle_log.py`, `tools/_dump_cell.py`, `tools/_inspect_nb.py` — diagnostic helpers.
- `tools/analyze_discarded_answers.py <results_dir>` — checks if any candidate solutions matched the answer but were ranked below #2.
- `scripts/build_separated_eval.py` — regenerates the separated dataset. Use `--seed 2602` for reproducibility.
- `scripts/mirror_qwen_models.py` — download sorokin's models + upload as Kaggle Models or Datasets. Note: Model API path needs `licenseName` field; Dataset path is what we used for the 4B mirror.

### Local data

- `data/arc-prize-2026-arc-agi-2/` — official 120 eval + 240 test data.
- `data/arc-agi-2-separated-eval-172/` — local source-of-truth for the separated set (latest: random IDs, seed=2602).
- `data/mirrors/4b_*/` — local copies of the qwen3 4B weights (already uploaded; can delete to free ~14 GB).

### Saved run results

- `results/results_vanilla-chunked-1415.9s-chunk0-all/` — vanilla chunk 0 (sorokin), 1/5 correct. **OLD chunking design.** Originals: `0934a4d8`, `135a2760`, `136b0064`, `13e47133`. Under new multi-only design, only `13e47133` is multi (the other 3 are excluded). Discarded-answer analysis included.
- `results/results_vanilla-chunked-1297.97s-chunk1-mirror/` — vanilla chunk 1 (mirror), 0/5 correct. **OLD chunking design.** Originals: `13e47133` (redundant), `142ca369`, `16b78196`. Under new design, `13e47133` and `142ca369` are multi-only (chunk 0 of new design). Discarded-answer analysis included.
- Both folders have `vanilla-chunked.log`, the notebook source, full `grouped_outputs/` bz2 dumps, and a `discarded_answers_analysis.{json,md}`.

## Verified findings (with dates)

- **2026-05-06 — `36a08778` is solved 2/2 by vanilla NVARC grouped.** Verified by re-scoring `submission_grouped.json` from `separated172-36a08778` v8 against `arc-agi_evaluation_solutions.json`. Implication: my prior assumption "vanilla fails multi-test ~99% of the time" was wrong; the v18-era 1/101 stat from state notes does not apply to the working 26.94 base.

- **2026-05-06 — separated172-all chunk 0 (OLD design): 1/4 correct.** Pipeline working end-to-end. `135a2760_0` correct; `0934a4d8_0` and `13e47133_0` wrong; `136b0064_0` produced no real attempt (worker fast-bailed in 66 s — content quirk, reproduces in grouped too). End-to-end pipeline confirmed.

- **2026-05-06 — vanilla-chunked chunk 0 grouped (OLD design): 1/5 correct.**
  - Single-test puzzles (`0934a4d8`, `135a2760`, `136b0064`) match separated chunk 0 outcome-for-outcome (sanity check passed; grouped == separated for n=1).
  - `13e47133[0]` wrong (matches separated).
  - **`13e47133[1]` was NOT attempted — budget starvation.** Worker hit the per-puzzle 1200 s decode cap before getting to test[1]. Concrete evidence the grouped budget asymmetry is real.

- **2026-05-06 — vanilla-chunked chunk 1 grouped (OLD design, mirror model): 0/5 correct.**
  - Mirror model verified working (`[model] using AGI2_MODEL_PATH override: ...` logged).
  - With smaller queue (3 puzzles vs 4 in chunk 0), `13e47133[1]` *did* get attempted — but still wrong. So `13e47133[1]` isn't purely budget-bound.
  - `142ca369` (n=2) and `16b78196` (n=1) all wrong.

- **2026-05-06 — discarded-answer analysis on 8 subtests with inference output:**
  - 1/8 correct in top-2 (just `135a2760`).
  - **0/8 had the right answer hidden in the candidate list.** Decoder selection algo NOT the bottleneck.
  - Candidate-count breadth varies wildly (1–22 unique grids). Some failures under-explored, some well-explored without finding answer. Both budget AND capability are at play.

## Recurring Kaggle quirks (gotchas)

- `model_sources` declared in `kernel-metadata.json` may not auto-attach on **new** kernel slugs. Workaround: `_resolve_model_path()` in `arc_solver.py` tries 10 known mount paths + globs for the model's `config.json`.
- `dataset_sources` likewise can land at unexpected paths (e.g. `/kaggle/input/datasets/<owner>/<slug>/...`). Glob fallback in cell-4 finder handles it.
- `machine_shape: NvidiaL4x4` is sometimes silently downgraded to single L4. Check via `kaggle kernels pull -m`.
- Unsloth import in a second subprocess crashes if the first run's `unsloth_compiled_cache/` and `__pycache__/` are still present. Clean them between phases.
- Kaggle CLI on Windows fails to write log files containing the unsloth 🦥 emoji (cp1252 codec). Use `tools/save_kernel_run.py` or `tools/_fetch_log3.py`.
- `kaggle kernels output` only fetches the LATEST version's session output. For older versions, use `kaggle kernels files` to confirm files exist, download via web UI, or use `tools/_fetch_kernel_version.py`.
- Kaggle Models API upload needs `licenseName` field; Dataset upload doesn't. Use Datasets for our mirrors.

## Standing workflow

1. **Edit notebook** via `tools/_build_*_nb.py` (don't hand-edit JSON).
2. **Local dry-run**: `python tools/dry_run_notebook.py notebooks/<name>.ipynb` — must pass before push.
3. **Push**: `kaggle kernels push -p notebooks/<name>-push`.
4. **Run on Kaggle** (web UI Save Version → Save & Run All; auto-runs after push).
5. **On completion, save run**: `python tools/save_kernel_run.py <owner>/<slug> <Xs> <descriptor>`.
6. **Score / inspect**: read the report JSON; verify against `data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json`. Optionally `tools/analyze_discarded_answers.py <results_dir>` for selection-algo diagnostics.

## Session pickup checklist

For a fresh context, run in order:
1. Read this `STATE.md` and `PLAN.md`.
2. `kaggle kernels status lucyellu/subprobs-vanilla120-r1` and `lucyellu/subprobs-separated172-r1`. If COMPLETE, run the pickup commands above.
3. Score results, run sanity checks listed above.
4. Update `STATE.md`'s "Verified findings" with the new datapoints (date them).
5. If chunk 0 results look promising, walk chunks 1..12 per `PLAN.md` P3.
6. Memory entries (auto-loaded via `MEMORY.md` index) cover Kaggle gotchas — don't relearn them.
