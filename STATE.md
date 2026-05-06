# State — agi2-v2

Live status. Update when something completes or changes status. Sister doc: `PLAN.md` (open questions, next experiments).

## Current investigation

**Question:** Does isolating multi-test ARC puzzles into single-test pseudo-puzzles (each with its own fresh TTT fine-tune) recover any subtests that vanilla grouped NVARC misses?

**Why it might:** vanilla TTT shares one fine-tune across all of a multi-test puzzle's subtests, plus shares the per-puzzle decode cap. Isolation gives each subtest its own fine-tune and its own full budget.

**Why it might not:** if vanilla failures on multi-test puzzles are capability-bound rather than budget-bound, isolation gains nothing.

**Status as of chunks 0+1+2:** Vanilla is **ahead** (vanilla 2/17 across chunks 0+1, separated 3/25 across chunks 0+1+2 with vanilla chunk 2 still running). No "vanilla miss → separated solve" datapoints yet. Tentative read: capability-bound on most chunk 0 puzzles, then chunks 1+2 surfaced wins both arms split fairly. See "Verified findings" below.

## In flight

Nothing — chunks 0+1+2 packaged. Next push is chunk 3 (problems 13–16: `36a08778`, `38007db0`, `3a25b0d8`, `3e6067c3`).

## Verified findings (with dates)

### Pre-r2 (OLD chunking design — r1 superseded)

- **2026-05-06 — `36a08778` solved 2/2 by vanilla NVARC grouped.** Re-scored from `separated172-36a08778` v8. Implication: prior assumption "vanilla fails multi-test ~99%" was wrong on the working 26.94 base.
- **OLD `vanilla-chunked` chunks 0/1**: 1/5 then 0/5. Discarded analysis: 0/8 right answers hidden in candidate list — selection algo NOT the bottleneck on those puzzles. Key quirk: `13e47133[1]` not attempted on chunk 0 (per-puzzle 1200s cap fired) — concrete budget-starvation evidence. Forensics in `results/archive/`.

### r1 design (chunks 0+1, original budget — now superseded)

- **r1 chunk 0 (8 subtests, 4 multi-test originals `13e47133`/`142ca369`/`16de56c4`/`195c6913`):** vanilla 0/8, separated 0/8. Capability-bound across the board. 0 discarded right answers in either run.
- **r1 chunk 1 (9 subtests, 4 originals `1ae2feb7` n=3 / `20270e3b` / `20a9e565` / `21897d95`):** vanilla 2/9, separated 1/9.
  - Both got `1ae2feb7_2` (5c).
  - Vanilla got `20270e3b_1` (6b) via attempt_2; separated missed.
  - **`1ae2feb7_1` had right answer at rank 7 in vanilla (DISCARDED).** First confirmed selection-algo failure — chunk 0 was 0 discarded, chunk 1 had this 1 case. Selection bottleneck IS sometimes real.
  - `21897d95` skipped by both runs (content fast-bail).
  - Separated did NOT auto-attach the mirror dataset on r1; ran on sorokin's mount. Same weights, but a sanity-check failure that motivated r2.

### r2 design (chunk 2, doubled budget — current)

- **r2 chunk 2 separated (8 subtests, originals `221dfab4`/`247ef758`/`269e22fb`/`2d0172a1`):** **2/8 — both subtests of `2d0172a1` (problem 12)** via attempt_1, rank 1 each. 8/8 attempted (no skips — `apply_puzzle_timeout` fix paid off). Mirror loaded correctly. 0/8 discarded. Wall 18.9 min.
- **r2 chunk 2 vanilla:** **2/8 — same `2d0172a1` both subtests, both attempt_1.** 8/8 attempted, 0/8 discarded. Wall 28.2 min. Per-puzzle finishes: `269e22fb` 4.7m, `247ef758` 6.8m, `2d0172a1` 7.3m, **`221dfab4` 22.0m** — single hard puzzle bottlenecked the wall (TTT only 62s; the 21 min was DFS decode, hitting the new 1080s cap).
- **Chunk 2 net:** both arms tied at 2/8. No isolation gain, no regression. Same puzzle solved both times. Per-subtest avg time: vanilla ~14 min, separated ~9.5 min — separated 50% faster on the same content because subtests parallelize across workers instead of serializing inside a single puzzle.

### Cumulative (chunks 0+1+2)

- Vanilla: 4/25 (16%) — `1ae2feb7_2`, `20270e3b_1` (chunk 1); `2d0172a1_0`, `2d0172a1_1` (chunk 2).
- Separated: 3/25 (12%) — `1ae2feb7_2` (chunk 1); `2d0172a1_0`, `2d0172a1_1` (chunk 2).
- Per-puzzle: of the 12 multi-test originals attempted, 3 had at least one subtest solved (`1ae2feb7`, `20270e3b`, `2d0172a1`); the other 9 were all-miss in both arms.
- **No "vanilla miss → separated solve" cases yet.** One regression (`20270e3b_1` vanilla solved, separated missed).
- **Separated is faster per-unit-of-work** but covers the same accuracy. Hybrid time-advantage hypothesis (PLAN.md P5) deferred until more chunks land.

## Active artifacts

### Kaggle kernels (lucyellu/)

| slug | purpose | status |
|---|---|---|
| `subprobs-vanilla120-r2` | Vanilla NVARC (grouped). r2 design: puzzle_timeout 2400s, DFS 1080s, RUN_HOURS 3.0, rerun-mode branch, submission_vanilla120_chunk{N}.json + submission.json. | **active** |
| `subprobs-separated172-r2` | Separated. r2 design: RUN_HOURS 4.0, rerun builds separated set inline, submission_sep172_chunk{N}.json human-keyed + submission.json. | **active** |
| `subprobs-vanilla120-r1` | r1 (1.5h, 1200s puzzle cap). Last v3 = chunk 1, 2/9. | superseded |
| `subprobs-separated172-r1` | r1 (2.0h, 1200s, mirror never auto-attached). v3 = chunk 1, 1/9. | superseded |
| `kaggle-v1-n8-r64-a16-nvarc-32-64-base` | 26.94 LB working base. Don't modify. | reference |
| `vanilla-chunked` / `separated172-all` | OLDEST (pre-r1). | orphaned |

### Kaggle datasets (lucyellu/)

- `arc-agi-2-separated-eval-172` v2 — 172 separated subtests with random hex IDs (seed=2602). Built by `scripts/build_separated_eval.py`.
- `qwen3-4b-grids15-sft139-mirror` — sorokin's 4B weights mirror (~7 GB). **Manual web-UI attach required** for new kernel slugs (metadata `dataset_sources` doesn't auto-attach).

### Local notebooks (active in `notebooks/`)

- `notebooks/subprobs-vanilla120-r2.ipynb` + push folder + slug `lucyellu/subprobs-vanilla120-r2`.
- `notebooks/subprobs-separated172-r2.ipynb` + push folder + slug `lucyellu/subprobs-separated172-r2`.
- `notebooks/kaggle-v1-n8-r64-a16-nvarc-32-64-base.ipynb` — build dependency (read by `_build_*_nb.py`). Don't move.

`notebooks/archive/` and `notebooks/reference/` are gitignored.

### Build / tooling

- `tools/_solver_patches.py` — patches: `apply_path_override`, `apply_model_resolver`, `apply_early_stop`, `apply_puzzle_timeout`. Both build scripts apply all four.
- `tools/_chunking.py` — `compute_chunk_mappings(chunk_size=4, multi_only=True)` → 49 multi-test originals split into 13 chunks of 4.
- `tools/_problem_index.py` — bidirectional `<pid>_<test_idx>` ↔ human index (`1a`, `5c`, `12b`). Baked into both notebooks at build time.
- `tools/_build_subprobs_vanilla120_r2_nb.py` and `tools/_build_subprobs_separated172_r2_nb.py` — produce the active notebooks. After rebuilding, `cp` the active `.ipynb` into the matching `-push/` folder.
- `tools/dry_run_notebook.py` — local exec with subprocess + GPU stubbed; ~5 s. Run before every push.
- `tools/save_kernel_run.py <slug> <duration> <descriptor>` — packages a finished Kaggle run. Naming: `results_subprobs-<descriptor>-<rest>-<duration>` so chunks group when sorted.
- `tools/analyze_discarded_answers.py` — selection-algo diagnostics (rank of correct answer if produced).
- `tools/similarity_analysis.py` — viewer-equivalent metrics with EXACT/CLOSE/ECHO/OFF/BLANK flags. Diff mode for two-submission comparison.
- `tools/merge_submissions.py` — combines per-chunk full-120-shape `submission.json` files into one leaderboard-ready file.
- `tools/_fetch_log3.py` — UTF-8 workaround for Kaggle CLI's cp1252 emoji crash.
- `scripts/build_separated_eval.py` — regenerates the separated dataset (`--seed 2602`).
- `scripts/mirror_qwen_models.py` — download + Kaggle Dataset upload of sorokin's weights.

### Local data

- `data/arc-prize-2026-arc-agi-2/` — official 120 eval + 240 test.
- `data/arc-agi-2-separated-eval-172/` — local source-of-truth for the separated set.
- `data/mirrors/4b_*/` — local copies of qwen3 4B weights (already uploaded; ~14 GB, deletable).

### Saved run results

Active forensics in `results/`:
- `results_subprobs-chunk0-vanilla120-r1-1187s/` — 0/8 (capability-bound, 0 discarded).
- `results_subprobs-chunk0-separated172-r1-1074s/` — 0/8.
- `results_subprobs-chunk1-vanilla120-r1-781s/` — 2/9 (5c, 6b).
- `results_subprobs-chunk1-separated172-r1-713s/` — 1/9 (5c).
- `results_subprobs-chunk2-separated172-r2-1135s/` — 2/8 (12a, 12b).

Older forensics in `results/archive/` (gitignored). Includes pre-r1 `vanilla-chunked-*` runs and `kaggle_multitest_failure_*` snapshots.

## Recurring Kaggle quirks

- `model_sources` / `dataset_sources` in `kernel-metadata.json` may not auto-attach on **new** kernel slugs. Manual web-UI attach. Resolver in `arc_solver.py` covers ~10 known mount paths.
- `machine_shape: NvidiaL4x4` is sometimes silently downgraded to single L4. Check via `kaggle kernels pull -m`.
- Unsloth import in a second subprocess crashes with stale `unsloth_compiled_cache/` and `__pycache__/`. Clean between phases.
- Kaggle CLI on Windows fails to write log files containing the unsloth 🦥 emoji (cp1252). Use `tools/save_kernel_run.py` or `tools/_fetch_log3.py`.
- `kaggle kernels output` only fetches the LATEST version's session output. For older versions use `kaggle kernels files` or `tools/_fetch_kernel_version.py`.
- Kaggle Models API upload needs `licenseName`; Dataset path doesn't.

## Open mysteries (no impact on workflow yet)

- `[EarlyStopOnLowLoss]` diagnostic prints never appear in v3/r2 logs despite the class being defined and `callbacks=[...]` wired into `UnslothFixedTrainer`. Training losses (0.019–0.044) sit well below the 0.05 threshold yet step counts hit the full schedule (32–64). Either Unsloth filters callback prints, on_log isn't called by `UnslothFixedTrainer`, or something else. Doesn't block analysis but tuning the threshold blind is unwise until understood.

## Standing workflow

1. **Edit notebook** via `tools/_build_*_nb.py` (don't hand-edit JSON).
2. **Local dry-run**: `python tools/dry_run_notebook.py notebooks/<name>.ipynb`.
3. **Sync push folder**: `cp notebooks/<name>.ipynb notebooks/<name>-push/<name>.ipynb`.
4. **Push**: `kaggle kernels push -p notebooks/<name>-push`.
5. **Manual attach in Kaggle web UI**: competition + datasets + model + unsloth-patch kernel (one-time per slug).
6. **Run on Kaggle** (Save Version → Save & Run All).
7. **On completion**: `tools/save_kernel_run.py`, then `analyze_discarded_answers.py` (always, per memory), then `similarity_analysis.py`.

## Session pickup checklist

1. Read this `STATE.md` and `PLAN.md`.
2. `kaggle kernels status lucyellu/subprobs-vanilla120-r2` and `subprobs-separated172-r2`.
3. If runs complete: package, score, run analyzers, append to "Verified findings".
4. Memory entries (auto-loaded via `MEMORY.md`) cover Kaggle gotchas — don't relearn them.
