# Plan — agi2-v2

Open questions and next experiments. Sister doc: `STATE.md` (current status, artifacts, recent findings).

## Open questions

1. **Does isolation help on multi-test puzzles vanilla actually fails?** Need full grouped + separated data on the 49 multi-test originals. Diff: any subtest where vanilla fails AND separated solves is real signal.

2. **Two-cell contingency to fill in across 101 multi-test subtests:**
   - vanilla solved + isolation solved → no signal (both fine)
   - vanilla solved + isolation missed → isolation *hurts* on this one (regression)
   - vanilla missed + isolation solved → **the result we're hunting**
   - vanilla missed + isolation missed → capability-bound, isolation doesn't fix it

3. **At what isolation rate is the experiment worth pursuing further?** If isolation recovers ~5+ subtests across 101, worth a leaderboard run. <2 is noise, abandon.

4. **(User hypothesis 2026-05-06)** Is the inference pipeline implicitly budgeting for 120 puzzles when handed 172? Quick code audit: search for any size-dependent assumption in `scripts/arc_solver.py` and `scripts/arc_loader.py`. Likely null result — TTT step count (`n=8`) and per-puzzle 1200 s decode cap are universal — but cheap to confirm.

5. **Why did `136b0064` finish in 66 s with no non-trivial attempt?** Reproduces in BOTH grouped and separated runs (under OLD chunking). Worker reported "finished" but no decode output. Either decode produced nothing valid (max_score threshold) or DFS branches all died. Worth pulling the bz2 inference output for that subkey. Not in active multi-only chunking, so deprioritized.

6. **Does loss-thresholded TTT preserve scores?** Reference notebook implies yes (29.72 LB). Now wired in via `tools/_solver_patches.apply_early_stop`. First chunk 0 results will tell us if early-stop fires AND if scores look reasonable vs the OLD design without early-stop.

## Next experiments (priority order)

### P1 — DONE (2026-05-06) — old-design chunks 0+1 ✓

OLD chunking design (4 separated keys/chunk; multi-test originals duplicated across vanilla chunks). Results in `results/results_vanilla-chunked-*`. Findings recorded in `STATE.md`.

### P1.5 — DONE (2026-05-06) — Refactor: chunk-on-originals + multi-only + random IDs ✓

- **Chunk on ORIGINAL puzzles** (4 per chunk) — vanilla and separated kernels share the same chunk → originals mapping. No more redundant grouped runs.
- **Multi-only filter:** 49 multi-test originals → 13 chunks of 4. Single-test puzzles excluded (no isolation signal).
- **Random hex IDs** in the separated dataset (seed=2602). Mimics the 120-eval ID format. Same input/output pairs preserved.
- **EarlyStopOnLowLoss callback** (threshold=0.05, patience=2, logging_steps=16). Ported from the 29.72-LB reference notebook into `tools/_solver_patches.apply_early_stop`.
- **Kernel slug versioning:** `r1` was the original chunked design (chunks 0+1). `r2` is the new design with doubled budget, rerun-mode branching, submission renames, and human-id reports — starting at chunk 2. Bump to `rN+1` when making another design overhaul.

### P2 — IN FLIGHT (2026-05-06) — chunk 0 of both kernels

Both kernels v1 pushed and running. Chunk 0 originals: `13e47133`, `142ca369`, `16de56c4`, `195c6913` (all multi-test n=2).

Pickup commands and sanity checks: see `STATE.md` "Pending Kaggle runs" section.

### P3 — Walk chunks 1..12 (after P2 confirms design works)

12 chunks remaining per kernel. ~30 min each. Bump `CHUNK_INDEX` in cell 0 default per push, OR override on Kaggle's web editor before Run.

**Action per chunk:**
1. Edit cell 0 in build script (or on Kaggle UI): `CHUNK_INDEX = int(os.getenv("AGI2_CHUNK_INDEX", "<N>"))`.
2. `python tools/dry_run_notebook.py notebooks/<name>.ipynb`.
3. `kaggle kernels push -p notebooks/<name>-push` (both kernels).
4. Run All on each.
5. After completion: `tools/save_kernel_run.py <slug> <Xs> chunk<N>`.

Total wall clock to finish: 12 chunks × ~30 min × 2 kernels = ~12 h sequential, or ~6 h if both kernels run in parallel.

After P3, we have a complete grouped-vs-separated comparison on all 49 multi-test originals (101 subtests).

### P4 — Quick code audit (open question 4)

Grep `scripts/arc_solver.py` and `scripts/arc_loader.py` for any size-dependent constants. Cheap, ~15 min.

### P5 — Fork the unsloth patch kernel

One-click "Copy & Edit" of `sorokin/pip-install-unsloth-flash-patch` on Kaggle web UI, save under `lucyellu/`. Update kernel-metadata `kernel_sources`. ~30 s. Insurance against sorokin breaking the patch.

### P6 — Mirror the 2B model (optional)

`python scripts/mirror_qwen_models.py --only-2b --skip-upload` to download. Then write a `dataset-metadata.json` and `kaggle datasets create -p ...` to upload as a Dataset. Only if you want to test the 2B variant.

### P7 (deferred until P3 confirms isolation helps) — submission notebook for hidden test set

Build `separated_submission.ipynb` that:
- Detects `KAGGLE_IS_COMPETITION_RERUN`.
- In rerun mode, builds the separated set in-notebook from `arc-agi_test_challenges.json` (240 puzzles → ~340 separated subtests, no solutions needed).
- Runs all separated, recombines to original-shape `submission.json` keyed by test puzzle ids.

Don't build until P3 confirms isolation actually helps.

### P8 (longer-term) — train our own grids model

**Goal:** own the model, not depend on sorokin. Probably qwen3-4b base + grid-format SFT on ARC train set + augmentation.

**Open subquestions:** dataset format (we have `scripts/arc_loader.py` formatting), training infra (Kaggle TPU? Lambda Labs? local 4090?), checkpoint sharing path.

## Decisions made

- **Mirror as Kaggle Dataset, not Model.** Model API upload kept hitting `licenseName` validation; Dataset path worked first try.
- **Chunk size 4 ORIGINALS** (not separated keys) for both kernels. Each chunk runs 4-12 separated subtests depending on multi-test composition.
- **Multi-only chunking.** 49 multi-test originals = 13 chunks; single-test puzzles excluded.
- **Random hex IDs** in the separated dataset (seed=2602). Mimics the 120-eval ID format. Same input/output pairs preserved.
- **`vanilla120-r1` is self-contained** — chunk → originals mapping baked in at build time, no separated-dataset attachment needed.
- **Mirror is preferred over sorokin's mount** — set explicitly via `AGI2_MODEL_PATH` in the run cell.
- **EarlyStopOnLowLoss is in the build pipeline** (threshold=0.05, patience=2, logging_steps=16).
- **Don't mirror competition data.** Comes from `competition_sources`, durable for active competition.
- **Naming convention for new experiment kernels:** `subprobs_<basetype><N>_r<runidx>`. Lets us run multiple experimental variants without slug collisions.

## Notes on the broader landscape

Few public ARC-AGI grid-tuned models exist on Kaggle Models. NVARC's qwen3 grid SFT line (sorokin's `qwen3_4b_grids15_sft139`, `qwen3_2b_grids15_sft141`) is one of the only public artifacts; the other 4B variant in that line was already deleted. Top performers historically keep weights private. TinyRecursiveModels has open checkpoints but on HuggingFace, not Kaggle, and uses a different architecture entirely. So sorokin's mirrors are essentially the only public NVARC-style weights — high concentration risk, hence we own a copy now.
