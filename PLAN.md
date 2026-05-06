# Plan — agi2-v2

Open questions and next experiments. Sister doc: `STATE.md` (current status, artifacts, recent findings).

## Open questions

1. **Does isolation recover any subtest vanilla misses?** Across chunks 0+1+2 (25 separated subtests, 17 vanilla subtests since vanilla chunk 2 just finished too): zero "vanilla miss → separated solve" datapoints. One regression (`20270e3b_1` vanilla solved, separated missed). Tentative read: **isolation does not increase coverage**. Will revisit after walking remaining chunks 3–12.

2. **Is the selection algorithm sometimes the bottleneck?** Yes — confirmed on `1ae2feb7_1` (chunk 1 vanilla) where the right answer was at rank 7. Chunks 0 and 2 had 0/16 discarded, so the rate is puzzle-dependent. Worth a separate experiment: bump `n_guesses=2` to 3 or 5 and see if a few discarded answers get recovered cheaply.

3. **Is separated faster than vanilla?** At small N (chunk 2): yes, 19 vs 28 min — but only because vanilla idled 3 workers. At leaderboard scale: no — empirical base completes 240 in ~3–4h; separated would add ~1h for extra TTT (no amortization). Both fit 12h easily. Time-advantage hybrid argument is weak; defer until accuracy data from chunks 3–12.

4. **Why don't `[EarlyStopOnLowLoss]` diagnostic prints appear in r1/r2 logs?** The class is in the .ipynb source, `callbacks=[EarlyStopOnLowLoss(...)]` is wired, training losses (0.017–0.162) cross the 0.05 threshold, but the callback's print never reaches the log. Either Unsloth's `UnslothFixedTrainer` filters callbacks, or `on_log` isn't called by the Unsloth-monkeypatched trainer. Tuning the threshold blind is unwise until understood.

5. **What budgets survive the 12h rerun?** Empirical: base notebook completes full 240 in ~3–4h on L4x4. r2 vanilla ≈ same (~4h); separated adds ~1h for extra TTT (~5h). Both fit 12h easily. Bump `RUN_HOURS` to ~6h vanilla / ~7h separated for margin (env override in rerun mode). `puzzle_timeout=2400s` is fine.

## Next experiments (priority order)

### P1 — DONE — old-chunked design (chunks 0+1, pre-r1) ✓

Forensics in `results/archive/results_vanilla-chunked-*` (gitignored). 1/5 then 0/5; multi-test subtest skips from 1200s cap; selection algo not the bottleneck on those puzzles.

### P1.5 — DONE — r1 design refactor: chunk-on-originals + multi-only + random IDs ✓

49 multi-test originals → 13 chunks of 4. EarlyStopOnLowLoss callback wired (but see open Q4). 172-puzzle separated dataset (seed=2602) on Kaggle.

### P2 — DONE — r1 chunks 0+1 ✓

- Chunk 0 (4 originals, 8 subtests): vanilla 0/8, separated 0/8 — capability-bound.
- Chunk 1 (4 originals, 9 subtests including one n=3): vanilla 2/9 (`1ae2feb7_2`, `20270e3b_1`), separated 1/9 (`1ae2feb7_2`).
- First confirmed selection-algo failure: `1ae2feb7_1` had the right answer at rank 7 in vanilla.

### P2.5 — DONE — r2 design overhaul ✓

- `apply_puzzle_timeout` patch: DFS 540→1080s, per-puzzle 1200→2400s. Eliminates the chunk-1-style budget-starvation skips.
- `RUN_HOURS` bumped (vanilla 3.0, separated 4.0).
- Rerun-mode branching (KAGGLE_IS_COMPETITION_RERUN): vanilla queues all 240 test puzzles; separated builds the separated set inline from `arc-agi_test_challenges.json` then runs all keys.
- Submission renaming + consolidated `submission.json` per kernel run + per-chunk JSON + human-id reports (`1a` … `49b`).
- New tools: `tools/_problem_index.py`, `tools/similarity_analysis.py` (with EXACT/CLOSE/ECHO/OFF/BLANK flags), `tools/merge_submissions.py`.

### P3 — IN FLIGHT — chunk 2 of r2 ✓ (just landed)

- Chunk 2 (4 originals 9–12: `221dfab4`/`247ef758`/`269e22fb`/`2d0172a1`, 8 subtests):
  - Vanilla 2/8 (`2d0172a1` both subtests).
  - Separated 2/8 (same puzzle, both subtests).
  - **Tied** on this chunk. 0/8 discarded in either arm.
  - 8/8 attempted in both arms — `apply_puzzle_timeout` paid off.
  - Wall: vanilla 28 min, separated 19 min — separated 50% faster (see open Q3).

### P4 — Walk chunks 3–12 of r2 (10 chunks remaining)

Per chunk: bump `CHUNK_INDEX` default in build script, rebuild, sync to push folder, push, manual web-UI attach (one-time per slug), Save Version & Run All, then `tools/save_kernel_run.py` + `tools/analyze_discarded_answers.py` + `tools/similarity_analysis.py`.

After P4, we have full chunks 0–12 coverage on the 49 multi-test originals (101 subtests). At that point either:
- Isolation has recovered ≥3 subtests vanilla missed → worth a leaderboard run.
- Isolation has 0 wins where vanilla missed → abandon coverage hypothesis, pivot to time-advantage hybrid (open Q3).

### P5 — Hybrid ensemble for final submission (user 2026-05-06)

Defer until chunks 3–12 supply accuracy data. The chunk-2 time advantage for separated doesn't survive scaling (open Q3); the accuracy argument is currently null. If we do build it:
- Run vanilla on all 240 test puzzles with a **tighter** `puzzle_timeout` (~720s) so it covers everything in 12 h.
- Identify puzzles vanilla skipped or used the full cap on (likely the n≥3 multi-test originals + a few hard n=2 ones).
- Run separated only on those, with the looser cap (since fewer puzzles = more headroom per).
- Consolidate via `tools/merge_submissions.py` (vanilla as base, separated overrides where vanilla had a placeholder or worse cell-match score).

Don't build until chunks 3–12 give us a confident ranking of which subtests separated beats vanilla on (or proves it never does, in which case skip P5).

### P6 — Investigate the EarlyStop diagnostic mystery

`[EarlyStopOnLowLoss]` prints don't reach the log despite the class being instantiated. Two cheap probes:
- Add a `__del__` print to confirm the callback object lives through TTT.
- Replace the `print` calls with `sys.stderr.write` to bypass any captured stdout.
- Check if Unsloth's monkeypatched trainer skips `TrainerCallback.on_log`.

Useful before tuning the threshold or relying on early stopping for the leaderboard rerun budget (open Q5).

### P7 — Leaderboard submission build

The r2 notebooks already detect `KAGGLE_IS_COMPETITION_RERUN` and produce a 240-shape `submission.json`. Outstanding:

- Bump `RUN_HOURS` for rerun mode (env override in `cell_run_src` or build-time conditional): ~6h vanilla / ~7h separated. Empirical base runtime is ~3–4h on full set; defaults of 3.0/4.0 are too tight.
- Manual web-UI attach on the submission slug.
- **Plan: submit both kernels** (user 2026-05-06) — math too close to predict; let the leaderboard resolve empirically.

### P8 — Selection-algorithm tweak (open Q2)

Bump `n_guesses` from 2 to 3 or 5 in `ArcDecoder` and re-score chunk 1 vanilla retroactively (the `1ae2feb7_1` rank-7 case). If we'd recover that subtest at n=3, run a chunk subset with the wider selection. Cheap one-day experiment.

### P9 (longer-term) — train our own grids model

Goal: own the model, not depend on sorokin. Probably qwen3-4b base + grid-format SFT on ARC train set + augmentation. Open subquestions: training infra (Kaggle TPU? Lambda? local 4090?), checkpoint sharing path, dataset format.

## Decisions made

- **r1 was the original chunked design (chunks 0+1); r2 doubles budgets, adds rerun mode, renames outputs, adds human IDs.** Bump to `rN+1` only for substantial design overhauls, not chunk progression.
- **Mirror as Kaggle Dataset, not Model.** Model API upload kept hitting `licenseName` validation; Dataset path worked first try. Always **manually web-UI attach** to new kernel slugs.
- **Chunk size 4 ORIGINALS** (not separated keys). Multi-only chunking (49 originals = 13 chunks).
- **Random hex IDs in the separated dataset (seed=2602).** Qwen never sees the puzzle ID (formatter only handles grids), so the hex format is a Kaggle-distribution-matching choice. For analysis output we translate to human IDs (`1a`/`12b`) via `_problem_index.py`.
- **Submissions named explicitly:** `submission_vanilla120_chunk{N}.json`, `submission_sep172_chunk{N}.json` (human-keyed), `submission.json` (full 120-shape with placeholders, mergeable).
- **Result folder format:** `results_subprobs-<descriptor>-<rest>-<duration>` so chunks group when sorted.
- **EarlyStopOnLowLoss is in the build pipeline** (threshold=0.05, patience=2, logging_steps=16) — but its diagnostic prints don't appear in logs. Open Q4.
- **Don't mirror competition data.** Comes from `competition_sources`, durable for active competition.

## Notes on the broader landscape

Few public ARC-AGI grid-tuned models exist on Kaggle Models. NVARC's qwen3 grid SFT line (sorokin's `qwen3_4b_grids15_sft139`, `qwen3_2b_grids15_sft141`) is one of the only public artifacts; the other 4B variant in that line was already deleted. Top performers historically keep weights private. TinyRecursiveModels has open checkpoints but on HuggingFace, not Kaggle, and uses a different architecture entirely. So sorokin's mirrors are essentially the only public NVARC-style weights — high concentration risk, hence we own a copy now.
