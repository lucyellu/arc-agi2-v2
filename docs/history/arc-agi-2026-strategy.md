# ARC-AGI 2026 — Research Notes & Strategy

*April 2026 | Covers AGI-2 (static grid) and AGI-3 (interactive game) tracks*

---

## Part I: AGI-2 (ARC Prize 2026 — Static Grid Challenge)

### Background

AGI-2 tests abstract reasoning: given a handful of input→output grid examples, infer the transformation rule and apply it to a new input. Grids are ≤30×30, up to 10 colors. The challenge is that each puzzle encodes a distinct rule — there is no shared vocabulary across tasks. This makes generalization from training data nearly impossible without test-time adaptation.

**Current position:** 28.19% public LB. NVARC (the 2025 winning approach) scored 27.64% using Qwen3-4B + test-time training (TTT). We're already at or above last year's winning score using the same stack.

**The ceiling:** ARChitects hit 53.5% in 2024 by ensembling TTT with program synthesis. Going from 28% → 40%+ requires a fundamentally different second method — not just tuning the existing one.

---

### What's Working (Don't Break It)

- **Qwen3-4B bf16 with LoRA (r=256) via Unsloth** — the proven model. No quantization (community finding: quantization hurts accuracy noticeably).
- **NVARC TTT hyperparameters** — lr=5e-5, r=256, alpha=32, n_aug=16, EarlyStop(threshold=0.05, patience=2), max_seq=8192. Validated at 30.97% in reproduction.
- **Augmentation** — transpose + 3 rotations + color permutations. Already in the pipeline.

---

### Hypothesis 1: Multi-Sub-Problem Tasks Are the Bottleneck

Several AGI-2 tasks present as a single grid containing multiple independent transformation regions — visually "stitched" sub-grids. The model currently tries to apply one global rule and fails. These tasks likely account for a disproportionate fraction of the remaining error gap.

**The diagnostic question:** Is the model failing because it can't *infer the rule* locally, or because its grid *reconstruction* is global (it can't limit changes to one region)?

**Proposed test (no new weights needed):** Format outputs as a sparse diff — a list of `(row, col, new_color)` tuples for changed cells only, reconstructing the full grid at eval time. This forces the model to commit only to cells it believes changed, making multi-region tasks structurally equivalent to single-region tasks. If accuracy on multi-sub-problem tasks improves in delta format vs. full-grid format with the same base model, the bottleneck is reconstruction, not inference.

**If confirmed:** Train a LoRA on delta-format outputs. This is a prompt/format change more than an architecture change — low implementation cost.

---

### Hypothesis 2: Stitched-Grid Training Teaches Positional Independence

If multi-sub-problem failure is partly about inference (not just reconstruction), the model needs to learn that rules can apply independently in spatially separated regions.

**Proposed approach:** Generate synthetic training examples by stitching 2–4 existing AGI-2 tasks side-by-side (separated by a blank row or column). Train a new LoRA on these. The model must learn to apply rules per-region rather than globally.

Two 30×30 grids stitched = 60×30 = 1800 tokens, well within the 8192 context limit. 2×2 stitching of 30×30 grids = 3600 tokens — still fine.

**Key design question:** The separator between regions must be visually consistent (same color, always present) so the model can learn the boundary signal. A blank row of color 0 is the simplest choice.

**Risk:** This requires new weights (new LoRA). Do the delta format test first — if failure is reconstruction-only, stitching may not be necessary.

---

### Hypothesis 3: Program Synthesis Ensemble Is the Path to 40%+

Every top-tier submission beyond ~35% has ensembled transduction (TTT) with induction (program synthesis). The two methods solve different subsets of tasks — their outputs are largely complementary.

**How it works:** For each task, generate candidate Python functions that correctly map all demo inputs to demo outputs. Score candidates by pass rate on the demos. Take top-k programs and ensemble their predictions with TTT outputs.

The key insight is that program synthesis doesn't require training — it's a search problem, and GPT-4o / o4-mini can generate plausible programs quickly. Ryan Greenblatt's standalone approach (no TTT, pure program search) reached 42% on AGI-1. Combined with TTT, the ceiling is substantially higher.

**The gap to close:** We don't currently have a program synthesis component. This is the highest-leverage addition for AGI-2 if the goal is pushing past 35%.

---

### AGI-2 Priority Sequence

1. **Delta format experiment** — zero-cost signal, L4, ~15 min. Identifies whether multi-sub-problem failure is reconstruction or inference.
2. **Program synthesis ensemble** — highest ceiling, independent of GPU experiments.
3. **Stitched-grid LoRA** — new weights, only pursue if delta format experiment shows inference failure.

---

## Part II: AGI-3 (ARC Prize 2026 — Interactive Game Challenge)

### Background

AGI-3 is structurally different. Rather than static grid-to-grid transformations, agents must interact with live game environments: observe pixel frames, take actions, and infer the hidden rule of each game without being told what it is. Grids are 64×64, up to 16 colors. Human players score 100%; frontier LLMs score below 1%.

There are two independent tracks being pursued:

**Track A (BFS / direct solver):** Load the game's Python source locally, simulate the environment, and search for a winning action sequence. This is already the best-scoring approach at 0.27% LB, driven primarily by keyboard-game victories.

**Track B (rule induction):** Extract (start, end) frame pairs from human replay recordings, treat them as AGI-2-style few-shot puzzles, and use the same TTT pipeline to predict the goal state for unseen levels.

---

### Track A — BFS/Solver: Where Things Stand

The BFS approach works when the game can be fully simulated locally (environment files are available) and the action space is tractable. Current wins come from `sb26`, a keyboard game where BFS finds solutions reliably.

The bottleneck is the action space on click-based games: brute-force BFS over all pixel positions is intractable. The sprite-click heuristic (restrict candidates to `sys_click`-tagged sprites) reduces the search space by 30–1000×, enabling BFS on click games.

**Near-term focus for Track A:** Extend the target-guided BFS to more environments. Priority should be given to games with smaller or more predictable action spaces.

---

### Track B — Rule Induction: The Hypothesis

The core idea: each AGI-3 game encodes a transformation rule (e.g., "player moves to collect all items of type X before time runs out"). If you extract the start and end state of each level from a human replay, you get a (before, after) demonstration pair that looks exactly like an AGI-2 task. With enough demonstrations of the same rule across different levels, a TTT-adapted model should be able to predict the end state of a new level.

**The original blocker:** NVARC's `arc_loader.py` (the AGI-2 TTT pipeline) was hard-coded for ≤10 colors and ≤30×30 grids. AGI-3 frames are 64×64 with 16 colors — deeply out-of-distribution.

**The solution implemented:** Palette reduction via adjacency-aware color merging. Colors that never share a cell edge can be merged without changing the topological structure of the grid. This collapses 16-color grids to ≤10 without altering the visible rules. Dataset (`arc-agi-3-induced-v2`) is built; TTT notebooks are ready to run pending a Kaggle environment fix.

**Open empirical question:** Will TTT actually help? It works on AGI-2 because each puzzle has one consistent rule across all demo pairs. AGI-3 has a complication: some environments have variable grid sizes across levels (e.g., `lp85` grows from 27×32 to 63×63). The same rule looks structurally different at different scales when rendered to a fixed 64×64 viewport — this may prevent the model from finding a consistent pattern.

---

### Track B — The Intermediate Frame Idea

Each AGI-3 human replay contains every frame of the game, not just start and end. Every consecutive pair `(frame_n, frame_n+1)` is a single-action transformation — the simplest possible version of the rule, with one edit at a time.

**What this offers:**
- 10–50× more training pairs per environment, at no additional data collection cost
- Each sub-step is simpler and more consistent than the full (start→end) transformation
- This is process supervision rather than outcome supervision — a known improvement in reasoning tasks

**Hypothesis:** Training TTT on sub-step pairs should generalize better than training on full (start→end) pairs, because the model is learning the atomic primitive of the rule rather than the composed result.

**The cleanest test:** For 2–3 environments, run TTT with (a) only (start, end) pairs and (b) all consecutive frame pairs. Compare cell accuracy on the held-out test level. This is a direct controlled comparison of process vs. outcome supervision.

---

### Overlap: What AGI-3 Work Gives Back to AGI-2

The rule induction track is not just relevant to AGI-3. The 25 induced tasks (one per game environment) represent a novel source of AGI-2-format training data with:
- Larger grids (up to 64×64 after palette reduction, still valid for the patched pipeline)
- Semantically different transformation types (game-world physics rather than geometric/color puzzles)
- Independent generation from the existing AGI-2 training distribution

Once TTT is validated on these tasks, the induced dataset can augment AGI-2 fine-tuning directly — adding distribution diversity that the model hasn't seen before.

---

### AGI-3 Priority Sequence

1. **Run TTT Phase 3 notebooks** — fix the docker image issue (numpy conflict from mismatched Kaggle environment) before re-pushing. Start with 2B model; check per-env cell accuracy, not just aggregate.
2. **Intermediate frame experiment** — once the TTT pipeline is validated, switch from (start, end) to consecutive frame pairs for training. Expect a meaningful accuracy improvement.
3. **Scale normalization (if TTT stalls)** — if variable grid size is the dominant failure mode, downscale each level to its intrinsic game grid dimensions before palette reduction. This normalizes the rule's visual appearance across levels.
4. **Hybrid integration** — once the rule inducer can predict goal states reliably for some environments, wire it into Track A as a goal-provider when BFS times out.

---

## Summary: What Is Different About Each Competition

| | AGI-2 | AGI-3 |
|---|---|---|
| **Task structure** | Static: given demos, predict output | Interactive: observe, act, infer rule |
| **Current score** | 28.19% | 0.27% |
| **Primary approach** | Qwen3-4B TTT (NVARC pipeline) | BFS solver + rule induction |
| **Key bottleneck** | Multi-sub-problem tasks; ceiling without program synthesis | Interaction loop; grid scale/color OOD for models |
| **Highest-leverage next step** | Delta format test → program synthesis | Fix TTT environment; intermediate frame training |
| **Shared technique** | Palette reduction produces AGI-2-compatible data from AGI-3 games | TTT pipeline is reused directly |
| **To reach next threshold** | 35%+: program synthesis ensemble | 1%+: validated TTT on at least a subset of envs |

The most important cross-cutting insight: **the AGI-3 rule induction pipeline, once working, generates novel AGI-2 training data for free.** Every environment solved in the rule induction track is a new puzzle type that can strengthen the AGI-2 model's generalization.
