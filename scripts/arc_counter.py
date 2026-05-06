"""
arc_counter.py — Color-count grid generator for ARC-AGI puzzles.

For each grid, produces a "count matrix": a 10×1 grid where row i = number of
squares with color i (colors 0–9). The diff matrix is output_counts − input_counts
(negative = color decreased, positive = increased, 0 = unchanged).

Two modes
---------
  challenges mode  — process challenges (+optional solutions) JSON
  submission mode  — process a submission JSON (attempt_1 / attempt_2 grids)

Two output formats (both modes)
--------------------------------
  count_only  — replaces spatial grids with count + diff matrices
  combined    — original grids + count/diff matrices side by side

Challenge JSON schema (count_only)
-----------------------------------
{
  "<puzzle_id>": {
    "train": [{"input": [[n0]...[n9]], "output": [[n0]...[n9]], "diff": [[d0]...[d9]]}, ...],
    "test":  [{"input": [[n0]...[n9]], "output": ..., "diff": ...}, ...]
  }
}

Submission JSON schema (count_only)
-------------------------------------
{
  "<puzzle_id>": [           // one entry per test pair
    {
      "attempt_1":        [[n0]...[n9]],   // 10×1 counts
      "attempt_2":        [[n0]...[n9]],
      "expected":         [[n0]...[n9]],   // only if solutions provided
      "diff_attempt_1":   [[d0]...[d9]],   // attempt_1 − expected
      "diff_attempt_2":   [[d0]...[d9]]
    }, ...
  ]
}

Submission JSON schema (combined) — same but keeps original grids and adds _counts keys.

CLI usage
---------
  # Challenges mode
  python arc_counter.py --challenges path/to/challenges.json [--solutions path/to/solutions.json]
      --out-count count.json --out-combined combined.json --mode both

  # Submission mode
  python arc_counter.py --submission path/to/submission.json [--solutions path/to/solutions.json]
      --out-count sub_count.json --out-combined sub_combined.json --mode both

Notebook usage
--------------
  from arc_counter import build_count_datasets, build_submission_count_datasets
"""

import json
import argparse
from pathlib import Path

COLORS = list(range(10))   # ARC colors 0–9


# ── Core helpers ──────────────────────────────────────────────────────────────

def count_colors(grid):
    """Return list of 10 ints: count_colors(grid)[c] = # squares with color c."""
    counts = [0] * 10
    for row in grid:
        for val in row:
            v = int(val)
            if 0 <= v <= 9:
                counts[v] += 1
    return counts


def counts_to_matrix(counts):
    """Convert list-of-10 counts → 10×1 grid (list of single-element lists)."""
    return [[c] for c in counts]


def diff_matrix(in_counts, out_counts):
    """Element-wise out_counts − in_counts → 10×1 grid."""
    return [[out_counts[i] - in_counts[i]] for i in range(10)]


# ── Per-puzzle transforms ─────────────────────────────────────────────────────

def puzzle_count_only(puzzle, solution=None):
    """
    Replace spatial grids with count matrices.
    solution: list of output grids for test cases (from solutions file), or None.
    """
    out = {"train": [], "test": []}

    for ex in puzzle.get("train", []):
        in_c  = count_colors(ex["input"])
        out_c = count_colors(ex["output"])
        out["train"].append({
            "input":  counts_to_matrix(in_c),
            "output": counts_to_matrix(out_c),
            "diff":   diff_matrix(in_c, out_c),
        })

    for idx, t in enumerate(puzzle.get("test", [])):
        in_c = count_colors(t["input"])
        entry = {"input": counts_to_matrix(in_c)}
        if solution is not None and idx < len(solution):
            sol_grid = solution[idx]
            # solutions file stores [[grid]] or [grid] — normalise
            if isinstance(sol_grid[0][0], list):
                sol_grid = sol_grid[0]
            out_c = count_colors(sol_grid)
            entry["output"] = counts_to_matrix(out_c)
            entry["diff"]   = diff_matrix(in_c, out_c)
        out["test"].append(entry)

    return out


def puzzle_combined(puzzle, solution=None):
    """
    Keep original grids; append count/diff matrices as extra keys.
    """
    import copy
    out = {"train": [], "test": []}

    for ex in puzzle.get("train", []):
        in_c  = count_colors(ex["input"])
        out_c = count_colors(ex["output"])
        entry = copy.deepcopy(ex)
        entry["input_counts"]  = counts_to_matrix(in_c)
        entry["output_counts"] = counts_to_matrix(out_c)
        entry["diff_counts"]   = diff_matrix(in_c, out_c)
        out["train"].append(entry)

    for idx, t in enumerate(puzzle.get("test", [])):
        in_c  = count_colors(t["input"])
        entry = copy.deepcopy(t)
        entry["input_counts"] = counts_to_matrix(in_c)
        if solution is not None and idx < len(solution):
            sol_grid = solution[idx]
            if isinstance(sol_grid[0][0], list):
                sol_grid = sol_grid[0]
            out_c = count_colors(sol_grid)
            entry["output_counts"] = counts_to_matrix(out_c)
            entry["diff_counts"]   = diff_matrix(in_c, out_c)
        out["test"].append(entry)

    return out


# ── Submission counter ────────────────────────────────────────────────────────

def _normalise_sol_grid(sol_item):
    """Solutions file may store [[grid]] or [grid] — unwrap to plain grid."""
    if isinstance(sol_item, list) and len(sol_item) > 0:
        if isinstance(sol_item[0], list) and len(sol_item[0]) > 0 and isinstance(sol_item[0][0], list):
            return sol_item[0]
    return sol_item


def submission_count_only(submission, solutions=None):
    """
    submission : {puzzle_id: [{attempt_1: grid, attempt_2: grid}, ...]}
    solutions  : {puzzle_id: [grid, ...]}  or None

    Returns count_only submission dict — same top-level keys, each value is a
    list-of-test-pairs where attempt grids are replaced by 10×1 count matrices.
    If solutions provided, also adds 'expected', 'diff_attempt_1', 'diff_attempt_2'.
    """
    out = {}
    for pid, pairs in submission.items():
        sol_list = solutions.get(pid) if solutions else None
        out_pairs = []
        for idx, pair in enumerate(pairs):
            entry = {}
            for key in ("attempt_1", "attempt_2"):
                grid = pair.get(key)
                if grid and not _is_dummy(grid):
                    entry[key] = counts_to_matrix(count_colors(grid))
            if sol_list is not None and idx < len(sol_list):
                sol_grid = _normalise_sol_grid(sol_list[idx])
                exp_c = count_colors(sol_grid)
                entry["expected"] = counts_to_matrix(exp_c)
                for key in ("attempt_1", "attempt_2"):
                    if key in entry:
                        att_c = [row[0] for row in entry[key]]
                        entry[f"diff_{key}"] = diff_matrix(exp_c, att_c)
            out_pairs.append(entry)
        out[pid] = out_pairs
    return out


def submission_combined(submission, solutions=None):
    """
    Same as submission_count_only but keeps original grids and adds _counts keys.
    """
    import copy
    out = {}
    for pid, pairs in submission.items():
        sol_list = solutions.get(pid) if solutions else None
        out_pairs = []
        for idx, pair in enumerate(pairs):
            entry = copy.deepcopy(pair)
            for key in ("attempt_1", "attempt_2"):
                grid = pair.get(key)
                if grid and not _is_dummy(grid):
                    entry[f"{key}_counts"] = counts_to_matrix(count_colors(grid))
            if sol_list is not None and idx < len(sol_list):
                sol_grid = _normalise_sol_grid(sol_list[idx])
                exp_c = count_colors(sol_grid)
                entry["expected_counts"] = counts_to_matrix(exp_c)
                for key in ("attempt_1", "attempt_2"):
                    if f"{key}_counts" in entry:
                        att_c = [row[0] for row in entry[f"{key}_counts"]]
                        entry[f"diff_{key}"] = diff_matrix(exp_c, att_c)
            out_pairs.append(entry)
        out[pid] = out_pairs
    return out


def _is_dummy(grid):
    """True if grid is the [[0]] placeholder (no real answer)."""
    return (isinstance(grid, list) and len(grid) == 1
            and isinstance(grid[0], list) and len(grid[0]) == 1
            and grid[0][0] == 0)


def build_submission_count_datasets(submission, solutions=None):
    """
    Returns (count_only_ds, combined_ds) for a submission file.
    """
    return submission_count_only(submission, solutions), submission_combined(submission, solutions)


# ── Dataset-level builder (main API) ─────────────────────────────────────────

def build_count_datasets(challenges, solutions=None):
    """
    challenges : dict  {puzzle_id: puzzle_dict}
    solutions  : dict  {puzzle_id: list_of_output_grids}  or None

    Returns (count_only_ds, combined_ds) — both are dicts with same keys as challenges.
    """
    count_ds    = {}
    combined_ds = {}

    for pid, puzzle in challenges.items():
        sol = solutions.get(pid) if solutions else None
        count_ds[pid]    = puzzle_count_only(puzzle, sol)
        combined_ds[pid] = puzzle_combined(puzzle, sol)

    return count_ds, combined_ds


# ── Notebook helper: auto-detect paths + mode, write outputs ─────────────────

def run_notebook(
    challenges_path,
    solutions_path=None,
    out_count_path="count_only.json",
    out_combined_path="combined.json",
    mode="both",
    verbose=True,
):
    """
    Load data, build datasets, write JSON files.
    Call this from a notebook cell. Returns (count_ds, combined_ds).
    """
    def log(msg):
        if verbose:
            print(msg)

    log(f"[arc_counter] Loading challenges: {challenges_path}")
    with open(challenges_path) as f:
        challenges = json.load(f)

    solutions = None
    if solutions_path and Path(solutions_path).exists():
        log(f"[arc_counter] Loading solutions:   {solutions_path}")
        with open(solutions_path) as f:
            solutions = json.load(f)
    else:
        log("[arc_counter] No solutions file — test diff/output omitted")

    log(f"[arc_counter] Processing {len(challenges)} puzzles...")
    count_ds, combined_ds = build_count_datasets(challenges, solutions)

    if mode in ("count_only", "both"):
        with open(out_count_path, "w") as f:
            json.dump(count_ds, f)
        log(f"[arc_counter] Written count_only  -> {out_count_path}")

    if mode in ("combined", "both"):
        with open(out_combined_path, "w") as f:
            json.dump(combined_ds, f)
        log(f"[arc_counter] Written combined     -> {out_combined_path}")

    log("[arc_counter] Done.")
    return count_ds, combined_ds


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ARC color-count grid generator")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--challenges",  help="Path to challenges JSON")
    grp.add_argument("--submission",  help="Path to submission JSON")
    ap.add_argument("--solutions",    default=None,   help="Path to solutions JSON (optional)")
    ap.add_argument("--out-count",    default="count_only.json")
    ap.add_argument("--out-combined", default="combined.json")
    ap.add_argument("--mode",         default="both",
                    choices=["count_only", "combined", "both"])
    args = ap.parse_args()

    def log(msg): print(msg)

    solutions = None
    if args.solutions and Path(args.solutions).exists():
        log(f"[arc_counter] Loading solutions: {args.solutions}")
        with open(args.solutions) as f:
            solutions = json.load(f)

    if args.challenges:
        run_notebook(
            challenges_path=args.challenges,
            solutions_path=args.solutions,
            out_count_path=args.out_count,
            out_combined_path=args.out_combined,
            mode=args.mode,
        )
    else:
        log(f"[arc_counter] Loading submission: {args.submission}")
        with open(args.submission) as f:
            submission = json.load(f)
        log(f"[arc_counter] Processing {len(submission)} puzzles...")
        count_ds, combined_ds = build_submission_count_datasets(submission, solutions)
        if args.mode in ("count_only", "both"):
            with open(args.out_count, "w") as f:
                json.dump(count_ds, f)
            log(f"[arc_counter] Written count_only -> {args.out_count}")
        if args.mode in ("combined", "both"):
            with open(args.out_combined, "w") as f:
                json.dump(combined_ds, f)
            log(f"[arc_counter] Written combined   -> {args.out_combined}")
        log("[arc_counter] Done.")


if __name__ == "__main__":
    main()
