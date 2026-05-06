"""For a finished run, check whether ANY candidate produced by the decoder
matched the correct answer — even if the selection algorithm ranked it below
attempt_1/attempt_2.

Each bz2 file in `<results_dir>/grouped_outputs/` (or `separated_outputs/`)
holds a list of decode candidates from one augmentation pass. The full-puzzle
candidate set is the union across all aug files for that subtest.

We:
1. Load every candidate solution for each subtest.
2. Apply the same selection algo as the decoder (score_full_probmul_3) to
   produce the actual ranking.
3. Find the rank of the correct answer (if it appears anywhere).
4. Report rank distribution + how often the right answer was generated but
   discarded by being ranked >2.
"""
import argparse
import bz2
import io
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def hashable(grid):
    return tuple(tuple(int(v) for v in row) for row in grid)


def score_full_probmul_3(guesses, baseline=3):
    import numpy as np
    inf_score = sum(baseline - g["beam_score"] for g in guesses)
    aug_score = float(np.mean([sum(baseline - s for s in g["score_aug"]) for g in guesses]))
    return inf_score + aug_score


def rank_candidates(candidates_per_aug):
    """Replicate ArcDecoder.score_full_probmul_3 logic: dedupe by grid, score, sort."""
    by_grid = {}
    for cand in candidates_per_aug:
        h = hashable(cand["solution"])
        by_grid.setdefault(h, [[], cand["solution"]])
        by_grid[h][0].append(cand)
    scored = [(score_full_probmul_3(group), grid) for group, grid in by_grid.values()]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored  # list of (score, grid) sorted best-first


def grids_equal(a, b):
    a = [list(map(int, row)) for row in a]
    b = [list(map(int, row)) for row in b]
    if len(a) != len(b):
        return False
    return all(ar == br for ar, br in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument(
        "--inference-subdir",
        default="grouped_outputs",
        help="grouped_outputs or separated_outputs",
    )
    ap.add_argument(
        "--solutions",
        type=Path,
        default=ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_solutions.json",
    )
    args = ap.parse_args()

    inf_dir = args.results_dir / args.inference_subdir
    if not inf_dir.is_dir():
        print(f"ERROR: {inf_dir} not found")
        sys.exit(1)

    solutions = json.loads(args.solutions.read_text())

    # Group bz2 files by their subkey (everything before the first '.')
    by_subkey = defaultdict(list)
    for f in inf_dir.iterdir():
        if not f.is_file():
            continue
        subkey = f.name.split(".")[0]  # e.g. "13e47133_0"
        by_subkey[subkey].append(f)

    if not by_subkey:
        print(f"ERROR: no files in {inf_dir}")
        sys.exit(1)

    rows = []
    for subkey, files in sorted(by_subkey.items()):
        # subkey shape: <pid>_<test_idx>
        pid, test_idx_str = subkey.rsplit("_", 1)
        try:
            test_idx = int(test_idx_str)
        except ValueError:
            print(f"  [skip] unrecognized subkey: {subkey}")
            continue

        if pid not in solutions or test_idx >= len(solutions[pid]):
            print(f"  [skip] no solution for {subkey}")
            continue
        expected = solutions[pid][test_idx]

        # Load all candidates from all aug files
        candidates = []
        for f in files:
            try:
                with bz2.BZ2File(f, "r") as fh:
                    data = pickle.load(fh)
                candidates.extend(data)
            except Exception as e:
                print(f"  [skip] {f.name}: {e}")
                continue

        if not candidates:
            rows.append({
                "subkey": subkey,
                "n_files": len(files),
                "n_candidates": 0,
                "n_unique_grids": 0,
                "rank_of_correct": None,
                "correct_in_top2": False,
                "correct_anywhere": False,
                "discarded_correct": False,
            })
            continue

        ranked = rank_candidates(candidates)
        n_unique = len(ranked)

        rank_of_correct = None
        for i, (score, grid) in enumerate(ranked, start=1):
            if grids_equal(grid, expected):
                rank_of_correct = i
                break

        correct_in_top2 = rank_of_correct is not None and rank_of_correct <= 2
        correct_anywhere = rank_of_correct is not None
        discarded_correct = correct_anywhere and not correct_in_top2

        rows.append({
            "subkey": subkey,
            "n_files": len(files),
            "n_candidates": len(candidates),
            "n_unique_grids": n_unique,
            "rank_of_correct": rank_of_correct,
            "correct_in_top2": correct_in_top2,
            "correct_anywhere": correct_anywhere,
            "discarded_correct": discarded_correct,
        })

    # Print table
    print()
    print(f"{'subkey':<20} {'aug':>4} {'cand':>5} {'uniq':>5} {'rank':>5} {'top2':>5} {'any':>4} {'discarded':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['subkey']:<20} {r['n_files']:>4} {r['n_candidates']:>5} {r['n_unique_grids']:>5} "
              f"{str(r['rank_of_correct']):>5} {('Y' if r['correct_in_top2'] else '.'):>5} "
              f"{('Y' if r['correct_anywhere'] else '.'):>4} {('YES' if r['discarded_correct'] else ''):>10}")

    n = len(rows)
    in_top2 = sum(r["correct_in_top2"] for r in rows)
    anywhere = sum(r["correct_anywhere"] for r in rows)
    discarded = sum(r["discarded_correct"] for r in rows)
    print()
    print(f"Subtests:                {n}")
    print(f"Correct in top-2:        {in_top2}/{n}")
    print(f"Correct in any rank:     {anywhere}/{n}")
    print(f"Right answer DISCARDED:  {discarded}/{n}  (model produced it but selection algo ranked it >2)")

    # Save reports — JSON for tooling, Markdown for human reading.
    json_path = args.results_dir / "discarded_answers_analysis.json"
    json_path.write_text(json.dumps({
        "n_subtests": n,
        "n_correct_in_top2": in_top2,
        "n_correct_anywhere": anywhere,
        "n_discarded_correct": discarded,
        "rows": rows,
    }, indent=2))
    print(f"\nSaved JSON: {json_path}")

    md_path = args.results_dir / "discarded_answers_analysis.md"
    lines = [
        f"# Discarded-answer analysis — {args.results_dir.name}",
        "",
        "**Question:** for each subtest, did the decoder ever produce the correct grid as one of its candidates? If yes, was it ranked in the top-2 (the attempts we submit) or discarded below that?",
        "",
        f"- Subtests with inference output: **{n}**",
        f"- Correct answer in top-2 (= we submitted it): **{in_top2}/{n}**",
        f"- Correct answer anywhere in candidate list: **{anywhere}/{n}**",
        f"- Right answer DISCARDED (model produced it, selection algo ranked it >2): **{discarded}/{n}**",
        "",
        "## Per-subtest detail",
        "",
        "| subkey | aug files | candidates | unique grids | correct rank | in top-2 | anywhere | discarded |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        rank = r["rank_of_correct"]
        rank_s = str(rank) if rank is not None else "—"
        lines.append(
            f"| `{r['subkey']}` | {r['n_files']} | {r['n_candidates']} | "
            f"{r['n_unique_grids']} | {rank_s} | "
            f"{'✓' if r['correct_in_top2'] else ''} | "
            f"{'✓' if r['correct_anywhere'] else ''} | "
            f"{'YES' if r['discarded_correct'] else ''} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if discarded == 0 and anywhere > 0:
        lines.append(f"All {anywhere} correct answers found by the model also made it into the top-2 submitted. **The selection algorithm is not the bottleneck** — when we miss, the model never produced the right grid.")
    elif discarded > 0:
        lines.append(f"**{discarded} subtest(s) had the right answer hidden in the candidate list.** Improving the selection algorithm could potentially recover these.")
    else:
        lines.append("No correct answers found in any candidate list. Pure capability/budget failures, not selection.")
    lines.append("")
    lines.append("Candidate-count breadth is also informative: a subtest with only 1–3 unique grids was under-explored (decode budget hit, max_score threshold too tight, or DFS branches died early). Subtests with 12+ unique grids that still missed are pure capability failures — the model explored a wide search but couldn't find the answer.")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved MD:   {md_path}")


if __name__ == "__main__":
    main()
