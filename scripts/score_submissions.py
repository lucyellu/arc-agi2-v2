"""
Local submission scorer and comparator.

Scores submission files against the 2026 evaluation solutions and shows
per-level breakdown. Supports multiple submissions side-by-side.

Usage:
  python score_submissions.py                          # auto-detect all local submissions
  python score_submissions.py path/to/sub1.json ...   # explicit files

NOTE: The 2026 eval set (120 puzzles) is separate from the test set (240 puzzles).
Kaggle's leaderboard score comes from the test set. This script scores the eval set only.
"""

import json
import os
import glob
import sys
import time
from pathlib import Path
from datetime import datetime

# -- Paths ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "arc-prize-2026-arc-agi-2"
EVAL_CHALLENGES = DATA_DIR / "arc-agi_evaluation_challenges.json"
EVAL_SOLUTIONS = DATA_DIR / "arc-agi_evaluation_solutions.json"

# Fallback to results/submissions copies
if not EVAL_CHALLENGES.exists():
    EVAL_CHALLENGES = ROOT / "results" / "submissions" / "arc-agi_evaluation_challenges.json"
    EVAL_SOLUTIONS = ROOT / "results" / "submissions" / "arc-agi_evaluation_solutions.json"

# ── Complexity formula ─────────────────────────────────────────────────────────
def calc_complexity(puzzle):
    train = puzzle.get("train", [])
    all_grids = [e["input"] for e in train] + [e["output"] for e in train]
    all_grids = [g for g in all_grids if g]
    if not all_grids:
        return 0
    areas  = [len(g) * (len(g[0]) if g and g[0] else 0) for g in all_grids]
    max_area = max(areas) if areas else 0
    colors = len(set(v for g in all_grids for row in g for v in row))
    size_ch = sum(
        1 for e in train
        if e.get("input") and e.get("output") and (
            len(e["input"]) != len(e["output"]) or
            len(e["input"][0]) != len(e["output"][0])
        )
    )
    n_train = len(train)
    return max_area + colors * 3 + 50 * size_ch + 20 * max(0, 3 - n_train)


def assign_levels(challenges):
    """Return dict {puzzle_id: level 1-10} using rank-based decile."""
    ids = sorted(challenges.keys())
    complexities = {pid: calc_complexity(challenges[pid]) for pid in ids}
    sorted_ids = sorted(ids, key=lambda k: complexities[k])
    n = len(sorted_ids)
    level = {k: min(10, int(i * 10 / n) + 1) for i, k in enumerate(sorted_ids)}
    return level, complexities


def _attempt_pair(att):
    """Return (attempt_1, attempt_2) from common ARC submission shapes."""
    if isinstance(att, dict):
        return att.get("attempt_1"), att.get("attempt_2")
    if isinstance(att, list):
        return (att[0] if len(att) > 0 else None), (att[1] if len(att) > 1 else None)
    return None, None


def score_submission(sub, solutions):
    """Returns {puzzle_id: [True/False per test output]} for all puzzle ids in sub."""
    results = {}
    for pid, attempts in sub.items():
        sol = solutions.get(pid, [])
        if not sol:
            results[pid] = None  # puzzle not in eval set
            continue
        per_test = []
        for i, expected in enumerate(sol):
            att = attempts[i] if isinstance(attempts, list) and i < len(attempts) else None
            a1, a2 = _attempt_pair(att)
            per_test.append(a1 == expected or (a2 is not None and a2 == expected))
        results[pid] = per_test
    return results


def is_trivial(attempts):
    for att in attempts:
        if isinstance(att, dict):
            a1 = att.get("attempt_1")
        elif isinstance(att, list):
            a1 = att[0] if att else None
        else:
            a1 = None
        if a1 not in (None, [[0]]):
            return False
    return True


def is_trivial_test(attempts, index):
    if not isinstance(attempts, list) or index >= len(attempts):
        return True
    a1, _ = _attempt_pair(attempts[index])
    return a1 in (None, [[0]])


def load_meta_times(meta_dir):
    """Load per-puzzle timing from .meta.json files if available."""
    times = {}
    if not os.path.isdir(meta_dir):
        return times
    for f in glob.glob(os.path.join(meta_dir, "*.meta.json")):
        try:
            d = json.load(open(f))
            times[d["key"]] = d.get("solve_time_s", 0)
        except Exception:
            pass
    return times


def find_submissions():
    """Auto-detect submission files in known locations."""
    patterns = [
        str(ROOT / "results" / "results_*" / "submission.json"),
        str(ROOT / "results" / "submissions" / "*.json"),
        str(ROOT / "results" / "*.json"),
    ]
    found = []
    for p in patterns:
        for f in glob.glob(p):
            name = os.path.basename(f)
            if name.startswith("arc-agi_") or "solutions" in name or "challenges" in name:
                continue
            found.append(f)
    # Also check kaggle output directories
    return sorted(set(found))


def label_submission(path):
    """Make a short human-readable label from file path."""
    parts = path.replace("\\", "/").split("/")
    # Use parent dir name if informative
    if len(parts) >= 2:
        parent = parts[-2]
        if "results_" in parent:
            return parent.replace("results_", "").strip("[]")
    return os.path.splitext(parts[-1])[0]


def print_table(rows, headers, col_widths=None):
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    sep = "  ".join("-" * w for w in col_widths)
    header_str = "  ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_str)
    print(sep)
    for row in rows:
        print("  ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load eval data
    if not EVAL_SOLUTIONS.exists():
        print(f"ERROR: Solutions not found at {EVAL_SOLUTIONS}")
        sys.exit(1)

    challenges = json.load(open(EVAL_CHALLENGES, encoding="utf-8"))
    solutions  = json.load(open(EVAL_SOLUTIONS,  encoding="utf-8"))
    print(f"Loaded {len(challenges)} eval challenges, {len(solutions)} solutions")

    level_map, complexity_map = assign_levels(challenges)

    # Determine which submission files to score
    if len(sys.argv) > 1:
        sub_paths = sys.argv[1:]
    else:
        sub_paths = find_submissions()
        if not sub_paths:
            print("No submission files found. Pass paths as arguments.")
            sys.exit(1)
        print(f"\nAuto-detected {len(sub_paths)} submission files:")
        for p in sub_paths:
            print(f"  {p}")

    print()

    # Score each submission
    all_results = []
    for path in sub_paths:
        if not os.path.exists(path):
            print(f"SKIP (not found): {path}")
            continue
        try:
            sub = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"SKIP (parse error): {path} — {e}")
            continue

        # Only keep eval puzzle ids
        eval_ids = set(solutions.keys())
        sub_eval = {k: v for k, v in sub.items() if k in eval_ids}
        if not sub_eval:
            print(f"SKIP (no eval puzzles): {path}")
            continue

        results  = score_submission(sub_eval, solutions)
        label    = label_submission(path)

        # Try to find meta timing dir alongside the submission
        meta_dir = os.path.dirname(path)
        times    = load_meta_times(meta_dir)

        total_eval = len(eval_ids)
        total_subtests = sum(len(solutions[pid]) for pid in eval_ids)
        attempted = sum(
            1
            for pid in eval_ids
            for i in range(len(solutions[pid]))
            if not is_trivial_test(sub.get(pid, []), i)
        )
        correct_subtest_ids = [
            f"{pid}[{i}]"
            for pid, oks in results.items()
            if oks is not None
            for i, ok in enumerate(oks)
            if ok
        ]
        correct_puzzle_ids = [
            pid
            for pid, oks in results.items()
            if oks is not None and len(oks) > 0 and all(oks)
        ]
        n_correct = len(correct_subtest_ids)
        n_puzzles_correct = len(correct_puzzle_ids)
        pct = n_correct / total_subtests * 100

        print(f"{'='*60}")
        print(f"  {label}")
        print(f"  Path: {path}")
        print(f"  Eval subtests: {n_correct}/{total_subtests} = {pct:.2f}%")
        print(f"  Fully solved puzzles: {n_puzzles_correct}/{total_eval}")
        print(f"  Non-trivial attempts: {attempted}/{total_subtests}")
        if times:
            total_t = sum(times.values())
            mean_t  = total_t / len(times)
            print(f"  Timing: {len(times)} puzzles timed, mean {mean_t:.0f}s, total {total_t/3600:.1f}h")
        print()

        # Per-level breakdown
        level_stats = {}
        for pid in eval_ids:
            lv = level_map.get(pid, 0)
            if lv not in level_stats:
                level_stats[lv] = {"total": 0, "attempted": 0, "correct": 0, "times": []}
            n_tests = len(solutions[pid])
            level_stats[lv]["total"] += n_tests
            level_stats[lv]["attempted"] += sum(
                1 for i in range(n_tests)
                if not is_trivial_test(sub.get(pid, []), i)
            )
            oks = results.get(pid) or []
            level_stats[lv]["correct"] += sum(1 for ok in oks if ok)
            if pid in times:
                level_stats[lv]["times"].append(times[pid])

        rows = []
        for lv in sorted(level_stats.keys()):
            s = level_stats[lv]
            acc = f"{s['correct']}/{s['total']}"
            pct_lv = f"{s['correct']/s['total']*100:.1f}%"
            att = str(s["attempted"])
            if s["times"]:
                avg_t = f"{sum(s['times'])/len(s['times']):.0f}s"
                tot_t = f"{sum(s['times'])/60:.0f}m"
            else:
                avg_t = "-"
                tot_t = "-"
            rows.append([f"L{lv}", s["total"], att, acc, pct_lv, avg_t, tot_t])

        headers = ["Level", "N", "Attempted", "Correct", "Acc%", "AvgTime", "TotalTime"]
        col_w   = [7, 5, 9, 10, 6, 8, 10]
        print_table(rows, headers, col_w)
        print()

        all_results.append({
            "label": label,
            "path": path,
            "n_correct": n_correct,
            "n_total": total_subtests,
            "n_puzzles_correct": n_puzzles_correct,
            "n_puzzles_total": total_eval,
            "pct": pct,
            "attempted": attempted,
            "correct_ids": correct_subtest_ids,
            "correct_puzzle_ids": correct_puzzle_ids,
            "level_stats": level_stats,
        })

    # Side-by-side comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        headers = ["Submission"] + [r["label"][:20] for r in all_results]
        col_w = [12] + [22] * len(all_results)
        rows = [
            ["Score"] + [f"{r['n_correct']}/{r['n_total']} ({r['pct']:.2f}%)" for r in all_results],
            ["Attempted"] + [str(r["attempted"]) for r in all_results],
        ]
        # Per-level
        all_levels = sorted(set(lv for r in all_results for lv in r["level_stats"]))
        for lv in all_levels:
            row = [f"L{lv} correct"]
            for r in all_results:
                s = r["level_stats"].get(lv, {})
                c, t = s.get("correct", 0), s.get("total", 0)
                row.append(f"{c}/{t} ({c/t*100:.0f}%)" if t else "—")
            rows.append(row)
        print_table(rows, headers, col_w)

        # Puzzles correct in one but not another
        if len(all_results) == 2:
            a_ids = set(all_results[0]["correct_ids"])
            b_ids = set(all_results[1]["correct_ids"])
            only_a = a_ids - b_ids
            only_b = b_ids - a_ids
            both   = a_ids & b_ids
            print(f"\nBoth correct: {len(both)}")
            print(f"Only in '{all_results[0]['label']}': {len(only_a)} — {sorted(only_a)}")
            print(f"Only in '{all_results[1]['label']}': {len(only_b)} — {sorted(only_b)}")

    # Save report JSON
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = ROOT / "results" / f"score_comparison_{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "generated_at": stamp,
            "eval_set_size": len(solutions),
            "note": "eval set (120) is separate from Kaggle test set (240); Kaggle leaderboard score is from test set",
            "submissions": [
                {k: v for k, v in r.items() if k != "level_stats"}
                for r in all_results
            ],
        }, f, indent=2)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
