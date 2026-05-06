"""Programmatic equivalent of the arcviewer's similarity stats.

Three percentages live side by side in the output and they MEAN DIFFERENT
THINGS — a high number is only "good" for one of them:

    attempt -> output    target: 100%  (only 100% counts as correct)
    attempt -> input     target: matches the input->output baseline
                          (a correct attempt diverges from the input by
                          exactly the amount the puzzle requires; an echo
                          failure stays at ~100%)
    input   -> output    intrinsic puzzle distance (informational baseline)

The printed table separates these explicitly with arrow-direction column
headers (a1->out%, a1->in%, in->out%) and a flag column (EXACT / CLOSE /
ECHO / OFF / BLANK) so partial matches aren't misread as success.

Aggregates match the viewer's `stats.json` field names where possible so
JSON output stays drop-in compatible with the viewer.

Usage:
    python tools/similarity_analysis.py <submission.json>
    python tools/similarity_analysis.py <submission.json> --solutions <path>
    python tools/similarity_analysis.py <submission.json> --diff <other.json>

Output:
    Console table + JSON saved next to the submission as
    <stem>_similarity.json. With --diff also writes
    <stem>_diff_<other>.md (side-by-side per-subtest table).
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOLUTIONS = ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_solutions.json"
DEFAULT_CHALLENGES = ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_challenges.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _problem_index import build_index
    HUMAN_INDEX = build_index()["subtest_to_human"]
except Exception:
    HUMAN_INDEX = {}


def is_real(grid):
    """Non-trivial = not None, not empty, not the [[0]] placeholder."""
    if grid is None:
        return False
    if grid == [[0]]:
        return False
    if not grid or not grid[0]:
        return False
    return True


def shape(grid):
    if not grid or not grid[0]:
        return (0, 0)
    return (len(grid), len(grid[0]))


def shape_match(a, b):
    return shape(a) == shape(b)


def cell_match_pct(a, b):
    """Fraction of cells equal. Returns 0 if shapes differ."""
    if not shape_match(a, b):
        return 0.0
    rows = len(a)
    if rows == 0:
        return 0.0
    cols = len(a[0])
    total = rows * cols
    if total == 0:
        return 0.0
    matches = sum(
        1
        for r in range(rows)
        for c in range(cols)
        if a[r][c] == b[r][c]
    )
    return matches / total


def exact_match(a, b):
    return shape_match(a, b) and cell_match_pct(a, b) == 1.0


def score_one(attempt, expected, input_grid):
    """Returns metrics dict for a single attempt vs expected (with input baseline)."""
    return {
        "is_real": is_real(attempt),
        "shape_match": shape_match(attempt, expected) if attempt is not None else False,
        "cell_match_pct": cell_match_pct(attempt, expected) if attempt is not None else 0.0,
        "exact_match": exact_match(attempt, expected) if attempt is not None else False,
        "echo_input_pct": cell_match_pct(attempt, input_grid) if attempt is not None else 0.0,
    }


def score_submission(submission, solutions, challenges):
    """Score every (key, test_idx) in the submission. Returns list of row dicts."""
    rows = []
    for key, expected_list in solutions.items():
        if key not in submission:
            continue
        attempts_list = submission[key]
        challenge = challenges.get(key, {})
        train_pairs = challenge.get("train", [])
        tests = challenge.get("test", [])
        n_train = len(train_pairs)
        n_tests = len(expected_list)

        for test_idx, expected in enumerate(expected_list):
            attempt = attempts_list[test_idx] if test_idx < len(attempts_list) else {}
            input_grid = tests[test_idx]["input"] if test_idx < len(tests) else []

            a1 = attempt.get("attempt_1") if isinstance(attempt, dict) else None
            a2 = attempt.get("attempt_2") if isinstance(attempt, dict) else None

            io_baseline = cell_match_pct(input_grid, expected)
            s1 = score_one(a1, expected, input_grid)
            s2 = score_one(a2, expected, input_grid)

            best_cell = max(s1["cell_match_pct"], s2["cell_match_pct"])
            best_exact = s1["exact_match"] or s2["exact_match"]

            subtest_id = f"{key}_{test_idx}"
            human_id = HUMAN_INDEX.get(subtest_id, "")

            rows.append({
                "puzzle_id": key,
                "test_index": test_idx,
                "subtest_id": subtest_id,
                "human_id": human_id,
                "n_train": n_train,
                "n_tests": n_tests,
                "input_to_output_pct": io_baseline,
                "a1": s1,
                "a2": s2,
                "best_cell_match_pct": best_cell,
                "best_exact_match": best_exact,
            })
    return rows


def aggregate(rows):
    n = len(rows)
    s1_real = sum(1 for r in rows if r["a1"]["is_real"])
    s2_real = sum(1 for r in rows if r["a2"]["is_real"])
    s1_correct = sum(1 for r in rows if r["a1"]["exact_match"])
    s2_correct = sum(1 for r in rows if r["a2"]["exact_match"])
    best_correct = sum(1 for r in rows if r["best_exact_match"])
    s1_real_correct = sum(1 for r in rows if r["a1"]["is_real"] and r["a1"]["exact_match"])
    s2_real_correct = sum(1 for r in rows if r["a2"]["is_real"] and r["a2"]["exact_match"])
    s1_sim = sum(r["a1"]["cell_match_pct"] for r in rows)
    s2_sim = sum(r["a2"]["cell_match_pct"] for r in rows)
    echo1 = sum(r["a1"]["echo_input_pct"] for r in rows) / n if n else 0.0
    echo2 = sum(r["a2"]["echo_input_pct"] for r in rows) / n if n else 0.0
    io_avg = sum(r["input_to_output_pct"] for r in rows) / n if n else 0.0

    by_n_train = {}
    by_subproblems = {}
    for r in rows:
        bk = str(r["n_train"])
        by_n_train.setdefault(bk, {"total": 0, "best_correct": 0, "s1_sim": 0.0, "s2_sim": 0.0})
        by_n_train[bk]["total"] += 1
        by_n_train[bk]["best_correct"] += int(r["best_exact_match"])
        by_n_train[bk]["s1_sim"] += r["a1"]["cell_match_pct"]
        by_n_train[bk]["s2_sim"] += r["a2"]["cell_match_pct"]

        bs = str(r["n_tests"])
        by_subproblems.setdefault(bs, {"total": 0, "best_correct": 0})
        by_subproblems[bs]["total"] += 1
        by_subproblems[bs]["best_correct"] += int(r["best_exact_match"])

    return {
        "total": n,
        "s1Correct": s1_correct,
        "s2Correct": s2_correct,
        "bestCorrect": best_correct,
        "s1RealCount": s1_real,
        "s2RealCount": s2_real,
        "s1RealCorrect": s1_real_correct,
        "s2RealCorrect": s2_real_correct,
        "s1SimSum": s1_sim,
        "s1Count": n,
        "s2SimSum": s2_sim,
        "s2Count": n,
        "s1AvgSim": s1_sim / n if n else 0.0,
        "s2AvgSim": s2_sim / n if n else 0.0,
        "bestAvgSim": sum(r["best_cell_match_pct"] for r in rows) / n if n else 0.0,
        "echoInputAvg_a1": echo1,
        "echoInputAvg_a2": echo2,
        "inputOutputBaseline": io_avg,
        "byNExamples": by_n_train,
        "bySubproblems": by_subproblems,
    }


def attempt_flag(s, baseline):
    """Diagnostic for a single attempt's metrics dict."""
    if s["exact_match"]:
        return "EXACT"
    out = s["cell_match_pct"]
    inp = s["echo_input_pct"]
    if not s["is_real"]:
        return "BLANK"
    if inp > 0.95:
        return "ECHO"
    if out < baseline:
        return "OFF"
    if out > 0.9:
        return "CLOSE"
    return ""


def print_summary(label, rows, agg):
    print(f"\n=== {label} ===")
    print(f"  subtests: {agg['total']}")
    print(f"  exact-match: a1={agg['s1Correct']}/{agg['total']}  a2={agg['s2Correct']}/{agg['total']}  best-of-2={agg['bestCorrect']}/{agg['total']}")
    print(f"  avg attempt->output: a1={agg['s1AvgSim']:.3f}  a2={agg['s2AvgSim']:.3f}  best={agg['bestAvgSim']:.3f}")
    print(f"  avg attempt->input:  a1={agg['echoInputAvg_a1']:.3f}  a2={agg['echoInputAvg_a2']:.3f}  (baseline input->output: {agg['inputOutputBaseline']:.3f})")
    print(f"  attempts non-trivial: a1={agg['s1RealCount']}/{agg['total']}  a2={agg['s2RealCount']}/{agg['total']}")
    print()
    print("  Legend (% values):")
    print("    a1->out% / a2->out%  attempt vs expected output (TARGET: 100% = correct)")
    print("    a1->in%  / a2->in%   attempt vs input (echo check; for a correct attempt converges to in->out%, NOT 100%)")
    print("    in->out%             input vs expected output (intrinsic transformation distance)")
    print("    flag      EXACT | CLOSE (>90% to out, not exact) | ECHO (>95% to in) | OFF (<baseline to out) | BLANK")
    print()
    hdr = (
        f"  {'human':<5} {'subtest':<14} "
        f"{'a1->out%':>9} {'a2->out%':>9} {'in->out%':>9} "
        f"{'a1->in%':>8} {'a2->in%':>8} "
        f"{'a1_flag':>7} {'a2_flag':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        baseline = r["input_to_output_pct"]
        a1_flag = attempt_flag(r["a1"], baseline)
        a2_flag = attempt_flag(r["a2"], baseline)
        print(
            f"  {r['human_id']:<5} {r['subtest_id']:<14} "
            f"{r['a1']['cell_match_pct']*100:>8.1f}% {r['a2']['cell_match_pct']*100:>8.1f}% "
            f"{baseline*100:>8.1f}% "
            f"{r['a1']['echo_input_pct']*100:>7.1f}% {r['a2']['echo_input_pct']*100:>7.1f}% "
            f"{a1_flag:>7} {a2_flag:>7}"
        )


def diff_mode(rows_a, rows_b, label_a, label_b, out_path):
    by_id_b = {r["subtest_id"]: r for r in rows_b}
    lines = [
        f"# Diff: {label_a} vs {label_b}",
        "",
        "Columns are share of cells matching expected output (target 100% = correct). "
        "`baseline` = intrinsic input-vs-output similarity for context — high baseline means the puzzle is near-identity and a high `->out%` may not be meaningful.",
        "",
        "| human | subtest | A best->out% | B best->out% | baseline | A exact | B exact | regression |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    a_only = b_only = both = 0
    for r in rows_a:
        b = by_id_b.get(r["subtest_id"])
        if b is None:
            continue
        a_ex = r["best_exact_match"]
        b_ex = b["best_exact_match"]
        if a_ex and b_ex:
            both += 1
        elif a_ex:
            a_only += 1
        elif b_ex:
            b_only += 1
        regress = "A->B lost" if (a_ex and not b_ex) else ("B->A lost" if (b_ex and not a_ex) else "")
        lines.append(
            f"| {r['human_id']} | `{r['subtest_id']}` | "
            f"{r['best_cell_match_pct']*100:.1f}% | {b['best_cell_match_pct']*100:.1f}% | "
            f"{r['input_to_output_pct']*100:.1f}% | "
            f"{'Y' if a_ex else '.'} | {'Y' if b_ex else '.'} | "
            f"{regress} |"
        )
    lines += [
        "",
        f"- both correct: {both}",
        f"- {label_a} only: {a_only}",
        f"- {label_b} only: {b_only}",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDiff written: {out_path}")
    print(f"  both correct: {both}   {label_a} only: {a_only}   {label_b} only: {b_only}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--solutions", type=Path, default=DEFAULT_SOLUTIONS)
    ap.add_argument("--challenges", type=Path, default=DEFAULT_CHALLENGES)
    ap.add_argument("--diff", type=Path, help="Second submission to compare against")
    args = ap.parse_args()

    submission = json.loads(args.submission.read_text())
    solutions = json.loads(args.solutions.read_text())
    challenges = json.loads(args.challenges.read_text())

    rows = score_submission(submission, solutions, challenges)
    agg = aggregate(rows)
    print_summary(args.submission.name, rows, agg)

    out_json = args.submission.with_name(args.submission.stem + "_similarity.json")
    out_json.write_text(json.dumps({"summary": agg, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_json}")

    if args.diff:
        sub_b = json.loads(args.diff.read_text())
        rows_b = score_submission(sub_b, solutions, challenges)
        agg_b = aggregate(rows_b)
        print_summary(args.diff.name, rows_b, agg_b)
        out_md = args.submission.with_name(
            f"diff_{args.submission.stem}_vs_{args.diff.stem}.md"
        )
        diff_mode(rows, rows_b, args.submission.stem, args.diff.stem, out_md)


if __name__ == "__main__":
    main()
