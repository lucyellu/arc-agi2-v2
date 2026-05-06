#!/usr/bin/env python3
"""Score a submission against the separated 172-problem eval dataset."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEP_DIR = ROOT / "data" / "arc-prize-2026-arc-agi-2-separated"
DEFAULT_SOLUTIONS = SEP_DIR / "arc-agi_evaluation_solutions_separated.json"
DEFAULT_MAPPING = SEP_DIR / "arc-agi_evaluation_separated_mapping.json"


def _attempt_pair(item):
    if isinstance(item, dict):
        return item.get("attempt_1"), item.get("attempt_2")
    if isinstance(item, list):
        return (item[0] if len(item) > 0 else None), (item[1] if len(item) > 1 else None)
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Score separated ARC eval submission")
    parser.add_argument("submission")
    parser.add_argument("--solutions", default=str(DEFAULT_SOLUTIONS))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument(
        "--report",
        default=str(ROOT / "results" / "separated_score_report.json"),
    )
    args = parser.parse_args()

    submission = json.loads(Path(args.submission).read_text(encoding="utf-8"))
    solutions = json.loads(Path(args.solutions).read_text(encoding="utf-8"))
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))

    rows = []
    for sid in sorted(solutions):
        expected = solutions[sid][0]
        attempts = submission.get(sid, [])
        attempt_item = attempts[0] if isinstance(attempts, list) and attempts else {}
        a1, a2 = _attempt_pair(attempt_item)
        correct = a1 == expected or (a2 is not None and a2 == expected)
        rows.append({
            "separated_id": sid,
            "correct": correct,
            **mapping.get(sid, {}),
        })

    correct_rows = [row for row in rows if row["correct"]]
    multi_rows = [row for row in rows if row.get("was_multi_test")]
    multi_correct = [row for row in multi_rows if row["correct"]]
    single_rows = [row for row in rows if not row.get("was_multi_test")]
    single_correct = [row for row in single_rows if row["correct"]]

    report = {
        "correct": len(correct_rows),
        "total": len(rows),
        "accuracy": len(correct_rows) / len(rows) if rows else 0.0,
        "single_test_original_correct": len(single_correct),
        "single_test_original_total": len(single_rows),
        "multi_test_original_subtests_correct": len(multi_correct),
        "multi_test_original_subtests_total": len(multi_rows),
        "correct_original_subtest_ids": [
            row.get("original_subtest_id") for row in correct_rows
        ],
        "rows": rows,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Separated score: {report['correct']}/{report['total']} = {report['accuracy']:.2%}")
    print(
        "Single-origin subtests: "
        f"{len(single_correct)}/{len(single_rows)}"
    )
    print(
        "Multi-origin subtests: "
        f"{len(multi_correct)}/{len(multi_rows)}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
