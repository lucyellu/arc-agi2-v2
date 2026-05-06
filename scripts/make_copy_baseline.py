#!/usr/bin/env python3
"""Create a valid ARC submission that copies each test input."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHALLENGES = ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_challenges.json"
DEFAULT_OUTPUT = ROOT / "results" / "submissions" / "latest" / "copy_eval_submission.json"


def make_submission(challenges):
    submission = {}
    for pid, puzzle in challenges.items():
        attempts = []
        for test in puzzle.get("test", []):
            grid = test.get("input", [[0]])
            attempts.append({
                "attempt_1": grid,
                "attempt_2": [[0]],
            })
        submission[pid] = attempts
    return submission


def main():
    parser = argparse.ArgumentParser(description="Build copy-input baseline submission")
    parser.add_argument("--challenges", default=str(DEFAULT_CHALLENGES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    challenges_path = Path(args.challenges)
    output_path = Path(args.output)
    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    submission = make_submission(challenges)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission), encoding="utf-8")
    n_tests = sum(len(v) for v in submission.values())
    print(f"Wrote {len(submission)} puzzles / {n_tests} test outputs to {output_path}")


if __name__ == "__main__":
    main()
