#!/usr/bin/env python3
"""Split ARC eval multi-test puzzles into single-test pseudo-puzzles.

The output format is normal ARC JSON:

- each new challenge has the original train examples
- each new challenge has exactly one test input
- each solution entry has exactly one output grid

This lets NVARC-style notebooks treat every eval subtest as an ordinary
single-test puzzle without custom subkey scheduling.
"""

import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "arc-prize-2026-arc-agi-2"
DEFAULT_CHALLENGES = DATA_DIR / "arc-agi_evaluation_challenges.json"
DEFAULT_SOLUTIONS = DATA_DIR / "arc-agi_evaluation_solutions.json"
DEFAULT_OUT_DIR = ROOT / "data" / "arc-prize-2026-arc-agi-2-separated"


def _new_id(rng, used):
    while True:
        value = f"{rng.getrandbits(32):08x}"
        if value not in used:
            used.add(value)
            return value


def split_eval(challenges, solutions, seed=2602):
    rng = random.Random(seed)
    used = set(challenges) | set(solutions)
    separated_challenges = {}
    separated_solutions = {}
    mapping = {}

    for original_id in sorted(challenges):
        puzzle = challenges[original_id]
        outputs = solutions[original_id]
        tests = puzzle.get("test", [])
        if len(tests) != len(outputs):
            raise ValueError(
                f"{original_id}: {len(tests)} test inputs but {len(outputs)} solutions"
            )

        for test_index, (test_item, solution_grid) in enumerate(zip(tests, outputs)):
            new_id = _new_id(rng, used)
            separated_challenges[new_id] = {
                "train": puzzle["train"],
                "test": [test_item],
            }
            separated_solutions[new_id] = [solution_grid]
            mapping[new_id] = {
                "original_puzzle_id": original_id,
                "original_test_index": test_index,
                "original_subtest_id": f"{original_id}_{test_index}",
                "original_num_tests": len(tests),
                "was_multi_test": len(tests) > 1,
            }

    return separated_challenges, separated_solutions, mapping


def make_sample_submission(challenges):
    return {
        pid: [{"attempt_1": [[0]], "attempt_2": [[0]]}]
        for pid in sorted(challenges)
    }


def main():
    parser = argparse.ArgumentParser(description="Create separated ARC eval JSON files")
    parser.add_argument("--challenges", default=str(DEFAULT_CHALLENGES))
    parser.add_argument("--solutions", default=str(DEFAULT_SOLUTIONS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=2602)
    args = parser.parse_args()

    challenges = json.loads(Path(args.challenges).read_text(encoding="utf-8"))
    solutions = json.loads(Path(args.solutions).read_text(encoding="utf-8"))
    separated_challenges, separated_solutions, mapping = split_eval(
        challenges, solutions, seed=args.seed
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arc-agi_evaluation_challenges_separated.json").write_text(
        json.dumps(separated_challenges), encoding="utf-8"
    )
    (out_dir / "arc-agi_evaluation_solutions_separated.json").write_text(
        json.dumps(separated_solutions), encoding="utf-8"
    )
    (out_dir / "arc-agi_evaluation_separated_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    (out_dir / "sample_submission_separated.json").write_text(
        json.dumps(make_sample_submission(separated_challenges)), encoding="utf-8"
    )

    multi_subtests = sum(1 for item in mapping.values() if item["was_multi_test"])
    print(f"Wrote {len(separated_challenges)} separated eval puzzles to {out_dir}")
    print(f"Original eval puzzles: {len(challenges)}")
    print(f"Separated subtests from multi-test originals: {multi_subtests}")


if __name__ == "__main__":
    main()
