"""Build the 172-subtest separated eval set as a static Kaggle dataset.

Each (puzzle_id, test_index) becomes its own single-test puzzle with a fresh
random 8-hex-char ID — mimicking the 120-eval set's ID format so the
inference pipeline sees no structural difference between an "original
puzzle" and a "separated subtest". Train pairs and test input/output are
preserved bit-for-bit from the original; only the wrapper id changes.

ID generation is deterministic via a fixed seed for reproducibility.

Usage:
    python scripts/build_separated_eval.py
    python scripts/build_separated_eval.py --out path/to/output_dir
    python scripts/build_separated_eval.py --seed 2602
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "arc-prize-2026-arc-agi-2"
DEFAULT_OUT = ROOT / "data" / "arc-agi-2-separated-eval-172"


def build_separated(challenges, solutions, seed=2602):
    """Each (puzzle, test_index) becomes its own single-test puzzle with a
    fresh random 8-hex-char ID. Same train pairs + test grids + answers as
    the original puzzle — only the wrapper sid changes.

    Random IDs are seeded for reproducibility.
    """
    rng = random.Random(seed)
    used = set(challenges)
    sep_challenges = {}
    sep_solutions = {}
    mapping = {}

    def new_id():
        while True:
            sid = f"{rng.getrandbits(32):08x}"
            if sid not in used and sid not in sep_challenges:
                used.add(sid)
                return sid

    for pid in sorted(challenges):
        puzzle = challenges[pid]
        outs = solutions[pid]
        tests = puzzle["test"]
        if len(tests) != len(outs):
            raise ValueError(f"{pid}: test/solution length mismatch")
        for i, (test, out) in enumerate(zip(tests, outs)):
            sid = new_id()
            sep_challenges[sid] = {"train": puzzle["train"], "test": [test]}
            sep_solutions[sid] = [out]
            mapping[sid] = {
                "original_puzzle_id": pid,
                "original_test_index": i,
                "original_subtest_id": f"{pid}_{i}",
                "original_n_tests": len(tests),
                "was_multi_test": len(tests) > 1,
            }
    return sep_challenges, sep_solutions, mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=2602, help="random seed for ID generation")
    args = ap.parse_args()

    chal_path = args.src / "arc-agi_evaluation_challenges.json"
    sol_path = args.src / "arc-agi_evaluation_solutions.json"
    challenges = json.loads(chal_path.read_text())
    solutions = json.loads(sol_path.read_text())

    sep_chal, sep_sol, mapping = build_separated(challenges, solutions, seed=args.seed)
    n_subtests = sum(len(p["test"]) for p in challenges.values())
    assert len(sep_chal) == n_subtests, (
        f"separated count {len(sep_chal)} != reported subtests {n_subtests}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "arc-agi_evaluation_challenges_separated.json").write_text(
        json.dumps(sep_chal)
    )
    (args.out / "arc-agi_evaluation_solutions_separated.json").write_text(
        json.dumps(sep_sol)
    )
    (args.out / "arc-agi_evaluation_separated_mapping.json").write_text(
        json.dumps(mapping, indent=2)
    )

    multi = sum(1 for m in mapping.values() if m["was_multi_test"])
    single = sum(1 for m in mapping.values() if not m["was_multi_test"])
    print(f"Wrote {len(sep_chal)} separated puzzles to {args.out}")
    print(f"  from-multi: {multi}   from-single: {single}")

    # Show 36a08778 and 13e47133 mapping as sanity samples.
    for target in ("36a08778", "13e47133"):
        target_sids = sorted(
            sid for sid, m in mapping.items() if m["original_puzzle_id"] == target
        )
        print(f"  {target} -> {target_sids}")


if __name__ == "__main__":
    main()
