"""Run the body of cell 4 against the local data dir and assert invariants."""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT / "data" / "arc-prize-2026-arc-agi-2"
chal = json.loads((data_dir / "arc-agi_evaluation_challenges.json").read_text())
sol = json.loads((data_dir / "arc-agi_evaluation_solutions.json").read_text())


def build_separated(challenges, solutions):
    sep_challenges = {}
    sep_solutions = {}
    mapping = {}
    used = set(challenges)
    for pid in sorted(challenges):
        puzzle = challenges[pid]
        outs = solutions[pid]
        tests = puzzle["test"]
        if len(tests) != len(outs):
            raise ValueError(f"{pid}: test/solution length mismatch")
        for i, (test, out) in enumerate(zip(tests, outs)):
            sid = pid[:7] + format(i, "x")
            suffix = 0
            while sid in used or sid in sep_challenges:
                suffix += 1
                sid = pid[:6] + format(i, "x") + format(suffix, "x")
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


sep_chal, sep_sol, mapping = build_separated(chal, sol)
n_subtests = sum(len(v["test"]) for v in chal.values())
print(f"original puzzles: {len(chal)}  reported subtests: {n_subtests}")
print(f"separated puzzles: {len(sep_chal)}  separated solutions: {len(sep_sol)}  mapping: {len(mapping)}")

assert len(chal) == 120
assert n_subtests == 172
assert len(sep_chal) == 172 == len(sep_sol) == len(mapping)

# all sep ids are 8-char hex, no underscores, distinct
import re

for sid in sep_chal:
    assert re.fullmatch(r"[0-9a-f]{8}", sid), sid
    assert "_" not in sid

# every sep puzzle has exactly one test
for sid, p in sep_chal.items():
    assert len(p["test"]) == 1
    assert len(sep_sol[sid]) == 1

# 36a08778 must be present
target = "36a08778"
target_sids = sorted(sid for sid, m in mapping.items() if m["original_puzzle_id"] == target)
print(f"target {target} -> {len(target_sids)} subtests -> {target_sids}")
assert target_sids, f"target {target} missing"
n_target = mapping[target_sids[0]]["original_n_tests"]
assert n_target == len(target_sids) == 2, f"expected 2 subtests for {target}, got {len(target_sids)}"

# verify the per-subtest content matches the original
for sid in target_sids:
    info = mapping[sid]
    idx = info["original_test_index"]
    assert sep_chal[sid]["test"] == [chal[target]["test"][idx]]
    assert sep_sol[sid] == [sol[target][idx]]

# breakdown of multi vs single
multi_origin = sum(1 for m in mapping.values() if m["was_multi_test"])
single_origin = sum(1 for m in mapping.values() if not m["was_multi_test"])
print(f"separated puzzles from multi-test originals: {multi_origin}")
print(f"separated puzzles from single-test originals: {single_origin}")
assert multi_origin + single_origin == 172

# show distribution of originals by n_tests
from collections import Counter

dist = Counter(len(p["test"]) for p in chal.values())
print(f"original n_tests distribution: {dict(dist)}")

print("OK — dataset construction is sound.")
