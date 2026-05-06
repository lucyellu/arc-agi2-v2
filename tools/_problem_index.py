"""Bidirectional map between `<original_pid>_<test_idx>` and human indices
(`1a`, `1b`, `5c`...) so analysis output is human-readable across the multi-test
isolation experiment.

Indexing scheme: 1-based numbering of the multi-test originals in chunk order
(chunks of 4, sorted alphabetically by puzzle id), with lowercase letters
appended for the per-puzzle subtest index. Chunk 0 -> 1..4, chunk 1 -> 5..8,
chunk 2 -> 9..12, etc.

Examples:
    13e47133_0 -> 1a    (chunk 0, puzzle 0, subtest 0)
    13e47133_1 -> 1b
    1ae2feb7_2 -> 5c    (chunk 1, puzzle 0, subtest 2 — n=3)
    20270e3b_1 -> 6b    (chunk 1, puzzle 1, subtest 1)

Usage as a library:
    from _problem_index import build_index
    idx = build_index()
    idx["subtest_to_human"]["20270e3b_1"]  # -> "6b"
    idx["human_to_subtest"]["6b"]          # -> "20270e3b_1"
    idx["pid_to_number"]["20270e3b"]       # -> 6

Usage as a CLI:
    python tools/_problem_index.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chunking import compute_chunk_mappings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHALLENGES = ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_challenges.json"

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def build_index(challenges_path=DEFAULT_CHALLENGES, chunk_size=4, multi_only=True):
    challenges = json.loads(Path(challenges_path).read_text())
    chunks_to_originals, _, _ = compute_chunk_mappings(chunk_size, multi_only)

    subtest_to_human = {}
    human_to_subtest = {}
    pid_to_number = {}
    number_to_pid = {}
    n = 1

    for chunk_idx in sorted(chunks_to_originals.keys()):
        for pid in chunks_to_originals[chunk_idx]:
            if pid not in challenges:
                raise KeyError(f"{pid} from chunking not present in {challenges_path}")
            pid_to_number[pid] = n
            number_to_pid[n] = pid
            n_tests = len(challenges[pid]["test"])
            if n_tests > len(LETTERS):
                raise ValueError(f"{pid} has {n_tests} subtests, exceeds letter alphabet")
            for i in range(n_tests):
                subtest_id = f"{pid}_{i}"
                human = f"{n}{LETTERS[i]}"
                subtest_to_human[subtest_id] = human
                human_to_subtest[human] = subtest_id
            n += 1

    return {
        "subtest_to_human": subtest_to_human,
        "human_to_subtest": human_to_subtest,
        "pid_to_number": pid_to_number,
        "number_to_pid": number_to_pid,
        "n_originals": n - 1,
    }


def main():
    idx = build_index()
    print(f"# {idx['n_originals']} multi-test originals indexed")
    print()
    print("# pid -> number (chunk-ordered)")
    for num in sorted(idx["number_to_pid"]):
        pid = idx["number_to_pid"][num]
        subs = sorted(h for h in idx["human_to_subtest"] if h.startswith(str(num)) and not h[len(str(num)):][0].isdigit())
        print(f"  {num:>2}  {pid}  -> subtests {subs}")


if __name__ == "__main__":
    main()
