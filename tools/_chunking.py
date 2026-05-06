"""Shared chunking utility: compute the chunk -> originals mapping from
the 120-puzzle eval set. CHUNK_SIZE=4 means 4 ORIGINAL puzzles per chunk.

Both `vanilla-chunked` (grouped) and `separated172-all` (separated) chunk
on the SAME unit so chunk N covers the same content in both kernels.

Returns two structures:
- CHUNK_TO_ORIGINALS: {chunk_idx: [pid, pid, pid, pid]}
- CHUNK_TO_SEPARATED_KEYS: {chunk_idx: [sep_id, ...]} — the separated subtest
  ids for those 4 originals (any number from 4 to 12 depending on multi-test
  composition).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_challenges.json"
SEP_MAP_PATH = ROOT / "data" / "arc-agi-2-separated-eval-172" / "arc-agi_evaluation_separated_mapping.json"


def compute_chunk_mappings(chunk_size=4, multi_only=False):
    """Build chunk -> originals + chunk -> separated-keys mappings.

    Originals are sorted alphabetically (deterministic). Each chunk holds
    `chunk_size` unique originals. Separated keys for a chunk = all
    separated subtests whose `original_puzzle_id` is in that chunk's originals.

    multi_only=True: skip single-test originals. Reduces 120 originals to
    49 multi-test originals (172 → 101 separated subtests). 13 chunks of 4.
    Used when the experiment is the isolation hypothesis specifically —
    single-test puzzles contribute no signal because grouped == separated.
    """
    eval_challenges = json.loads(EVAL_PATH.read_text())
    sep_mapping = json.loads(SEP_MAP_PATH.read_text())

    if multi_only:
        # An original is "multi-test" if any of its separated subtests has the flag.
        multi_originals = {
            info["original_puzzle_id"]
            for info in sep_mapping.values()
            if info.get("was_multi_test")
        }
        all_originals = sorted(pid for pid in eval_challenges if pid in multi_originals)
    else:
        all_originals = sorted(eval_challenges)

    n_chunks = (len(all_originals) + chunk_size - 1) // chunk_size

    by_original = {}
    for sid, info in sep_mapping.items():
        by_original.setdefault(info["original_puzzle_id"], []).append(sid)
    for pid in by_original:
        by_original[pid].sort()

    chunk_to_originals = {}
    chunk_to_separated = {}
    for idx in range(n_chunks):
        originals = all_originals[idx * chunk_size : (idx + 1) * chunk_size]
        separated = []
        for pid in originals:
            separated.extend(by_original.get(pid, []))
        chunk_to_originals[idx] = originals
        chunk_to_separated[idx] = separated

    return chunk_to_originals, chunk_to_separated, n_chunks


if __name__ == "__main__":
    for label, multi_only in (("ALL", False), ("MULTI-ONLY", True)):
        print(f"\n=== {label} ===")
        co, cs, n = compute_chunk_mappings(multi_only=multi_only)
        total_originals = sum(len(v) for v in co.values())
        total_separated = sum(len(v) for v in cs.values())
        print(f"{n} chunks of 4 originals = {total_originals} originals, {total_separated} separated subtests")
        for i in range(min(3, n)):
            print(f"  chunk {i}: originals={co[i]}")
            print(f"            separated={cs[i]}")
