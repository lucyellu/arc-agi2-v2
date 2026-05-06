"""Merge per-chunk full-120-shape submission.json files into one consolidated
submission.json suitable for leaderboard.

Each input is the `submission.json` written by a chunked kernel run — full
120-shape with [[0]] placeholders for puzzles outside that chunk and real
attempts for the chunk's 4 puzzles. We overwrite placeholders with whichever
file has real data.

Usage:
    python tools/merge_submissions.py FILE [FILE ...] -o OUTPUT

Example:
    python tools/merge_submissions.py \
        results/results_subprobs-vanilla120-*/submission.json \
        -o results/submissions/latest/submission_vanilla120_merged.json
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PLACEHOLDER = [[0]]


def is_placeholder(grid):
    return grid is None or grid == PLACEHOLDER or grid == []


def is_real_entry(entry):
    if not isinstance(entry, dict):
        return False
    a1 = entry.get("attempt_1")
    a2 = entry.get("attempt_2")
    return not (is_placeholder(a1) and is_placeholder(a2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("-o", "--output", required=True, type=Path)
    args = ap.parse_args()

    merged = {}
    source_of = {}  # (puzzle, test_idx) -> file that contributed the real attempt
    conflicts = []

    for fp in args.files:
        sub = json.loads(fp.read_text())
        for puzzle_id, entries in sub.items():
            if puzzle_id not in merged:
                merged[puzzle_id] = list(entries)
                for i, entry in enumerate(entries):
                    if is_real_entry(entry):
                        source_of[(puzzle_id, i)] = fp.name
                continue

            existing_entries = merged[puzzle_id]
            for i, incoming in enumerate(entries):
                if i >= len(existing_entries):
                    existing_entries.append(incoming)
                    if is_real_entry(incoming):
                        source_of[(puzzle_id, i)] = fp.name
                    continue
                existing = existing_entries[i]
                if is_real_entry(incoming) and not is_real_entry(existing):
                    existing_entries[i] = incoming
                    source_of[(puzzle_id, i)] = fp.name
                elif is_real_entry(incoming) and is_real_entry(existing):
                    if existing != incoming:
                        conflicts.append({
                            "puzzle_id": puzzle_id,
                            "test_index": i,
                            "kept_from": source_of.get((puzzle_id, i), "<unknown>"),
                            "rejected_from": fp.name,
                        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged), encoding="utf-8")

    n_total = sum(len(v) for v in merged.values())
    n_real = sum(1 for v in merged.values() for e in v if is_real_entry(e))
    print(f"Merged {len(args.files)} files -> {args.output}")
    print(f"  puzzles: {len(merged)}")
    print(f"  subtest entries: {n_total}")
    print(f"  real attempts: {n_real} ({n_real / n_total:.1%})")
    print(f"  placeholders: {n_total - n_real}")

    if conflicts:
        print(f"\n{len(conflicts)} conflicts (kept first non-placeholder):")
        for c in conflicts[:10]:
            print(f"  {c['puzzle_id']}[{c['test_index']}]: kept from {c['kept_from']}, rejected {c['rejected_from']}")


if __name__ == "__main__":
    main()
