"""
Compare two run result directories to track LOO vs baseline progress.

Usage:
    python scripts/analyze_run.py <baseline_dir> <new_dir>

Each dir should contain stats_rank*.jsonl files (from the zip download).
Prints: solved/failed comparisons, timing changes, and regressions.

Example:
    python scripts/analyze_run.py results_2/results_17800s results_2/results_v11
"""

import json, sys, glob, os
from pathlib import Path


def load_stats(dirpath):
    records = {}
    for path in sorted(glob.glob(str(Path(dirpath) / "stats_rank*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records[rec["key"]] = rec
    if not records:
        # Try looking one level up for stats files
        for path in sorted(glob.glob(str(Path(dirpath) / "**" / "stats_rank*.jsonl"),
                                     recursive=True)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    records[rec["key"]] = rec
    return records


def load_score_from_stats(dirpath):
    """Try to read final score from submission_stats_*.json snapshots."""
    snaps = sorted(glob.glob(str(Path(dirpath) / "submission_stats_*.json")))
    if not snaps:
        snaps = sorted(glob.glob(str(Path(dirpath) / "**" / "submission_stats_*.json"),
                                 recursive=True))
    if not snaps:
        return None, {}
    with open(snaps[-1]) as f:
        snap = json.load(f)
    correct_map = {p["key"]: p.get("correct") for p in snap.get("per_puzzle", [])}
    return snap.get("score"), correct_map


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    base_dir, new_dir = sys.argv[1], sys.argv[2]
    base_stats = load_stats(base_dir)
    new_stats  = load_stats(new_dir)

    base_score, base_correct = load_score_from_stats(base_dir)
    new_score,  new_correct  = load_score_from_stats(new_dir)

    print(f"=== Baseline: {base_dir}  ({len(base_stats)} puzzles) ===")
    if base_score is not None:
        print(f"    Score: {base_score}/{len(base_stats)}")
    print(f"=== New:      {new_dir}  ({len(new_stats)} puzzles) ===")
    if new_score is not None:
        print(f"    Score: {new_score}/{len(new_stats)}")
    print()

    both = sorted(set(base_stats) & set(new_stats))
    only_base = sorted(set(base_stats) - set(new_stats))
    only_new  = sorted(set(new_stats) - set(base_stats))

    # Per-puzzle comparisons (correct/incorrect)
    regressions = []
    improvements = []
    for k in both:
        bc = base_correct.get(k)
        nc = new_correct.get(k)
        if bc is True and nc is False:
            regressions.append(k)
        elif bc is False and nc is True:
            improvements.append(k)

    if improvements:
        print(f"IMPROVED ({len(improvements)}) - newly solved:")
        for k in improvements:
            br = base_stats[k]; nr = new_stats[k]
            print(f"  {k}: base beams={br.get('beams',0)} | new beams={nr.get('beams',0)}"
                  f"  n_loo={nr.get('n_loo_ex',0)}  train={nr.get('t_train_s',0):.0f}s")

    if regressions:
        print(f"\nREGRESSED ({len(regressions)}) - now failing:")
        for k in regressions:
            br = base_stats[k]; nr = new_stats[k]
            print(f"  {k}: base beams={br.get('beams',0)} | new beams={nr.get('beams',0)}"
                  f"  train={nr.get('t_train_s',0):.0f}s")

    # Timing summary
    if both:
        def avg(recs, key):
            vs = [r.get(key, 0) for r in recs if r.get(key) is not None]
            return sum(vs) / len(vs) if vs else 0

        base_recs = [base_stats[k] for k in both]
        new_recs  = [new_stats[k]  for k in both]
        print(f"\nTiming (avg over {len(both)} shared puzzles):")
        print(f"  train:   base={avg(base_recs,'t_train_s'):.0f}s  "
              f"new={avg(new_recs,'t_train_s'):.0f}s")
        print(f"  dfs:     base={avg(base_recs,'t_dfs_s'):.0f}s    "
              f"new={avg(new_recs,'t_dfs_s'):.0f}s")
        print(f"  elapsed: base={avg(base_recs,'elapsed_s'):.0f}s  "
              f"new={avg(new_recs,'elapsed_s'):.0f}s")
        print(f"  beams:   base={avg(base_recs,'beams'):.1f}  "
              f"new={avg(new_recs,'beams'):.1f}")

        # LOO-specific stats
        loo_recs = [r for r in new_recs if r.get("n_loo_ex", 0) > 0]
        if loo_recs:
            print(f"\nLOO-activated puzzles: {len(loo_recs)}/{len(new_recs)}")
            print(f"  avg loo_ex={avg(loo_recs,'n_loo_ex'):.0f}  "
                  f"avg std_ex={avg(loo_recs,'n_std_ex'):.0f}")

    if only_base:
        print(f"\nOnly in baseline ({len(only_base)}): {only_base[:10]}")
    if only_new:
        print(f"\nOnly in new run ({len(only_new)}): {only_new[:10]}")

    # Puzzles with 0 beams in new run (candidates for improvement)
    zero_beams = [k for k in both if new_stats[k].get("beams", 0) == 0]
    if zero_beams:
        print(f"\nZero beams in new run ({len(zero_beams)}):")
        for k in zero_beams[:20]:
            r = new_stats[k]
            print(f"  {k}: N={r.get('N',0)} elapsed={r.get('elapsed_s',0):.0f}s")


if __name__ == "__main__":
    main()
