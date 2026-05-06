"""Construct notebooks/vanilla_chunked.ipynb: vanilla NVARC (grouped) over the
120-puzzle eval, but chunked the same way separated172-all is.

Each CHUNK_INDEX selects the same content as the corresponding separated chunk:
the 4 ORIGINAL puzzles whose subtests appear in the separated chunk's keys.
This gives us paired grouped/separated runs per chunk for direct comparison.

Reuses the working arc_loader/arc_decoder from the base. arc_solver gets
the defensive _resolve_model_path() patch so it finds sorokin OR our mirror
wherever Kaggle mounts them.
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _solver_patches import apply_path_override, apply_model_resolver, apply_early_stop
from _chunking import compute_chunk_mappings

# Multi-only: 49 multi-test originals → 13 chunks of 4. Single-test puzzles
# excluded because grouped == separated for n=1 (no isolation signal).
CHUNK_TO_ORIGINALS, CHUNK_TO_SEPARATED, N_CHUNKS = compute_chunk_mappings(
    chunk_size=4, multi_only=True
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks" / "kaggle-v1-n8-r64-a16-nvarc-32-64-base.ipynb"
OUT = ROOT / "notebooks" / "vanilla_chunked.ipynb"

base_nb = json.loads(BASE.read_text(encoding="utf-8"))
base_cells = base_nb["cells"]

loader_cell_src = "".join(base_cells[2]["source"])
decoder_cell_src = "".join(base_cells[3]["source"])
solver_src = "".join(base_cells[4]["source"])
assert loader_cell_src.startswith("%%writefile arc_loader.py")
assert decoder_cell_src.startswith("%%writefile arc_decoder.py")
assert solver_src.startswith("%%writefile arc_solver.py")

# Apply standard patches: env-overridable challenges path + defensive model
# resolver + EarlyStopOnLowLoss callback (loss-thresholded TTT).
solver_src = apply_path_override(solver_src)
solver_src = apply_model_resolver(solver_src)
solver_src = apply_early_stop(solver_src)

# starter.py — same env-driven version we use in separated172-all.
STARTER_SRC = """%%writefile starter.py
import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):

    os.environ[\"CUDA_VISIBLE_DEVICES\"] = str(rank)
    torch.set_default_device(\"cpu\")

    if rank > 0:
        while not os.path.exists(f\"/kaggle/worker{rank-1}\"):
            time.sleep(5)

    from arc_solver import worker

    with open(f\"/kaggle/worker{rank}\", \"w\") as f:
        f.write(\"Ok\")

    print(f\"[Rank {rank}] start!\")

    worker(rank, queue, end_time)

    print(f\"[Rank {rank}] done!\")


if __name__ == \"__main__\":

    parser = argparse.ArgumentParser()
    parser.add_argument(\"--end-time\", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv(\"KAGGLE_IS_COMPETITION_RERUN\")

    override = os.environ.get(\"AGI2_CHALLENGES_PATH\")
    if override:
        test_path = override
    elif rerun_mode:
        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\"
    else:
        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json\"

    with open(test_path, \"r\") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    env_keys = os.environ.get(\"AGI2_DEV_KEYS\", \"\").split()
    if env_keys:
        eval_keys = [k for k in env_keys if k in data]
        missing = [k for k in env_keys if k not in data]
        if missing:
            print(f\"WARNING: AGI2_DEV_KEYS missing from challenges: {missing}\")
    else:
        eval_keys = sorted(data.keys())

    print(f\"Queueing {len(eval_keys)} puzzles from {test_path}\")
    for key in eval_keys:
        queue.put(key)

    nprocs = int(os.environ.get(\"AGI2_NPROCS\", \"0\")) or min(4, max(1, torch.cuda.device_count()))
    print(f\"Using nprocs={nprocs}\")
    for _ in range(nprocs):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=nprocs)
"""

cell0_src = """import os
import time

# Vanilla NVARC (grouped) baseline, chunked the same way separated172-all is.
# Each CHUNK_INDEX runs the same 4 ORIGINAL puzzles whose subtests appear in
# the separated chunk's keys. Pair with the matching separated chunk for a
# direct grouped-vs-separated comparison.
CHUNK_SIZE = int(os.getenv(\"AGI2_CHUNK_SIZE\", \"4\"))
CHUNK_INDEX = int(os.getenv(\"AGI2_CHUNK_INDEX\", \"1\"))
RUN_HOURS = float(os.getenv(\"AGI2_RUN_HOURS\", \"1.5\"))

global_end_time = time.time() + RUN_HOURS * 3600 - 300
print(f\"Chunk index={CHUNK_INDEX} size={CHUNK_SIZE}  budget={RUN_HOURS:.2f}h  end_time={global_end_time:.0f}\")
"""

cell1_src = "!pip uninstall -y tensorflow"

cell_locate_src = f'''from pathlib import Path

# Pre-baked at build time from compute_chunk_mappings(multi_only=True).
# 13 chunks of 4 multi-test originals each. Vanilla grouped processes the
# whole original puzzle (all its subtests) per chunk entry — so a chunk
# of 4 multi-test originals (each n=2 typically) produces 8+ subtest
# results. Runtime needs no extra data attachments beyond the official
# 120-eval set.
N_CHUNKS_TOTAL = {N_CHUNKS}
CHUNK_TO_ORIGINALS = {json.dumps(CHUNK_TO_ORIGINALS, indent=4)}

assert 0 <= CHUNK_INDEX < N_CHUNKS_TOTAL, (
    f"CHUNK_INDEX={{CHUNK_INDEX}} out of range; valid 0..{{N_CHUNKS_TOTAL - 1}}"
)
key_lookup = str(CHUNK_INDEX) if str(CHUNK_INDEX) in CHUNK_TO_ORIGINALS else CHUNK_INDEX
CHUNK_ORIGINAL_KEYS = CHUNK_TO_ORIGINALS[key_lookup]
print(f"Multi-only chunk {{CHUNK_INDEX}}/{{N_CHUNKS_TOTAL - 1}}")
print(f"  {{len(CHUNK_ORIGINAL_KEYS)}} multi-test originals: {{CHUNK_ORIGINAL_KEYS}}")


def find_original_eval_files():
    candidates = [
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
        "/kaggle/input/arc-prize-2026-arc-agi-2",
        str(Path.cwd().parent / "data" / "arc-prize-2026-arc-agi-2"),
        str(Path.cwd() / "data" / "arc-prize-2026-arc-agi-2"),
    ]
    for base in candidates:
        chal = Path(base) / "arc-agi_evaluation_challenges.json"
        sol = Path(base) / "arc-agi_evaluation_solutions.json"
        if chal.exists() and sol.exists():
            return str(chal), str(sol)
    raise FileNotFoundError("original eval data missing")


ORIGINAL_CHALLENGES_PATH, ORIGINAL_SOLUTIONS_PATH = find_original_eval_files()
'''

cell_run_src = '''import os
import shutil
import subprocess
import sys
from pathlib import Path

INFER_DIR = Path("/kaggle/inference_outputs")
GROUPED_OUT_DIR = (
    Path("/kaggle/working/grouped_outputs")
    if Path("/kaggle/working").exists()
    else Path.cwd() / "grouped_outputs"
)
INFER_DIR.mkdir(parents=True, exist_ok=True)
GROUPED_OUT_DIR.mkdir(parents=True, exist_ok=True)
for f in list(INFER_DIR.iterdir()):
    if f.is_file():
        f.unlink()

phase_env = os.environ.copy()
# No AGI2_CHALLENGES_PATH override -> uses the original 120-eval challenges.
phase_env.pop("AGI2_CHALLENGES_PATH", None)
phase_env["AGI2_DEV_KEYS"] = " ".join(CHUNK_ORIGINAL_KEYS)
phase_env["UNSLOTH_DISABLE_STATISTICS"] = "1"
phase_env["OMP_NUM_THREADS"] = "12"
if Path("/usr/local/cuda/bin/ptxas").exists():
    phase_env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

# Use our mirror by default. Set AGI2_USE_SOROKIN_MODEL=1 to force sorokin's
# original mount instead, or set AGI2_MODEL_PATH=<...> for an explicit path.
mirror_root = "/kaggle/input/datasets/lucyellu/qwen3-4b-grids15-sft139-mirror"
if os.path.exists(os.path.join(mirror_root, "config.json")) and os.environ.get("AGI2_USE_SOROKIN_MODEL", "0") != "1":
    phase_env["AGI2_MODEL_PATH"] = mirror_root
    print(f"Using mirror model: {mirror_root}")
elif os.environ.get("AGI2_MODEL_PATH"):
    phase_env["AGI2_MODEL_PATH"] = os.environ["AGI2_MODEL_PATH"]
    print(f"Using AGI2_MODEL_PATH: {phase_env['AGI2_MODEL_PATH']}")
else:
    print("Falling back to default model resolver (will pick sorokin's mount or glob).")

print(f"=== Vanilla grouped chunk {CHUNK_INDEX} ({len(CHUNK_ORIGINAL_KEYS)} puzzles) ===")
sys.stdout.flush()
ret = subprocess.run(
    [sys.executable, "starter.py", "--end-time", str(global_end_time)],
    env=phase_env,
)
print(f"starter.py exited with code {ret.returncode}")

# Snapshot inference outputs.
for f in list(GROUPED_OUT_DIR.iterdir()):
    if f.is_file():
        f.unlink()
for f in INFER_DIR.iterdir():
    if f.is_file():
        shutil.copy(f, GROUPED_OUT_DIR / f.name)
print(f"Snapshotted: {len(list(GROUPED_OUT_DIR.iterdir()))} files in {GROUPED_OUT_DIR}")
'''

cell_report_src = '''import json
from pathlib import Path
import numpy as np
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

# Decode this chunk's grouped runs against the original eval set.
data = ArcDataset.from_file(ORIGINAL_CHALLENGES_PATH, keys=CHUNK_ORIGINAL_KEYS)
data = data.load_replies(ORIGINAL_SOLUTIONS_PATH)
decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
decoder.load_decoded_results(str(GROUPED_OUT_DIR))
sub = data.get_submission(decoder.run_selection_algo())

rows = []
for pid in CHUNK_ORIGINAL_KEYS:
    expected_list = data.replies.get(pid, [])
    attempts_list = sub.get(pid, [])
    for i, expected in enumerate(expected_list):
        attempt = attempts_list[i] if i < len(attempts_list) else {}
        a1 = attempt.get("attempt_1")
        a2 = attempt.get("attempt_2")
        correct_attempts = []
        for name, guess in (("attempt_1", a1), ("attempt_2", a2)):
            if guess is not None and np.array_equal(np.asarray(guess), np.asarray(expected)):
                correct_attempts.append(name)
        is_attempted = a1 not in (None, [[0]]) or a2 not in (None, [[0]])
        rows.append({
            "puzzle_id": pid,
            "test_index": i,
            "n_tests": len(expected_list),
            "is_multi_test": len(expected_list) > 1,
            "correct": bool(correct_attempts),
            "correct_attempts": correct_attempts,
            "attempted": is_attempted,
        })

n_correct = sum(r["correct"] for r in rows)
n_attempted = sum(r["attempted"] for r in rows)
n_multi = sum(1 for r in rows if r["is_multi_test"])
multi_correct = sum(1 for r in rows if r["is_multi_test"] and r["correct"])

out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
(out_dir / f"submission_grouped_chunk{CHUNK_INDEX}.json").write_text(json.dumps(sub))

report = {
    "chunk_index": CHUNK_INDEX,
    "chunk_size": CHUNK_SIZE,
    "chunk_original_keys": CHUNK_ORIGINAL_KEYS,
    "n_subtests": len(rows),
    "n_correct": n_correct,
    "n_attempted": n_attempted,
    "subtest_accuracy": n_correct / len(rows) if rows else 0.0,
    "multi_test_subtests": {
        "n_correct": multi_correct,
        "n_total": n_multi,
    },
    "rows": rows,
}
report_path = out_dir / f"vanilla_chunked_chunk{CHUNK_INDEX}_report.json"
report_path.write_text(json.dumps(report, indent=2))

print(f"\\n=== Vanilla chunk {CHUNK_INDEX} results ===")
print(f"Subtests correct: {n_correct}/{len(rows)}")
print(f"  multi-test subtests: {multi_correct}/{n_multi}")
print(f"  attempted (non-trivial): {n_attempted}/{len(rows)}")
print()
for r in rows:
    print(f"  {r['puzzle_id']} test[{r['test_index']}]: correct={r['correct']} via={r['correct_attempts']}")
print()
print(f"submission_grouped_chunk{CHUNK_INDEX}.json: {out_dir / ('submission_grouped_chunk' + str(CHUNK_INDEX) + '.json')}")
print(f"Report: {report_path}")
'''


def make_code_cell(src):
    if not src.endswith("\n"):
        src = src + "\n"
    lines = src.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


cells = [
    make_code_cell(cell0_src),
    make_code_cell(cell1_src),
    make_code_cell(loader_cell_src),
    make_code_cell(decoder_cell_src),
    make_code_cell(cell_locate_src),
    make_code_cell(solver_src),
    make_code_cell(STARTER_SRC),
    make_code_cell(cell_run_src),
    make_code_cell(cell_report_src),
]

nb = {
    "cells": cells,
    "metadata": base_nb.get("metadata", {}),
    "nbformat": base_nb.get("nbformat", 4),
    "nbformat_minor": base_nb.get("nbformat_minor", 5),
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells)")
