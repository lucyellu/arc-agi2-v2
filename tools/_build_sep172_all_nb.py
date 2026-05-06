"""Construct notebooks/separated172_all.ipynb: run vanilla NVARC on all 172
separated puzzles (the entire isolation arm, no grouped comparison, no target
focus). Reuses the working arc_loader/arc_decoder/arc_solver/starter from the
base; the only added logic is a dataset-locator + a 172-wide
decode/recombine/score cell.
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _solver_patches import apply_path_override, apply_model_resolver, apply_early_stop
from _chunking import compute_chunk_mappings

# This notebook is multi-only: chunks 4 multi-test originals at a time and
# runs ALL their separated subtests (4-12 per chunk, was-multi-test only).
CHUNK_TO_ORIGINALS, CHUNK_TO_SEPARATED, N_CHUNKS = compute_chunk_mappings(
    chunk_size=4, multi_only=True
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks" / "kaggle-v1-n8-r64-a16-nvarc-32-64-base.ipynb"
OUT = ROOT / "notebooks" / "separated172_all.ipynb"

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

# starter.py — env-driven challenges path, env-driven dev keys (default = ALL),
# adaptive nprocs.
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

# Cell 0 — budget + chunk selection. CHUNK_SIZE is in ORIGINAL PUZZLES per
# chunk (each runs all its separated subtests). 30 chunks total to cover
# the 120-puzzle eval at 4 originals/chunk. A chunk has 4-12 separated
# subtests depending on multi-test composition.
cell0_src = """import os
import time

CHUNK_SIZE = int(os.getenv(\"AGI2_CHUNK_SIZE\", \"4\"))   # originals per chunk
CHUNK_INDEX = int(os.getenv(\"AGI2_CHUNK_INDEX\", \"0\"))
RUN_HOURS = float(os.getenv(\"AGI2_RUN_HOURS\", \"2.0\"))

global_end_time = time.time() + RUN_HOURS * 3600 - 300
print(f\"Chunk index={CHUNK_INDEX} size={CHUNK_SIZE} originals  budget={RUN_HOURS:.2f}h  end_time={global_end_time:.0f}\")
"""

cell1_src = "!pip uninstall -y tensorflow"

# Cell 4 — locate the separated dataset; no target picking.
cell_locate_src = '''import json
from pathlib import Path


def find_separated_files():
    chal_name = "arc-agi_evaluation_challenges_separated.json"
    sol_name = "arc-agi_evaluation_solutions_separated.json"
    map_name = "arc-agi_evaluation_separated_mapping.json"

    direct = [
        "/kaggle/input/arc-agi-2-separated-eval-172",
        "/kaggle/input/datasets/lucyellu/arc-agi-2-separated-eval-172",
        "/kaggle/input/lucyellu/arc-agi-2-separated-eval-172",
        str(Path.cwd().parent / "data" / "arc-agi-2-separated-eval-172"),
        str(Path.cwd() / "data" / "arc-agi-2-separated-eval-172"),
    ]
    for base in direct:
        bp = Path(base)
        if (bp / chal_name).exists() and (bp / sol_name).exists() and (bp / map_name).exists():
            print(f"Separated dataset (direct): {base}")
            return str(bp / chal_name), str(bp / sol_name), str(bp / map_name)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        hits = list(kaggle_input.rglob(chal_name))
        if hits:
            chal = hits[0]
            sol = chal.parent / sol_name
            mp = chal.parent / map_name
            if sol.exists() and mp.exists():
                print(f"Separated dataset (glob): {chal.parent}")
                return str(chal), str(sol), str(mp)

    raise FileNotFoundError(
        "Separated dataset not attached. Add the Kaggle dataset "
        "`lucyellu/arc-agi-2-separated-eval-172` to this kernel."
    )


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


SEPARATED_CHALLENGES_PATH, SEPARATED_SOLUTIONS_PATH, SEPARATED_MAPPING_PATH = find_separated_files()
ORIGINAL_CHALLENGES_PATH, ORIGINAL_SOLUTIONS_PATH = find_original_eval_files()

with open(SEPARATED_MAPPING_PATH) as f:
    sep_mapping = json.load(f)
assert len(sep_mapping) == 172, f"separated dataset should have 172 entries, got {len(sep_mapping)}"

# Chunk -> originals + chunk -> separated-keys mappings baked at notebook
# BUILD time (multi-only: 49 multi-test originals, 13 chunks of 4).
N_CHUNKS_TOTAL = ''' + str(N_CHUNKS) + '''
CHUNK_TO_ORIGINALS = ''' + json.dumps(CHUNK_TO_ORIGINALS, indent=4) + '''
CHUNK_TO_SEPARATED = ''' + json.dumps(CHUNK_TO_SEPARATED, indent=4) + '''

assert 0 <= CHUNK_INDEX < N_CHUNKS_TOTAL, (
    f"CHUNK_INDEX={CHUNK_INDEX} out of range; valid 0..{N_CHUNKS_TOTAL - 1}"
)
# JSON dict keys come back as strings; handle both int and str index lookup.
key_lookup = str(CHUNK_INDEX) if str(CHUNK_INDEX) in CHUNK_TO_ORIGINALS else CHUNK_INDEX
CHUNK_ORIGINAL_KEYS = CHUNK_TO_ORIGINALS[key_lookup]
CHUNK_KEYS = CHUNK_TO_SEPARATED[key_lookup]
print(f"Multi-only chunk {CHUNK_INDEX}/{N_CHUNKS_TOTAL - 1}")
print(f"  {len(CHUNK_ORIGINAL_KEYS)} original puzzles: {CHUNK_ORIGINAL_KEYS}")
print(f"  {len(CHUNK_KEYS)} separated subtests to run:")
for sid in CHUNK_KEYS:
    info = sep_mapping[sid]
    print(f"    {sid} <- {info['original_subtest_id']}")
'''

# Cell 7 — run inference once over all 172 separated puzzles.
cell_run_src = '''import os
import shutil
import subprocess
import sys
from pathlib import Path

INFER_DIR = Path("/kaggle/inference_outputs")
SEPARATED_OUT_DIR = (
    Path("/kaggle/working/separated_outputs")
    if Path("/kaggle/working").exists()
    else Path.cwd() / "separated_outputs"
)
INFER_DIR.mkdir(parents=True, exist_ok=True)
SEPARATED_OUT_DIR.mkdir(parents=True, exist_ok=True)
for f in list(INFER_DIR.iterdir()):
    if f.is_file():
        f.unlink()

phase_env = os.environ.copy()
phase_env["AGI2_CHALLENGES_PATH"] = SEPARATED_CHALLENGES_PATH
# Queue only this chunk.
phase_env["AGI2_DEV_KEYS"] = " ".join(CHUNK_KEYS)
phase_env["UNSLOTH_DISABLE_STATISTICS"] = "1"
phase_env["OMP_NUM_THREADS"] = "12"
if Path("/usr/local/cuda/bin/ptxas").exists():
    phase_env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

# Use our mirror by default (matches vanilla_chunked). Set
# AGI2_USE_SOROKIN_MODEL=1 to force sorokin's mount, or AGI2_MODEL_PATH=<...>
# for an explicit path.
mirror_root = "/kaggle/input/datasets/lucyellu/qwen3-4b-grids15-sft139-mirror"
if os.path.exists(os.path.join(mirror_root, "config.json")) and os.environ.get("AGI2_USE_SOROKIN_MODEL", "0") != "1":
    phase_env["AGI2_MODEL_PATH"] = mirror_root
    print(f"Using mirror model: {mirror_root}")
elif os.environ.get("AGI2_MODEL_PATH"):
    phase_env["AGI2_MODEL_PATH"] = os.environ["AGI2_MODEL_PATH"]
    print(f"Using AGI2_MODEL_PATH: {phase_env['AGI2_MODEL_PATH']}")
else:
    print("Falling back to default model resolver (will pick sorokin's mount or glob).")

print(f"=== Separated chunk {CHUNK_INDEX} ({len(CHUNK_KEYS)} puzzles) ===")
sys.stdout.flush()
ret = subprocess.run(
    [sys.executable, "starter.py", "--end-time", str(global_end_time)],
    env=phase_env,
)
print(f"starter.py exited with code {ret.returncode}")

# Snapshot inference outputs to a stable dir for the decode cell.
for f in list(SEPARATED_OUT_DIR.iterdir()):
    if f.is_file():
        f.unlink()
for f in INFER_DIR.iterdir():
    if f.is_file():
        shutil.copy(f, SEPARATED_OUT_DIR / f.name)
print(f"Snapshotted: {len(list(SEPARATED_OUT_DIR.iterdir()))} files in {SEPARATED_OUT_DIR}")
'''

# Cell 8 — decode all 172 separated outputs, score, recombine.
cell_report_src = '''import json
from pathlib import Path
import numpy as np
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

mapping = json.loads(Path(SEPARATED_MAPPING_PATH).read_text())

# Decode only the puzzles in this chunk.
data = ArcDataset.from_file(SEPARATED_CHALLENGES_PATH, keys=CHUNK_KEYS)
data = data.load_replies(SEPARATED_SOLUTIONS_PATH)
decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
decoder.load_decoded_results(str(SEPARATED_OUT_DIR))
sep_submission = data.get_submission(decoder.run_selection_algo())

# Per-subtest correctness against the separated solutions.
rows = []
n_correct = 0
n_attempted = 0
multi_correct = 0
multi_total = 0
single_correct = 0
single_total = 0
for sid in CHUNK_KEYS:
    info = mapping[sid]
    expected = data.replies.get(sid, [])
    expected_grid = expected[0] if expected else None
    attempt_block = sep_submission.get(sid, [{}])
    attempt = attempt_block[0] if attempt_block else {}
    a1 = attempt.get("attempt_1")
    a2 = attempt.get("attempt_2")
    correct_attempts = []
    if expected_grid is not None:
        for name, guess in (("attempt_1", a1), ("attempt_2", a2)):
            if guess is not None and np.array_equal(np.asarray(guess), np.asarray(expected_grid)):
                correct_attempts.append(name)
    is_correct = bool(correct_attempts)
    is_attempted = a1 not in (None, [[0]]) or a2 not in (None, [[0]])
    rows.append({
        "separated_id": sid,
        "original_puzzle_id": info["original_puzzle_id"],
        "original_subtest_id": info["original_subtest_id"],
        "was_multi_test": info["was_multi_test"],
        "correct": is_correct,
        "correct_attempts": correct_attempts,
        "attempted": is_attempted,
    })
    n_correct += int(is_correct)
    n_attempted += int(is_attempted)
    if info["was_multi_test"]:
        multi_total += 1
        if is_correct:
            multi_correct += 1
    else:
        single_total += 1
        if is_correct:
            single_correct += 1

# Recombine separated attempts into the original 120-puzzle eval submission shape.
with open(ORIGINAL_CHALLENGES_PATH) as f:
    original_challenges = json.load(f)
recombined = {}
for pid, puzzle in original_challenges.items():
    n_tests = len(puzzle["test"])
    recombined[pid] = [
        {"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(n_tests)
    ]
for sid, attempts in sep_submission.items():
    info = mapping[sid]
    pid = info["original_puzzle_id"]
    idx = info["original_test_index"]
    if attempts:
        recombined[pid][idx] = attempts[0]

out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
(out_dir / f"submission_chunk{CHUNK_INDEX}.json").write_text(json.dumps(recombined))
(out_dir / f"submission_separated_chunk{CHUNK_INDEX}.json").write_text(json.dumps(sep_submission))

report = {
    "chunk_index": CHUNK_INDEX,
    "chunk_size": CHUNK_SIZE,
    "chunk_keys": CHUNK_KEYS,
    "n_separated_puzzles": len(rows),
    "n_correct_subtests": n_correct,
    "n_attempted_subtests": n_attempted,
    "subtest_accuracy": n_correct / len(rows) if rows else 0.0,
    "from_multi_test": {
        "n_correct": multi_correct,
        "n_total": multi_total,
        "accuracy": multi_correct / multi_total if multi_total else 0.0,
    },
    "from_single_test": {
        "n_correct": single_correct,
        "n_total": single_total,
        "accuracy": single_correct / single_total if single_total else 0.0,
    },
    "rows": rows,
}
report_path = out_dir / f"separated172_chunk{CHUNK_INDEX}_report.json"
report_path.write_text(json.dumps(report, indent=2))

print(f"\\n=== Chunk {CHUNK_INDEX} results ===")
print(f"Separated subtests correct: {n_correct}/{len(rows)}")
print(f"  from multi-test originals:  {multi_correct}/{multi_total}")
print(f"  from single-test originals: {single_correct}/{single_total}")
print(f"  attempted (non-trivial):    {n_attempted}/{len(rows)}")
print()
print(f"submission_chunk{CHUNK_INDEX}.json: {out_dir / ('submission_chunk' + str(CHUNK_INDEX) + '.json')}")
print(f"submission_separated_chunk{CHUNK_INDEX}.json: {out_dir / ('submission_separated_chunk' + str(CHUNK_INDEX) + '.json')}")
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
