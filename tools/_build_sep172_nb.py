"""Construct notebooks/separated172_36a08778.ipynb from the working base.

Only the cells that need to change are rewritten in this script. The two
verbatim cells (arc_loader.py and arc_decoder.py) are pulled directly out of
the base notebook so they stay byte-identical.
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks" / "kaggle-v1-n8-r64-a16-nvarc-32-64-base.ipynb"
OUT = ROOT / "notebooks" / "separated172_36a08778.ipynb"

base_nb = json.loads(BASE.read_text(encoding="utf-8"))
base_cells = base_nb["cells"]

# Lift loader/decoder cells verbatim — these are unchanged.
loader_cell_src = "".join(base_cells[2]["source"])
decoder_cell_src = "".join(base_cells[3]["source"])
assert loader_cell_src.startswith("%%writefile arc_loader.py"), "base layout drifted"
assert decoder_cell_src.startswith("%%writefile arc_decoder.py"), "base layout drifted"

# Lift the base arc_solver.py text and rewrite ONLY the path resolution.
solver_src = "".join(base_cells[4]["source"])
assert solver_src.startswith("%%writefile arc_solver.py"), "base layout drifted"

OLD_BLOCK = (
    "    if rerun_mode:\n"
    "        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\"\n"
    "    else:\n"
    "        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json\"\n"
)
NEW_BLOCK = (
    "    override = os.environ.get(\"AGI2_CHALLENGES_PATH\")\n"
    "    if override:\n"
    "        test_path = override\n"
    "    elif rerun_mode:\n"
    "        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\"\n"
    "    else:\n"
    "        test_path = \"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json\"\n"
)
assert OLD_BLOCK in solver_src, "could not locate solver test_path block"
solver_src = solver_src.replace(OLD_BLOCK, NEW_BLOCK)

# Defensive model-path lookup. Kaggle mounts model_sources at unpredictable
# paths; we try the canonical one first, then glob for any directory under
# /kaggle/input/ containing the qwen3 model's config.json. This mirrors the
# dataset finder.
OLD_MODEL_LOAD = (
    "    model, tokenizer = FastLanguageModel.from_pretrained(\n"
    "        model_name=\"/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",\n"
)
NEW_MODEL_LOAD = (
    "    model_name = _resolve_model_path()\n"
    "    print(f\"[Rank {rank}] resolved model_name -> {model_name}\", flush=True)\n"
    "    model, tokenizer = FastLanguageModel.from_pretrained(\n"
    "        model_name=model_name,\n"
)
assert OLD_MODEL_LOAD in solver_src, "could not locate solver model_load block"
solver_src = solver_src.replace(OLD_MODEL_LOAD, NEW_MODEL_LOAD)

# Inject _resolve_model_path() near the top of arc_solver.py.
RESOLVER_FN = """

def _resolve_model_path():
    \"\"\"Find the qwen3_4b model wherever Kaggle mounted it.

    Kaggle mounts model_sources inconsistently — sometimes at the simple
    /kaggle/input/<slug>/... path, sometimes nested under /kaggle/input/
    models/<owner>/<slug>/<framework>/<variant>/<version>/. We try the
    well-known paths first, then glob /kaggle/input/ for any qwen3-named
    directory containing a config.json.
    \"\"\"
    override = os.environ.get(\"AGI2_MODEL_PATH\")
    if override and os.path.exists(os.path.join(override, \"config.json\")):
        print(f\"[model] using AGI2_MODEL_PATH override: {override}\", flush=True)
        return override

    direct = [
        \"/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",
        \"/kaggle/input/datasets/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",
        \"/kaggle/input/datasets/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1\",
        \"/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",
        \"/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1\",
        \"/kaggle/input/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1\",
    ]
    for path in direct:
        if os.path.exists(os.path.join(path, \"config.json\")):
            print(f\"[model] direct hit: {path}\", flush=True)
            return path

    # Glob fallback. Walk /kaggle/input/ for qwen3-named dirs with a config.
    if os.path.isdir(\"/kaggle/input\"):
        from pathlib import Path as _P
        for cfg in _P(\"/kaggle/input\").rglob(\"config.json\"):
            cfg_str = str(cfg)
            if \"qwen3_4b_grids15_sft139\" in cfg_str:
                model_dir = str(cfg.parent)
                print(f\"[model] glob hit: {model_dir}\", flush=True)
                return model_dir
        # Diagnostics: list what IS under /kaggle/input/ so we can see why
        # the model isn't found.
        try:
            entries = sorted(os.listdir(\"/kaggle/input\"))
        except OSError:
            entries = []
        print(f\"[model] /kaggle/input/ entries: {entries}\", flush=True)
        for e in entries:
            sub = os.path.join(\"/kaggle/input\", e)
            try:
                print(f\"[model]   {e}/ -> {sorted(os.listdir(sub))[:8]}\", flush=True)
            except OSError:
                pass

    raise RuntimeError(
        \"Could not locate qwen3_4b_grids15_sft139 model under /kaggle/input/. \"
        \"Attach the model to this kernel via the web UI (Add Input -> Models -> \"
        \"sorokin/qwen3_4b_grids15_sft139, version Transformers/bfloat16/1).\"
    )

"""
# Place the resolver after the initial imports/constants block, right before
# the first `class` or `def`. Insert after the `EOS_ID = 15` line which is a
# stable marker in the base solver.
MARKER = "EOS_ID = 15\n"
assert MARKER in solver_src, "could not locate EOS_ID marker for resolver insert"
solver_src = solver_src.replace(MARKER, MARKER + RESOLVER_FN, 1)

# Lift the base starter.py text and rewrite to support env-var overrides + adaptive nprocs.
NEW_STARTER = """%%writefile starter.py
import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)

    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")

    worker(rank, queue, end_time)

    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    override = os.environ.get("AGI2_CHALLENGES_PATH")
    if override:
        test_path = override
    elif rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    if rerun_mode:
        eval_keys = sorted(data.keys())
    else:
        env_keys = os.environ.get("AGI2_DEV_KEYS", "").split()
        if env_keys:
            eval_keys = [k for k in env_keys if k in data]
            missing = [k for k in env_keys if k not in data]
            if missing:
                print(f"WARNING: AGI2_DEV_KEYS missing from challenges: {missing}")
        else:
            default_keys = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]
            eval_keys = [k for k in default_keys if k in data]

    print(f"Queueing {len(eval_keys)} puzzles from {test_path}")
    for key in eval_keys:
        queue.put(key)

    nprocs = int(os.environ.get("AGI2_NPROCS", "0")) or min(4, max(1, torch.cuda.device_count()))
    print(f"Using nprocs={nprocs}")
    for _ in range(nprocs):
        queue.put(None)

    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=nprocs)
"""

# Cell 0 — budget + experiment config.
cell0_src = """import os
import time

# Two-phase experiment on a single multi-test puzzle:
#   Phase A (grouped): run 36a08778 from the original 120-puzzle eval set,
#       so both subtests share one TTT fine-tune (= vanilla NVARC behavior).
#   Phase B (separated): run the same puzzle's 2 subtests as 2 standalone
#       single-test puzzles in our 172-puzzle set, each with its own TTT.
# The pair gives a head-to-head signal even with no checked-in baseline.
RUN_HOURS = float(os.getenv("AGI2_RUN_HOURS", "1.5"))
TARGET_BASE_ID = os.getenv("AGI2_TARGET_BASE_ID", "36a08778")

global_end_time = time.time() + RUN_HOURS * 3600 - 300
print(f"Budget: {RUN_HOURS:.2f}h  end_time={global_end_time:.0f}")
print(f"Target multi-test puzzle: {TARGET_BASE_ID}")
"""

# Cell 1 — same pip line as base.
cell1_src = "!pip uninstall -y tensorflow"

# Cell — locate the pre-built 172-puzzle separated dataset (Kaggle dataset
# `lucyellu/arc-agi-2-separated-eval-172`, attached as kernel input) and
# pick the subtests for our target puzzle. No data construction at runtime.
cell5_src = '''import json
import os
from pathlib import Path


def find_separated_files():
    """Locate the separated dataset wherever Kaggle decided to mount it.

    Try the well-known direct paths first, then fall back to a glob walk
    under /kaggle/input/ so we don't have to guess Kaggle's mount layout
    (it sometimes lands at /kaggle/input/<slug>/, sometimes at
    /kaggle/input/datasets/<owner>/<slug>/).
    """
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

    # Glob fallback — searches anywhere under /kaggle/input/.
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
        f"Separated dataset not found. Attach the Kaggle dataset "
        f"`lucyellu/arc-agi-2-separated-eval-172` to this kernel. "
        f"Tried: {direct} and globbed /kaggle/input/."
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


SEPARATED_CHALLENGES_PATH, SEPARATED_SOLUTIONS_PATH, SEPARATED_MAPPING_PATH = (
    find_separated_files()
)
ORIGINAL_CHALLENGES_PATH, ORIGINAL_SOLUTIONS_PATH = find_original_eval_files()

with open(SEPARATED_MAPPING_PATH) as f:
    sep_mapping = json.load(f)

# Sanity-check the dataset shape.
assert len(sep_mapping) == 172, (
    f"separated dataset should have 172 entries, got {len(sep_mapping)}"
)

# Pick the separated subtests that came from TARGET_BASE_ID.
TARGET_SEPARATED_IDS = sorted(
    sid for sid, m in sep_mapping.items() if m["original_puzzle_id"] == TARGET_BASE_ID
)
assert TARGET_SEPARATED_IDS, f"target {TARGET_BASE_ID} not found in separated mapping"
n_target = sep_mapping[TARGET_SEPARATED_IDS[0]]["original_n_tests"]
print(f"Target {TARGET_BASE_ID} -> {n_target} subtest(s) -> separated ids: {TARGET_SEPARATED_IDS}")
'''

# Per-phase runners: two cells use this structure — Python config, then a
# !-shell call to starter.py, then a Python snapshot. Both phases write
# inference outputs to /kaggle/inference_outputs/ (hardcoded in arc_solver.py),
# so we snapshot to phase-specific dirs between runs.

# Per-phase runners — call starter.py via subprocess so we don't have to
# rely on IPython's `!cmd {var}` interpolation, which doesn't fire reliably
# on Kaggle/papermill when the shell line is mixed with surrounding Python.

cell_grouped_src = '''import os
import shutil
import subprocess
import sys
from pathlib import Path

INFER_DIR = Path("/kaggle/inference_outputs")
GROUPED_OUT_DIR = Path("/kaggle/working/grouped_outputs") if Path("/kaggle/working").exists() else Path.cwd() / "grouped_outputs"
SEPARATED_OUT_DIR = Path("/kaggle/working/separated_outputs") if Path("/kaggle/working").exists() else Path.cwd() / "separated_outputs"
INFER_DIR.mkdir(parents=True, exist_ok=True)
GROUPED_OUT_DIR.mkdir(parents=True, exist_ok=True)
SEPARATED_OUT_DIR.mkdir(parents=True, exist_ok=True)
for f in list(INFER_DIR.iterdir()):
    if f.is_file():
        f.unlink()

# Phase A: grouped run on the original 120-puzzle eval. The solver/starter
# fall through to /kaggle/input/... when AGI2_CHALLENGES_PATH is unset.
phase_env = os.environ.copy()
phase_env.pop("AGI2_CHALLENGES_PATH", None)
phase_env["AGI2_DEV_KEYS"] = TARGET_BASE_ID
phase_env["UNSLOTH_DISABLE_STATISTICS"] = "1"
phase_env["OMP_NUM_THREADS"] = "12"
if Path("/usr/local/cuda/bin/ptxas").exists():
    phase_env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

print("=== Phase A (grouped baseline) ===")
print(f"Running {TARGET_BASE_ID} from original 120-puzzle eval")
sys.stdout.flush()
ret = subprocess.run(
    [sys.executable, "starter.py", "--end-time", str(global_end_time)],
    env=phase_env,
)
print(f"Phase A starter.py exited with code {ret.returncode}")

# Snapshot grouped outputs so they survive the next run's clear-step.
for f in list(GROUPED_OUT_DIR.iterdir()):
    if f.is_file():
        f.unlink()
for f in INFER_DIR.iterdir():
    if f.is_file():
        shutil.copy(f, GROUPED_OUT_DIR / f.name)
print(f"Phase A snapshotted: {len(list(GROUPED_OUT_DIR.iterdir()))} files in {GROUPED_OUT_DIR}")
'''

cell_separated_src = '''import os
import shutil
import subprocess
import sys
from pathlib import Path

INFER_DIR = Path("/kaggle/inference_outputs")
SEPARATED_OUT_DIR = Path("/kaggle/working/separated_outputs") if Path("/kaggle/working").exists() else Path.cwd() / "separated_outputs"
SEPARATED_OUT_DIR.mkdir(parents=True, exist_ok=True)
for f in list(INFER_DIR.iterdir()):
    if f.is_file():
        f.unlink()

# Phase B: separated run — point the solver at the 172-puzzle separated
# challenges file and queue only the subtests for our target puzzle.
phase_env = os.environ.copy()
phase_env["AGI2_CHALLENGES_PATH"] = SEPARATED_CHALLENGES_PATH
phase_env["AGI2_DEV_KEYS"] = " ".join(TARGET_SEPARATED_IDS)
phase_env["UNSLOTH_DISABLE_STATISTICS"] = "1"
phase_env["OMP_NUM_THREADS"] = "12"
if Path("/usr/local/cuda/bin/ptxas").exists():
    phase_env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

print("=== Phase B (separated) ===")
print(f"Running {TARGET_SEPARATED_IDS} from 172-puzzle separated eval")
sys.stdout.flush()
ret = subprocess.run(
    [sys.executable, "starter.py", "--end-time", str(global_end_time)],
    env=phase_env,
)
print(f"Phase B starter.py exited with code {ret.returncode}")

for f in list(SEPARATED_OUT_DIR.iterdir()):
    if f.is_file():
        f.unlink()
for f in INFER_DIR.iterdir():
    if f.is_file():
        shutil.copy(f, SEPARATED_OUT_DIR / f.name)
print(f"Phase B snapshotted: {len(list(SEPARATED_OUT_DIR.iterdir()))} files in {SEPARATED_OUT_DIR}")
'''

# Final cell: decode both phases, score, write submission files.
cell_report_src = '''import json
from pathlib import Path
import numpy as np
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder


def per_subtest_correctness(data, submission, expected_lookup):
    """Return per-subtest [{id, correct, correct_attempts}] rows."""
    rows = []
    for key in data.keys:
        expected_list = expected_lookup(key)
        attempts_list = submission.get(key, [])
        for i, expected in enumerate(expected_list):
            attempt = attempts_list[i] if i < len(attempts_list) else {}
            a1 = attempt.get("attempt_1")
            a2 = attempt.get("attempt_2")
            correct_attempts = []
            for name, guess in (("attempt_1", a1), ("attempt_2", a2)):
                if guess is not None and np.array_equal(np.asarray(guess), np.asarray(expected)):
                    correct_attempts.append(name)
            rows.append({
                "key": key,
                "test_index": i,
                "correct": bool(correct_attempts),
                "correct_attempts": correct_attempts,
            })
    return rows


# ---- Phase A: decode grouped outputs against the ORIGINAL eval set. ----
grouped_data = ArcDataset.from_file(ORIGINAL_CHALLENGES_PATH, keys=[TARGET_BASE_ID])
grouped_data = grouped_data.load_replies(ORIGINAL_SOLUTIONS_PATH)
grouped_decoder = ArcDecoder(grouped_data.split_multi_replies(), n_guesses=2)
grouped_decoder.load_decoded_results(str(GROUPED_OUT_DIR))
grouped_submission = grouped_data.get_submission(grouped_decoder.run_selection_algo())
grouped_rows = per_subtest_correctness(
    grouped_data,
    grouped_submission,
    expected_lookup=lambda k: grouped_data.replies.get(k, []),
)

# ---- Phase B: decode separated outputs against the SEPARATED eval set. ----
mapping = json.loads(Path(SEPARATED_MAPPING_PATH).read_text())
sep_data = ArcDataset.from_file(SEPARATED_CHALLENGES_PATH, keys=TARGET_SEPARATED_IDS)
sep_data = sep_data.load_replies(SEPARATED_SOLUTIONS_PATH)
sep_decoder = ArcDecoder(sep_data.split_multi_replies(), n_guesses=2)
sep_decoder.load_decoded_results(str(SEPARATED_OUT_DIR))
sep_submission = sep_data.get_submission(sep_decoder.run_selection_algo())
sep_rows = per_subtest_correctness(
    sep_data,
    sep_submission,
    expected_lookup=lambda k: sep_data.replies.get(k, []),
)

# ---- Recombine separated submission back into original 36a08778 shape. ----
recombined_target = [
    {"attempt_1": [[0]], "attempt_2": [[0]]}
    for _ in range(mapping[TARGET_SEPARATED_IDS[0]]["original_n_tests"])
]
for sid in TARGET_SEPARATED_IDS:
    info = mapping[sid]
    idx = info["original_test_index"]
    attempts = sep_submission.get(sid, [])
    if attempts:
        recombined_target[idx] = attempts[0]

# ---- Build a correctly-formatted full-eval submission.json. -------------
# All 120 eval puzzles, with our target slot filled from the separated run
# and every other slot a placeholder. This is the shape Kaggle's scorer expects.
with open(ORIGINAL_CHALLENGES_PATH) as f:
    original_challenges = json.load(f)
full_submission = {}
for pid, puzzle in original_challenges.items():
    n_tests = len(puzzle["test"])
    if pid == TARGET_BASE_ID:
        full_submission[pid] = recombined_target
    else:
        full_submission[pid] = [
            {"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(n_tests)
        ]

# ---- Write all the output files we want to be able to inspect. -----------
out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()

# 1) submission.json — Kaggle-shaped 120-puzzle eval submission. This is the
#    "correct" file. The separated-run results land inside the 36a08778 slot;
#    every other puzzle is a placeholder.
submission_path = out_dir / "submission.json"
submission_path.write_text(json.dumps(full_submission))

# 2) submission_separated.json — keys are the SEPARATED ids (e.g. 36a08770).
#    Just for our inspection — not a valid Kaggle submission shape.
submission_separated_path = out_dir / "submission_separated.json"
submission_separated_path.write_text(json.dumps(sep_submission))

# 3) submission_grouped.json — Phase-A grouped submission, keyed by the
#    original puzzle id. One puzzle, original eval shape.
submission_grouped_path = out_dir / "submission_grouped.json"
submission_grouped_path.write_text(json.dumps(grouped_submission))

# ---- Side-by-side report. ------------------------------------------------
def fmt_row(r):
    return f"test[{r['test_index']}]: correct={r['correct']} via={r['correct_attempts']}"

n_grouped_correct = sum(r["correct"] for r in grouped_rows)
n_sep_correct = sum(r["correct"] for r in sep_rows)
report = {
    "target_base_id": TARGET_BASE_ID,
    "separated_ids": TARGET_SEPARATED_IDS,
    "phase_a_grouped": {
        "n_correct": n_grouped_correct,
        "n_total": len(grouped_rows),
        "rows": grouped_rows,
    },
    "phase_b_separated": {
        "n_correct": n_sep_correct,
        "n_total": len(sep_rows),
        "rows": sep_rows,
    },
    "diff_subtests_only_separated": [
        i for i, (g, s) in enumerate(zip(grouped_rows, sep_rows))
        if s["correct"] and not g["correct"]
    ],
    "diff_subtests_only_grouped": [
        i for i, (g, s) in enumerate(zip(grouped_rows, sep_rows))
        if g["correct"] and not s["correct"]
    ],
}
report_path = out_dir / f"separated_vs_grouped_{TARGET_BASE_ID}.json"
report_path.write_text(json.dumps(report, indent=2))

print()
print(f"=== Results for {TARGET_BASE_ID} ===")
print(f"Phase A (grouped):    {n_grouped_correct}/{len(grouped_rows)}")
for r in grouped_rows:
    print(f"  grouped   {fmt_row(r)}")
print(f"Phase B (separated):  {n_sep_correct}/{len(sep_rows)}")
for r in sep_rows:
    print(f"  separated {fmt_row(r)}")
print()
print(f"submission.json (full 120-puzzle eval shape): {submission_path}")
print(f"submission_separated.json (separated ids):    {submission_separated_path}")
print(f"submission_grouped.json (grouped, original ids): {submission_grouped_path}")
print(f"Report: {report_path}")
'''


def make_code_cell(src):
    if not src.endswith("\n"):
        src = src + "\n"
    lines = src.splitlines(keepends=True)
    # Strip the trailing newline on the last line so the notebook viewer
    # doesn't add a phantom blank — match the base notebook's convention.
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
    make_code_cell(cell5_src),
    make_code_cell(solver_src),
    make_code_cell(NEW_STARTER),
    make_code_cell(cell_grouped_src),
    make_code_cell(cell_separated_src),
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
