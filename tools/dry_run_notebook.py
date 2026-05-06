"""Dry-run a Kaggle inference notebook locally — no GPU required.

Usage:
    python tools/dry_run_notebook.py notebooks/separated172_36a08778.ipynb

Approach:
- exec each code cell in a shared namespace, in order.
- Skip %%writefile (write to disk), !shell (skip), %magic (skip).
- Stub subprocess.run so any GPU phase becomes a no-op that drops a
  synthetic bz2 inference output (using real solutions) into the
  redirected inference_outputs dir.
- Replace `/kaggle/working` and `/kaggle/inference_outputs` and
  `/kaggle/input/competitions/...` in cell source with our temp-tree paths
  before exec. Cells that already fall back to `Path.cwd()` when
  `/kaggle/working` doesn't exist also work — both paths converge on the
  temp tree.

What it catches:
- Syntax errors in any cell
- f-string / IPython-magic interpolation bugs (the v2 {global_end_time} bug)
- Env-var plumbing (AGI2_CHALLENGES_PATH / AGI2_DEV_KEYS handoff)
- Data-shape bugs (cell 4 separated set, cell 10 decode/recombine)
- Submission-file shape

What it does NOT catch:
- Bugs that only manifest with the real model loaded
- Kaggle-side quirks (model_sources mounting, machine_shape downgrade)
- TTT / inference correctness
"""
import argparse
import bz2
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "arc-prize-2026-arc-agi-2"
SEP_DATA_DIR = ROOT / "data" / "arc-agi-2-separated-eval-172"


def write_synthetic_inference_output(out_dir: Path, sub_key: str, solution_grid):
    """Mimic arc_solver.worker output: bz2-pickle list of decode dicts.

    Decoder reads files named <subkey-with-aug-suffix>; subkey before the
    first '.' is parsed to test_id. Use a minimal aug suffix.

    Solver writes np.ndarray solutions; the decoder later calls .tolist()
    on them. Mirror that shape so dry-run plumbing matches.
    """
    import numpy as np

    fname = f"{sub_key}.permute0123456789"
    out_path = out_dir / fname
    decoded_result = [
        {
            "beam_score": 0.1,
            "score_aug": [0.05] * 8,
            "solution": np.asarray(solution_grid, dtype=int),
        }
    ]
    with bz2.BZ2File(out_path, "w") as f:
        pickle.dump(decoded_result, f)


def make_fake_subprocess_run(infer_dir: Path):
    orig_chal = json.loads((DATA_DIR / "arc-agi_evaluation_challenges.json").read_text())
    orig_sol = json.loads((DATA_DIR / "arc-agi_evaluation_solutions.json").read_text())

    def fake_run(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args", [])
        env = kwargs.get("env", os.environ)
        dev_keys_str = env.get("AGI2_DEV_KEYS", "")
        dev_keys = dev_keys_str.split() if dev_keys_str else []
        chal_path = env.get("AGI2_CHALLENGES_PATH", "")
        print(f"  [fake subprocess] argv={argv}")
        print(f"  [fake subprocess] AGI2_CHALLENGES_PATH={chal_path or '<original eval>'}")
        print(f"  [fake subprocess] AGI2_DEV_KEYS={dev_keys}")
        # v2 bug check: --end-time arg must parse as float.
        if "--end-time" in argv:
            i = argv.index("--end-time")
            val = argv[i + 1]
            try:
                float(val)
            except (ValueError, TypeError):
                print(
                    f"  [fake subprocess] BUG: --end-time arg is {val!r} — "
                    f"not a float. Likely an unexpanded f-string or "
                    f"IPython placeholder."
                )
                return subprocess.CompletedProcess(argv, returncode=2)
        # Pick the right challenges/solutions for synthesis.
        if chal_path and Path(chal_path).exists():
            chal = json.loads(Path(chal_path).read_text())
            sol_path = (
                Path(chal_path).parent
                / Path(chal_path).name.replace("challenges", "solutions")
            )
            sol = json.loads(sol_path.read_text()) if sol_path.exists() else {}
        else:
            chal, sol = orig_chal, orig_sol
        infer_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for key in dev_keys:
            if key not in chal:
                continue
            n_tests = len(chal[key]["test"])
            for i in range(n_tests):
                grid = sol.get(key, [[[0]]] * n_tests)[i] if key in sol else [[0]]
                write_synthetic_inference_output(infer_dir, f"{key}_{i}", grid)
                n_written += 1
        print(
            f"  [fake subprocess] wrote {n_written} synthetic outputs to {infer_dir}"
        )
        return subprocess.CompletedProcess(argv, returncode=0)

    return fake_run


def cell_kind(src):
    s = src.lstrip()
    if s.startswith("%%writefile"):
        return "writefile"
    if s.startswith("!"):
        return "shell"
    if s.startswith("%"):
        return "magic"
    return "python"


def handle_writefile_cell(src, base_dir):
    first_nl = src.find("\n")
    header, body = src[:first_nl], src[first_nl + 1:]
    parts = header.split()
    if len(parts) < 2:
        raise ValueError(f"malformed writefile header: {header!r}")
    target = base_dir / parts[1]
    target.write_text(body, encoding="utf-8")
    print(f"  [writefile] wrote {target} ({len(body)} chars)")


def rewrite_cell_paths(src, working_str, inference_str, comp_input_str, sep_input_str):
    """Replace /kaggle/... string literals so cells run against our temp tree.

    Backslashes in Windows paths are escaped so they survive Python string
    literals (we substitute INTO source code).
    """
    def esc(s):
        return s.replace("\\", "\\\\")

    out = src
    out = out.replace('"/kaggle/inference_outputs"', f'r"{inference_str}"')
    out = out.replace("'/kaggle/inference_outputs'", f"r'{inference_str}'")
    out = out.replace('"/kaggle/working"', f'r"{working_str}"')
    out = out.replace("'/kaggle/working'", f"r'{working_str}'")
    out = out.replace(
        '"/kaggle/input/competitions/arc-prize-2026-arc-agi-2"',
        f'r"{comp_input_str}"',
    )
    out = out.replace(
        "'/kaggle/input/competitions/arc-prize-2026-arc-agi-2'",
        f"r'{comp_input_str}'",
    )
    out = out.replace(
        '"/kaggle/input/arc-agi-2-separated-eval-172"',
        f'r"{sep_input_str}"',
    )
    out = out.replace(
        "'/kaggle/input/arc-agi-2-separated-eval-172'",
        f"r'{sep_input_str}'",
    )
    # Catch the "/kaggle/working/foo" pattern (used for output paths)
    out = re.sub(
        r'"/kaggle/working/([^"]+)"',
        lambda m: f'r"{esc(working_str)}\\\\{m.group(1)}"',
        out,
    )
    out = re.sub(
        r"'/kaggle/working/([^']+)'",
        lambda m: f"r'{esc(working_str)}\\\\{m.group(1)}'",
        out,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    cells = nb["cells"]

    tmp = Path(tempfile.mkdtemp(prefix="kaggle_dry_"))
    print(f"Temp dir: {tmp}")
    working = tmp / "working"
    inference = tmp / "inference_outputs"
    comp_input = tmp / "input" / "competitions" / "arc-prize-2026-arc-agi-2"
    sep_input = tmp / "input" / "arc-agi-2-separated-eval-172"
    for d in (working, inference, comp_input, sep_input):
        d.mkdir(parents=True, exist_ok=True)
    # Stage eval data so cell 4's find_eval_files can pick it up.
    for name in (
        "arc-agi_evaluation_challenges.json",
        "arc-agi_evaluation_solutions.json",
        "arc-agi_test_challenges.json",
        "sample_submission.json",
    ):
        src = DATA_DIR / name
        if src.exists():
            shutil.copy(src, comp_input / name)
    # Stage the separated dataset (the one we'd attach as a Kaggle input).
    if SEP_DATA_DIR.exists():
        for src in SEP_DATA_DIR.iterdir():
            if src.is_file() and src.suffix == ".json":
                shutil.copy(src, sep_input / src.name)
        print(f"Staged separated dataset from {SEP_DATA_DIR} -> {sep_input}")
    else:
        print(
            f"WARNING: {SEP_DATA_DIR} missing — run "
            f"`python scripts/build_separated_eval.py` first"
        )

    cwd_orig = Path.cwd()
    os.chdir(working)
    sys.path.insert(0, str(working))

    fake_run = make_fake_subprocess_run(inference)
    real_subprocess_run = subprocess.run
    subprocess.run = fake_run

    # Inject a minimal `transformers` stub so `from transformers import
    # AutoTokenizer` in arc_loader.py succeeds on machines whose torch is
    # broken (or where torch isn't installed). The tokenizer object is
    # never called along the decode path that cell 10 exercises.
    import types

    if "transformers" not in sys.modules:
        stub = types.ModuleType("transformers")

        class _StubAutoTokenizer:  # noqa: D401
            """Placeholder; never instantiated by the dry-run path."""

            @staticmethod
            def from_pretrained(*a, **kw):
                raise RuntimeError(
                    "transformers stub: from_pretrained not available in dry-run"
                )

        stub.AutoTokenizer = _StubAutoTokenizer
        sys.modules["transformers"] = stub

    ns = {"__name__": "__main__"}
    statuses = []

    working_str = str(working)
    inference_str = str(inference)
    comp_input_str = str(comp_input)
    sep_input_str = str(sep_input)

    try:
        for i, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            kind = cell_kind(src)
            print(f"\n=== cell {i} ({kind}) ===")
            try:
                if kind == "python":
                    rewritten = rewrite_cell_paths(
                        src,
                        working_str,
                        inference_str,
                        comp_input_str,
                        sep_input_str,
                    )
                    exec(compile(rewritten, f"<cell {i}>", "exec"), ns, ns)
                    statuses.append((i, "ok"))
                elif kind == "writefile":
                    handle_writefile_cell(src, working)
                    statuses.append((i, "writefile"))
                elif kind == "shell":
                    print(f"  [skip shell] {src.splitlines()[0][:80]}")
                    statuses.append((i, "skip-shell"))
                else:
                    print(f"  [skip magic] {src.splitlines()[0][:80]}")
                    statuses.append((i, "skip-magic"))
            except SystemExit as e:
                print(f"  [SystemExit code={e.code}]")
                statuses.append((i, f"exit:{e.code}"))
            except Exception as e:  # noqa: BLE001
                print(f"  [ERROR] {type(e).__name__}: {e}")
                print(traceback.format_exc())
                statuses.append((i, f"error:{type(e).__name__}"))
                break
    finally:
        subprocess.run = real_subprocess_run
        os.chdir(cwd_orig)

    print("\n=== Summary ===")
    for i, st in statuses:
        print(f"  cell {i:>2}: {st}")

    print("\n=== Expected output files in working tree ===")
    expected = [
        "submission.json",
        "submission_separated.json",
        "submission_grouped.json",
    ]
    for name in expected:
        p = working / name
        if p.exists():
            print(f"  OK  {name}  ({p.stat().st_size} bytes)")
        else:
            print(f"  --  {name}  (missing)")

    failed = [s for s in statuses if s[1].startswith(("error", "exit"))]
    if failed:
        print(f"\nFAILED: {len(failed)} cell(s) errored.")
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(1)
    print("\nAll cells executed (subprocess + GPU phases stubbed).")
    if args.keep:
        print(f"Temp dir kept at {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
