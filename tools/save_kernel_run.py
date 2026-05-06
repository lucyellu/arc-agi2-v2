"""Package a finished Kaggle kernel run into results/ with our naming convention.

Usage:
    python tools/save_kernel_run.py <owner>/<slug> <duration_label> <descriptor>

Example:
    python tools/save_kernel_run.py lucyellu/vanilla-chunked 1297.97s chunk1-mirror

Produces:
    results/results_<slug>-<duration_label>-<descriptor>/
        <slug>-<duration_label>-<descriptor>.ipynb   (kernel source)
        <slug>.log                                    (full log via UTF-8 fetch)
        <all kernel output files>                     (submission, reports, etc.)

Why this exists:
- `kaggle kernels output` writes the log file in cp1252 on Windows and crashes
  on the unsloth 🦥 emoji, leaving an empty .log. We fetch via the lower-level
  API and write UTF-8.
- The notebook source isn't auto-included by `kernels output`. We add it via
  `kernels pull`.
"""
import argparse
import io
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", help="<owner>/<slug>")
    ap.add_argument("duration", help="e.g. 1297.97s, or 21min")
    ap.add_argument("descriptor", help="short name for this run, e.g. chunk1-mirror")
    args = ap.parse_args()

    owner, slug = args.kernel.split("/", 1)
    folder_name = f"results_{slug}-{args.descriptor}-{args.duration}"
    out = ROOT / "results" / folder_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Target: {out}")

    # 1. Pull all output files via Kaggle CLI. Suppress the cp1252 crash by
    #    routing stdout through utf-8.
    env = {"PYTHONIOENCODING": "utf-8"}
    print("Pulling output files...")
    subprocess.run(
        ["kaggle", "kernels", "output", args.kernel, "-p", str(out), "-o", "-q"],
        env={**__import__("os").environ, **env},
        check=False,
    )

    # 2. Replace the (likely-empty) log file with one fetched via UTF-8 API.
    log_path = out / f"{slug}.log"
    print(f"Fetching log via UTF-8 API to {log_path.name}")
    subprocess.run(
        ["python", str(ROOT / "tools" / "_fetch_log3.py"), owner, slug, str(log_path)],
        env={**__import__("os").environ, **env},
        check=True,
    )
    # Clean up any other .log file the CLI dropped (zero bytes from cp1252 crash).
    for stale in out.glob(f"{slug}*.log"):
        if stale.name != log_path.name and stale.stat().st_size == 0:
            stale.unlink()

    # 3. Pull the kernel notebook source and rename to convention.
    print("Pulling notebook source...")
    subprocess.run(
        ["kaggle", "kernels", "pull", args.kernel, "-p", str(out)],
        env={**__import__("os").environ, **env},
        check=False,
    )
    # `kernels pull` saves as <slug>.ipynb (with hyphens not underscores).
    pulled = out / f"{slug}.ipynb"
    target = out / f"{slug}-{args.descriptor}-{args.duration}.ipynb"
    if pulled.exists() and pulled != target:
        shutil.move(str(pulled), str(target))
    # Some kernels save with underscored variants — handle both.
    for candidate in out.glob("*.ipynb"):
        if candidate.name == target.name:
            continue
        # Rename if it looks like the kernel source (top-level .ipynb, not a saved-output one).
        if not any(p in candidate.name for p in (args.duration, args.descriptor)):
            shutil.move(str(candidate), str(target))
            break

    print(f"\nDone. Contents of {out}:")
    for p in sorted(out.iterdir()):
        size = p.stat().st_size if p.is_file() else "(dir)"
        print(f"  {p.name}  {size}")


if __name__ == "__main__":
    main()
