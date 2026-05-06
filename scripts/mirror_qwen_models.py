"""Mirror sorokin's Qwen3 grid-tuned models to our Kaggle account as Kaggle Models.

Why: sorokin already deleted one of his Qwen3 grid models from Kaggle.
If they delete `qwen3_4b_grids15_sft139` we lose our inference path. This
script downloads sorokin's public model versions and re-uploads them as
Kaggle Models under `lucyellu/`. Models (not Datasets) is the right
primitive — they're attached to kernels via `model_sources`, the catalog
shows framework/variant metadata, and it aligns with our eventual goal
of training and publishing our own grid-tuned model.

Usage:
    python scripts/mirror_qwen_models.py            # mirror both 4B and 2B
    python scripts/mirror_qwen_models.py --only-4b  # just the one we use
    python scripts/mirror_qwen_models.py --skip-upload  # download only

Cost (rough):
- 4B bf16 ≈ 8–9 GB
- 2B bf16 ≈ 4–5 GB
- Plan for 1–2+ hours on a typical home connection. Disk peak ≈ sum of
  selected sizes (download files stay on disk through upload).

Idempotency:
- Already-downloaded local files are skipped.
- Already-existing Kaggle Models / Instances / Versions are detected and
  uploaded as a NEW version (not duplicated).
"""
import argparse
import io
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKDIR = ROOT / "data" / "mirrors"
OWNER = "lucyellu"


# Each entry mirrors one Kaggle Model instance version on sorokin's account
# to one Kaggle Model on our account.
MIRRORS = [
    {
        "name": "4b",
        "src_slug": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        "dst": {
            "model_slug": "qwen3-4b-grids15-sft139-mirror",
            "title": "Qwen3 4B Grids15 SFT139 (mirror)",
            "subtitle": "Mirror of sorokin/qwen3_4b_grids15_sft139 for durability.",
            "framework": "transformers",
            "instance_slug": "transformers",
            "variation": "bfloat16",
            "version_number": "1",
        },
        "approx_gb": 9,
    },
    {
        "name": "2b",
        "src_slug": "sorokin/qwen3_2b_grids15_sft141/transformers/bfloat16/1",
        "dst": {
            "model_slug": "qwen3-2b-grids15-sft141-mirror",
            "title": "Qwen3 2B Grids15 SFT141 (mirror)",
            "subtitle": "Mirror of sorokin/qwen3_2b_grids15_sft141 for durability.",
            "framework": "transformers",
            "instance_slug": "transformers",
            "variation": "bfloat16",
            "version_number": "1",
        },
        "approx_gb": 5,
    },
]


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def run_capture(cmd):
    return subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def download_model(workdir: Path, mirror: dict) -> Path:
    target = workdir / f"{mirror['name']}_dl"
    if target.exists() and any(target.iterdir()):
        print(f"[{mirror['name']}] already populated at {target}, skipping download")
        return target
    target.mkdir(parents=True, exist_ok=True)
    free = free_gb(workdir)
    print(
        f"[{mirror['name']}] downloading ~{mirror['approx_gb']} GB to {target} "
        f"(free disk: {free:.1f} GB)"
    )
    if free < mirror["approx_gb"] * 1.5:
        raise SystemExit(
            "Refusing to start: less than 1.5x the expected size free on this drive. "
            "Free disk space first."
        )
    run([
        "kaggle", "models", "instances", "versions", "download",
        mirror["src_slug"], "-p", str(target), "--untar",
    ])
    return target


def model_exists(slug: str) -> bool:
    """`kaggle models get <owner>/<slug>` exits 0 if the model exists."""
    return run_capture(["kaggle", "models", "get", slug]).returncode == 0


def instance_exists(model_slug: str, instance_slug: str) -> bool:
    return run_capture([
        "kaggle", "models", "instances", "get",
        f"{model_slug}/{instance_slug}",
    ]).returncode == 0


def write_model_metadata(staging: Path, mirror: dict, src_slug: str):
    dst = mirror["dst"]
    body = (
        f"Mirror of [{src_slug}](https://www.kaggle.com/models/"
        f"{src_slug.split('/')[0]}/{src_slug.split('/')[1]}) copied "
        f"{date.today().isoformat()}.\n\n"
        "Re-hosted under lucyellu/ for durability — sorokin has deleted "
        "qwen3 grid-tuned models in the past. Please cite the original "
        "author for any use.\n"
    )
    meta = {
        "ownerSlug": OWNER,
        "title": dst["title"],
        "slug": dst["model_slug"],
        "subtitle": dst["subtitle"],
        "isPrivate": True,
        "description": body,
        "publishTime": "",
        "provenanceSources": src_slug,
    }
    (staging / "model-metadata.json").write_text(json.dumps(meta, indent=2))


def write_instance_metadata(staging: Path, mirror: dict):
    dst = mirror["dst"]
    body = (
        "Mirror of sorokin's Qwen3 grid-tuned SFT model. Same weights, "
        "same tokenizer, same architecture as the source. Use as a "
        "drop-in replacement.\n"
    )
    meta = {
        "ownerSlug": OWNER,
        "modelSlug": dst["model_slug"],
        "instanceSlug": dst["instance_slug"],
        "framework": dst["framework"],
        "licenseName": "Apache 2.0",
        "overview": body,
        "usage": (
            "Attach to a Kaggle kernel via model_sources, or read directly "
            "from /kaggle/input/<mount>/config.json."
        ),
        "fineTunable": True,
        "trainingData": [],
        "modelInstanceType": "Unspecified",
        "baseModelInstanceInformation": None,
        "externalBaseModelUrl": (
            f"https://www.kaggle.com/models/{mirror['src_slug'].split('/')[0]}/"
            f"{mirror['src_slug'].split('/')[1]}"
        ),
    }
    (staging / "model-instance-metadata.json").write_text(json.dumps(meta, indent=2))


def write_version_metadata(staging: Path, mirror: dict):
    meta = {
        "versionNotes": (
            f"Mirror of {mirror['src_slug']} ({mirror['dst']['variation']}) "
            f"copied {date.today().isoformat()}."
        ),
    }
    (staging / "model-instance-version-metadata.json").write_text(
        json.dumps(meta, indent=2)
    )


def stage_files(download_dir: Path, staging: Path):
    """Copy/move the downloaded files into staging, then write metadata next to them."""
    staging.mkdir(parents=True, exist_ok=True)
    for src in download_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(download_dir)
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            # hardlink if possible (same drive, free), else copy
            try:
                dst.hardlink_to(src)
            except (OSError, AttributeError):
                shutil.copy2(src, dst)


def upload_model(staging: Path, mirror: dict):
    dst = mirror["dst"]
    full_model_slug = f"{OWNER}/{dst['model_slug']}"
    full_instance_slug = f"{full_model_slug}/{dst['framework']}/{dst['variation']}"

    # 1. Create the top-level Model entity if it doesn't exist.
    write_model_metadata(staging, mirror, mirror["src_slug"])
    if model_exists(full_model_slug):
        print(f"[{mirror['name']}] model {full_model_slug} exists, skipping create")
    else:
        print(f"[{mirror['name']}] creating model {full_model_slug}")
        run(["kaggle", "models", "create", "-p", str(staging)])

    # 2. Create the instance (framework/variation) if it doesn't exist.
    write_instance_metadata(staging, mirror)
    if instance_exists(full_model_slug, f"{dst['framework']}/{dst['variation']}"):
        print(
            f"[{mirror['name']}] instance {full_instance_slug} exists, "
            "uploading new version"
        )
    else:
        print(f"[{mirror['name']}] creating instance {full_instance_slug}")
        run(["kaggle", "models", "instances", "create", "-p", str(staging)])

    # 3. Upload a new version (always — Kaggle handles version numbering).
    write_version_metadata(staging, mirror)
    print(f"[{mirror['name']}] uploading version files")
    run([
        "kaggle", "models", "instances", "versions", "create",
        full_instance_slug,
        "-p", str(staging),
    ])

    print(
        f"[{mirror['name']}] -> https://www.kaggle.com/models/{full_model_slug}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--only-4b", action="store_true")
    ap.add_argument("--only-2b", action="store_true")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    selected = MIRRORS
    if args.only_4b:
        selected = [m for m in MIRRORS if m["name"] == "4b"]
    elif args.only_2b:
        selected = [m for m in MIRRORS if m["name"] == "2b"]

    total_gb = sum(m["approx_gb"] for m in selected)
    print(
        f"Mirroring {[m['name'] for m in selected]} "
        f"(~{total_gb} GB on disk + same upload bandwidth)"
    )
    free = free_gb(args.workdir)
    print(f"Free disk on workdir drive: {free:.1f} GB")

    for mirror in selected:
        download_dir = args.workdir / f"{mirror['name']}_dl"
        staging = args.workdir / f"{mirror['name']}_staging"
        if not args.skip_download:
            download_dir = download_model(args.workdir, mirror)
        if not args.skip_upload:
            stage_files(download_dir, staging)
            upload_model(staging, mirror)

    print("\nDone.")
    print(
        "After mirror lands, switch kernel-metadata.json `model_sources` to "
        "use our copy alongside (or instead of) sorokin's:"
    )
    for m in selected:
        print(
            f"  {OWNER}/{m['dst']['model_slug']}/"
            f"{m['dst']['framework']}/{m['dst']['variation']}/"
            f"{m['dst']['version_number']}"
        )
    print(
        "\nThe defensive _resolve_model_path() in arc_solver.py already globs "
        "for `qwen3_4b_grids15_sft139/.../config.json` under /kaggle/input/, "
        "so the mirror's mount path is picked up automatically (the dir name "
        "is preserved by Kaggle on attach)."
    )


if __name__ == "__main__":
    main()
