#!/usr/bin/env python3
"""
Visualize ARC puzzle predictions as PNG images.

Usage:
  python viz_arc.py --challenges ../data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json
                    --submission results/test_submission.json
                    [--solutions ../data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json]
                    [--out results/viz]
                    [--ids 007bbfb7 00d62c1b]
                    [--source-label "2024 ARC-AGI-1 training"]

Outputs one PNG per puzzle to --out directory.
Filename format: {source_tag}_{YYYYMMDD_HHMMSS}_{checkpoint}_{puzzle_id}.png
Each run produces a unique filename — existing PNGs are never overwritten.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise SystemExit("Pillow required: pip install Pillow")

# ARC canonical color palette (index 0-9)
ARC_COLORS = [
    (0,   0,   0),    # 0 black
    (0,  116, 217),   # 1 blue
    (255, 65,  54),   # 2 red
    (46, 204,  64),   # 3 green
    (255,220,   0),   # 4 yellow
    (170,170, 170),   # 5 grey
    (240, 18, 190),   # 6 magenta
    (255,133,  27),   # 7 orange
    (127,219, 255),   # 8 azure
    (135, 12,  37),   # 9 maroon
]

CELL        = 28    # px per grid cell
PAD         = 14    # px between sections
LABEL_H     = 32    # px reserved for section label text
BORDER      = 3     # grid border thickness
BG          = (245, 245, 245)
TITLE_BG    = (30,  30,  30)
TITLE_FG    = (255, 255, 255)
SECTION_FG  = (50,  50,  50)
TRAIN_FG    = (0,   80, 180)
TEST_FG     = (140,  0, 200)
CORRECT_BG  = (0,  160,   0)
WRONG_BG    = (200,  0,   0)
ATTEMPT_FG  = (0,   80, 180)
ARROW_FG    = (80,  80,  80)


def arc_color(val):
    return ARC_COLORS[int(val)] if 0 <= int(val) <= 9 else (200, 200, 200)


def grid_size(grid):
    rows = len(grid)
    cols = max(len(r) for r in grid) if rows else 1
    return rows, cols


def grid_px(grid):
    r, c = grid_size(grid)
    return c * CELL, r * CELL   # width, height


def load_font(size, bold=False):
    names = (["arialbd.ttf", "Arial Bold.ttf"] if bold else
             ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_label_box(draw, x, y, text, font, bg, fg, pad=6):
    """Draw a filled rectangle with centered text, returns (width, height)."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    bw = tw + pad * 2
    bh = th + pad * 2
    draw.rectangle([x, y, x + bw, y + bh], fill=bg)
    draw.text((x + pad, y + pad), text, fill=fg, font=font)
    return bw, bh


def draw_grid(draw, grid, x0, y0, border_color=(80, 80, 80)):
    rows, cols = grid_size(grid)
    w = cols * CELL
    h = rows * CELL
    draw.rectangle([x0 - BORDER, y0 - BORDER,
                    x0 + w + BORDER, y0 + h + BORDER],
                   outline=border_color, width=BORDER)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            draw.rectangle([x0 + c * CELL, y0 + r * CELL,
                            x0 + (c+1) * CELL - 1, y0 + (r+1) * CELL - 1],
                           fill=arc_color(val))


def render_puzzle(pid, puzzle, pred_attempts, solution=None,
                  source_label="", source_tag="",
                  puzzle_index=None, total_puzzles=None,
                  checkpoint_name=None, exec_time_s=None):
    """
    Renders a full puzzle image:
      - Bold title bar with puzzle ID + dataset source
      - Train rows: each with labeled Input -> Output
      - Separator
      - Test row: Test Input | attempt_1 | attempt_2 | ... | [Solution + CORRECT/WRONG badge]
    """
    font_title  = load_font(28, bold=True)
    font_head   = load_font(22, bold=True)
    font_label  = load_font(18)
    font_badge  = load_font(20, bold=True)

    train  = puzzle["train"]
    test_in = puzzle["test"][0]["input"]

    # --- Measure max grid dimensions ---
    max_train_in_w  = max(grid_px(ex["input"])[0]  for ex in train)
    max_train_out_w = max(grid_px(ex["output"])[0] for ex in train)
    max_train_h     = max(
        max(grid_px(ex["input"])[1], grid_px(ex["output"])[1]) for ex in train
    )
    ARROW_W = 36   # px for arrow column

    # --- Flatten all attempts ---
    # pred_attempts is a list of dicts like {"attempt_1": grid, "attempt_2": grid, ...}
    attempt_grids = []
    for att in pred_attempts:
        for key in sorted(att.keys()):
            g = att.get(key)
            if g is not None:
                attempt_grids.append((key, g))

    sol_grid = solution[0] if solution else None

    # --- Compute image width ---
    TITLE_H      = 72
    TRAIN_ROW_H  = LABEL_H + max_train_h + PAD
    SEP_H        = PAD
    TEST_LABEL_H = LABEL_H

    train_w = max_train_in_w + ARROW_W + max_train_out_w + PAD * 2
    ti_w, ti_h = grid_px(test_in)
    test_w = PAD + ti_w
    for _, g in attempt_grids:
        test_w += PAD + grid_px(g)[0]
    if sol_grid:
        test_w += PAD + grid_px(sol_grid)[0] + PAD   # + badge width approx
    test_w += PAD

    img_w = max(train_w, test_w) + PAD * 2
    test_h = max(
        ti_h,
        max((grid_px(g)[1] for _, g in attempt_grids), default=0),
        grid_px(sol_grid)[1] if sol_grid else 0,
    )
    BADGE_H = 40   # height for CORRECT/WRONG badge below solution
    img_h = (TITLE_H
             + len(train) * TRAIN_ROW_H + PAD
             + SEP_H
             + TEST_LABEL_H + test_h + BADGE_H + PAD * 2)

    img  = Image.new("RGB", (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)

    # === Title bar ===
    draw.rectangle([0, 0, img_w, TITLE_H], fill=TITLE_BG)
    index_prefix = (f"#{puzzle_index} / {total_puzzles}  " if puzzle_index is not None else "")
    title_text  = f"{index_prefix}Puzzle: {pid}"
    draw.text((PAD, 6),  title_text,  fill=TITLE_FG, font=font_title)
    # Subtitle line: source | checkpoint | exec time
    subtitle_parts = []
    if source_label or source_tag:
        subtitle_parts.append(source_label or source_tag)
    if checkpoint_name:
        subtitle_parts.append(f"model: {checkpoint_name}")
    if exec_time_s is not None:
        t = float(exec_time_s)
        subtitle_parts.append(f"inference: {t:.1f}s" if t < 60 else f"inference: {t/60:.1f}min")
    if subtitle_parts:
        draw.text((PAD, 44), "  |  ".join(subtitle_parts), fill=(180, 220, 255), font=font_label)

    y = TITLE_H + PAD

    # === Train rows ===
    for i, ex in enumerate(train):
        x = PAD

        # "Train N" label
        draw.text((x, y), f"Train example {i+1}", fill=TRAIN_FG, font=font_head)

        row_y = y + LABEL_H

        # Input
        inp_w, inp_h = grid_px(ex["input"])
        in_label_x = x + (max_train_in_w - inp_w) // 2
        draw.text((in_label_x, row_y - 1), "Input", fill=SECTION_FG, font=font_label)
        draw_grid(draw, ex["input"], x + (max_train_in_w - inp_w) // 2, row_y + 16,
                  border_color=(100, 100, 100))

        # Arrow
        ax = x + max_train_in_w + ARROW_W // 2 - 8
        ay = row_y + max_train_h // 2
        draw.text((ax, ay), "->", fill=ARROW_FG, font=font_head)

        # Output
        out_w, out_h = grid_px(ex["output"])
        ox = x + max_train_in_w + ARROW_W
        draw.text((ox, row_y - 1), "Output", fill=SECTION_FG, font=font_label)
        draw_grid(draw, ex["output"], ox + (max_train_out_w - out_w) // 2, row_y + 16,
                  border_color=(100, 100, 100))

        y += TRAIN_ROW_H

    # Separator
    draw.line([(PAD, y + 4), (img_w - PAD, y + 4)], fill=(180, 180, 180), width=2)
    y += SEP_H + 4

    # === Test row ===
    # One row of labels, then grids below
    x = PAD
    labels_y = y
    grids_y  = y + LABEL_H

    draw.text((x, labels_y), "Test Input", fill=TEST_FG, font=font_head)
    draw_grid(draw, test_in, x, grids_y, border_color=(100, 100, 100))
    x += ti_w + PAD

    # Attempts — badge (CORRECT / WRONG) shown below each attempt, not on solution
    for att_label, g in attempt_grids:
        aw, ah = grid_px(g)
        is_correct = (sol_grid is not None and g == sol_grid)
        border_col = (0, 160, 0) if is_correct else (60, 60, 200)
        draw.text((x, labels_y), att_label, fill=ATTEMPT_FG, font=font_head)
        draw_grid(draw, g, x, grids_y, border_color=border_col)
        if sol_grid is not None:
            badge_text = "CORRECT" if is_correct else "WRONG"
            badge_bg   = CORRECT_BG if is_correct else WRONG_BG
            draw_label_box(draw, x, grids_y + ah + 6,
                           badge_text, font_badge, badge_bg, (255, 255, 255), pad=8)
        x += aw + PAD

    # Solution column — label only, no badge
    if sol_grid:
        draw.text((x, labels_y), "Solution", fill=SECTION_FG, font=font_head)
        draw_grid(draw, sol_grid, x, grids_y, border_color=(80, 80, 80))

    return img


def source_tag_from_path(path: str) -> str:
    """Derive a short tag from the challenges file path, e.g. '2025eval'."""
    p = Path(path)
    parts = p.parts
    # find the data dir name  e.g. 'arc-prize-2025'
    for part in parts:
        if "2024" in part:
            year = "2024"
        elif "2025" in part:
            year = "2025"
        elif "2026" in part:
            year = "2026"
        else:
            continue
        stem = p.stem  # e.g. arc-agi_training_challenges
        if "training" in stem:
            split = "train"
        elif "evaluation" in stem:
            split = "eval"
        elif "test" in stem:
            split = "test"
        else:
            split = "unknown"
        return f"{year}{split}"
    return p.stem


def source_label_from_path(path: str) -> str:
    p = Path(path)
    stem = p.stem
    # find parent folder name
    folder = ""
    for part in p.parts:
        if "arc-prize" in part or "arc_prize" in part:
            folder = part
    split = ("training" if "training" in stem
             else "evaluation" if "evaluation" in stem
             else "test" if "test" in stem else stem)
    return f"Dataset: {folder}  |  Split: {split}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenges",    required=True,
                        help="Path to ARC challenges JSON")
    parser.add_argument("--submission",    required=True,
                        help="Path to submission JSON")
    parser.add_argument("--solutions",     default=None,
                        help="Path to solutions JSON (optional)")
    parser.add_argument("--out",           default="results/viz",
                        help="Output directory for PNGs")
    parser.add_argument("--ids",           nargs="*",
                        help="Specific puzzle IDs to render (default: all in submission)")
    parser.add_argument("--source-label",  default=None,
                        help="Override source label shown in title (e.g. '2024 ARC-AGI-1 training')")
    parser.add_argument("--index",           type=int,   default=None,
                        help="1-based index of this puzzle in the current view (shown in title)")
    parser.add_argument("--total",           type=int,   default=None,
                        help="Total number of puzzles in the current view (shown in title)")
    parser.add_argument("--checkpoint-name", default=None,
                        help="Model / checkpoint label shown in subtitle and filename")
    parser.add_argument("--exec-time",       type=float, default=None,
                        help="Inference time in seconds for this puzzle (shown in subtitle)")
    args = parser.parse_args()

    challenges = json.load(open(args.challenges))
    submission = json.load(open(args.submission))
    solutions  = json.load(open(args.solutions)) if args.solutions else {}

    tag   = source_tag_from_path(args.challenges)
    label = args.source_label or source_label_from_path(args.challenges)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.ids or list(submission.keys())
    total = args.total or len(ids)

    # Timestamp shared by all PNGs in this run — prevents overwriting previous runs
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Safe checkpoint slug for filenames  (e.g. "trm_v2", "qwen3_4B")
    ckpt_slug = ""
    if args.checkpoint_name:
        ckpt_slug = "_" + re.sub(r"[^A-Za-z0-9_.-]", "_", args.checkpoint_name)[:24]

    saved = []
    for i, pid in enumerate(ids):
        if pid not in challenges:
            print(f"WARNING: {pid} not in challenges, skipping")
            continue
        if pid not in submission:
            print(f"WARNING: {pid} not in submission, skipping")
            continue

        puzzle = challenges[pid]
        preds  = submission[pid]
        sol    = solutions.get(pid)
        index  = args.index if args.index is not None else (i + 1)

        img = render_puzzle(pid, puzzle, preds, solution=sol,
                            source_label=label, source_tag=tag,
                            puzzle_index=index, total_puzzles=total,
                            checkpoint_name=args.checkpoint_name,
                            exec_time_s=args.exec_time)

        # Filename: {tag}_{YYYYMMDD_HHMMSS}_{ckpt}_{pid}.png
        # Never collides with previous runs — safe to compare side-by-side
        fname = f"{tag}_{run_ts}{ckpt_slug}_{pid}.png"
        out_path = out_dir / fname
        img.save(out_path)
        print(f"Saved: {out_path}")
        saved.append(str(out_path))

    print(f"\nDone. {len(saved)} image(s) in {out_dir}/")


if __name__ == "__main__":
    main()
