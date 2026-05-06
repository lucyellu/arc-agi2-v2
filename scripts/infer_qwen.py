#!/usr/bin/env python3
"""
Qwen3 inference for ARC puzzles — auto-adapts to available GPU.

Hardware tiers (auto-detected unless --quant is set):
  < 6 GB VRAM  → 4-bit (1050 Ti, MX series, CPU fallback)
  6–14 GB VRAM → 8-bit (T4, 3060, 2080)
  14+ GB VRAM  → float16/bfloat16 (A100, H100, RTX 4090)
  Multi-GPU    → device_map=auto across all detected GPUs

Usage:
  python infer_qwen.py \
    --challenges ../data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json \
    --output ../results/submissions/latest/qwen_submission.json \
    [--model Qwen/Qwen3-4B-Instruct] \
    [--lora ../external/models/architects_lora] \
    [--ids 00576224 007bbfb7] \
    [--quant auto|4bit|8bit|none] \
    [--attempts 2] \
    [--max-new-tokens 512]
"""
import argparse
import json
import signal
import sys
from pathlib import Path

import constraint_filter as CF

# ── GPU detection ──────────────────────────────────────────────────────────────

def detect_hardware():
    """Return a dict describing the best available compute config."""
    try:
        import torch
    except ImportError:
        return {"device": "cpu", "dtype_str": "float32", "quant": None, "gpus": 0, "vram_gb": 0}

    if not torch.cuda.is_available():
        print("[Qwen] No CUDA found — running on CPU (slow)", flush=True)
        return {"device": "cpu", "dtype_str": "float32", "quant": None, "gpus": 0, "vram_gb": 0}

    n = torch.cuda.device_count()
    vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9
    cap  = torch.cuda.get_device_capability(0)[0]   # major version (8=Ampere, 7=Turing, 6=Pascal)
    name = torch.cuda.get_device_name(0)

    print(f"[Qwen] GPU: {name} × {n}  |  VRAM: {vram:.1f} GB  |  Compute: {cap}.x", flush=True)

    # bfloat16 only on Ampere (8.x) and above
    dtype_str = "bfloat16" if cap >= 8 else "float16"

    if vram < 6:
        quant = "4bit"
        print(f"[Qwen] Low VRAM ({vram:.1f} GB) → 4-bit quantization", flush=True)
    elif vram < 14:
        quant = "8bit"
        print(f"[Qwen] Mid VRAM ({vram:.1f} GB) → 8-bit quantization", flush=True)
    else:
        quant = None
        print(f"[Qwen] High VRAM ({vram:.1f} GB) → {dtype_str} full precision", flush=True)

    return {"device": "cuda", "dtype_str": dtype_str, "quant": quant, "gpus": n, "vram_gb": vram}


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name, lora_path, hw, quant_override):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    quant = quant_override if quant_override != "auto" else hw["quant"]
    dtype_str = hw["dtype_str"]
    dtype = getattr(torch, dtype_str)

    print(f"[Qwen] Loading tokenizer: {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    bnb_cfg = None
    if quant == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quant == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

    print(f"[Qwen] Loading model ({quant or 'full'}, {dtype_str})…", flush=True)
    kwargs = dict(
        trust_remote_code=True,
        device_map="auto" if hw["device"] == "cuda" else None,
    )
    if bnb_cfg:
        kwargs["quantization_config"] = bnb_cfg
    elif hw["device"] == "cuda":
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    # Apply LoRA adapter if provided
    if lora_path and Path(lora_path).exists():
        print(f"[Qwen] Applying LoRA: {lora_path}", flush=True)
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            print("[Qwen] LoRA merged.", flush=True)
        except ImportError:
            print("[Qwen] WARNING: peft not installed, skipping LoRA. Run: pip install peft", flush=True)

    model.eval()
    return model, tok


# ── ARC puzzle formatting ──────────────────────────────────────────────────────

def grid_to_str(grid) -> str:
    """Convert grid to compact digit string, one row per line (no spaces)."""
    return "\n".join("".join(str(int(v)) for v in row) for row in grid)


def grids_equal(a, b) -> bool:
    """Return True if two grids are identical in shape and values."""
    if not a or not b or len(a) != len(b): return False
    return all(
        len(ra) == len(rb) and all(x == y for x, y in zip(ra, rb))
        for ra, rb in zip(a, b)
    )


def str_to_grid(text: str):
    """Parse a digit string back to a list-of-lists grid. Returns None on failure."""
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = [int(c) for c in line if c.isdigit()]
            if row:
                rows.append(row)
        except ValueError:
            continue
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [0] * (width - len(r)) for r in rows]
    return rows


def _count_regions(grid) -> int:
    """Count connected components of non-zero (non-black) cells via BFS."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    visited = [[False] * cols for _ in range(rows)]
    count = 0
    for sr in range(rows):
        for sc in range(cols):
            if grid[sr][sc] != 0 and not visited[sr][sc]:
                count += 1
                queue = [(sr, sc)]
                visited[sr][sc] = True
                while queue:
                    r, c = queue.pop()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
    return count


def analyze_puzzle(puzzle: dict) -> dict:
    """
    Compute transformation statistics from training pairs.
    Used to build informative prompts and detect congruent patterns.
    """
    pair_stats = []
    for p in puzzle["train"]:
        inp, out = p["input"], p["output"]
        in_r, in_c  = len(inp), len(inp[0]) if inp else 0
        out_r, out_c = len(out), len(out[0]) if out else 0
        size_preserved = (in_r == out_r and in_c == out_c)

        pct_changed = None
        if size_preserved:
            total = in_r * in_c
            changed = sum(
                1 for r in range(in_r) for c in range(in_c)
                if inp[r][c] != out[r][c]
            )
            pct_changed = round(changed / total * 100) if total else 0

        in_colors  = set(v for row in inp for v in row)
        out_colors = set(v for row in out for v in row)
        pair_stats.append({
            "in_shape":  f"{in_r}×{in_c}",
            "out_shape": f"{out_r}×{out_c}",
            "size_preserved": size_preserved,
            "pct_changed":    pct_changed,
            "colors_added":   sorted(out_colors - in_colors),
            "colors_removed": sorted(in_colors - out_colors),
            "out_regions":    _count_regions(out),
        })

    all_size_preserved = all(s["size_preserved"] for s in pair_stats)
    pcts = [s["pct_changed"] for s in pair_stats if s["pct_changed"] is not None]
    avg_pct = round(sum(pcts) / len(pcts)) if pcts else None
    congruent = (max(pcts) - min(pcts) < 15) if len(pcts) > 1 else True

    # All colors ever seen in training (allowed palette for output)
    allowed_colors = sorted(set(
        v for p in puzzle["train"]
        for grid in (p["input"], p["output"])
        for row in grid for v in row
    ))

    # Colors present in the test input
    test_input_colors = sorted(set(
        v for row in puzzle["test"][0]["input"] for v in row
    ))

    # True only if at least one training example introduces a color not in its own input
    examples_introduce_colors = any(len(s["colors_added"]) > 0 for s in pair_stats)

    # Max number of disconnected non-black regions seen in any training output
    max_example_out_regions = max((s["out_regions"] for s in pair_stats), default=1)

    return {
        "pairs":                    pair_stats,
        "all_size_preserved":       all_size_preserved,
        "avg_pct_changed":          avg_pct,
        "congruent":                congruent,
        "allowed_colors":           allowed_colors,
        "test_input_colors":        test_input_colors,
        "examples_introduce_colors": examples_introduce_colors,
        "max_example_out_regions":  max_example_out_regions,
    }


def build_prompt(puzzle: dict) -> str:
    """
    Build a structured prompt that includes:
    - Hard constraints (size, color palette)
    - Transformation statistics per training pair (% changed, congruence)
    - Explicit chain-of-thought instruction before outputting the grid
    """
    stats = analyze_puzzle(puzzle)
    test_input = puzzle["test"][0]["input"]
    rows, cols = len(test_input), len(test_input[0]) if test_input else 0

    lines = []
    lines.append("You are solving an ARC (Abstraction and Reasoning Corpus) puzzle.")
    lines.append("Each cell is a digit 0–9 (color), one row per line, no spaces.")
    lines.append("")

    lines.append("HARD CONSTRAINTS — violations mean the answer is automatically wrong:")
    lines.append(f"  1. Output must be EXACTLY {rows}×{cols} (same dimensions as Test Input)")
    lines.append("  COLOR COUNT RULES (derived from training examples — must be obeyed):")
    lines.append(CF.signature_hint(puzzle["train"]))
    if not stats["examples_introduce_colors"]:
        lines.append(f"  2. NEVER introduce a color that is not already in the Test Input. "
                     f"Allowed colors: {stats['test_input_colors']}. "
                     f"None of the training examples introduce a new color, so neither should you.")
    else:
        lines.append(f"  2. Output may only use these colors: {stats['allowed_colors']}")
    lines.append("  3. The same transformation rule must apply consistently to ALL examples")
    if stats["max_example_out_regions"] == 1:
        lines.append("  4. The output must be a single connected region — do NOT produce two or more "
                     "separate disconnected patches. None of the training outputs have multiple "
                     "disconnected regions.")
    else:
        lines.append(f"  4. Training outputs contain up to {stats['max_example_out_regions']} "
                     f"disconnected regions — match that pattern.")
    lines.append("")

    lines.append("TRANSFORMATION STATISTICS (computed from examples):")
    if stats["all_size_preserved"]:
        lines.append("  - Size-preserving: all example outputs are the same shape as their inputs")
    if stats["avg_pct_changed"] is not None:
        note = " (consistent)" if stats["congruent"] else " (varies — look carefully)"
        lines.append(f"  - ~{stats['avg_pct_changed']}% of cells change per example{note}")
    for i, s in enumerate(stats["pairs"]):
        detail = f"  - Example {i+1}: {s['in_shape']}→{s['out_shape']}"
        if s["pct_changed"] is not None:
            detail += f", {s['pct_changed']}% cells changed"
        if s["colors_added"]:
            detail += f", new colors introduced: {s['colors_added']}"
        if s["colors_removed"]:
            detail += f", colors removed: {s['colors_removed']}"
        lines.append(detail)
    lines.append("")

    lines.append("EXAMPLES:")
    for i, ex in enumerate(puzzle["train"]):
        lines.append(f"Input {i+1}:\n{grid_to_str(ex['input'])}")
        lines.append(f"Output {i+1}:\n{grid_to_str(ex['output'])}\n")

    lines.append(f"Test Input:\n{grid_to_str(test_input)}")
    lines.append("")
    lines.append("INSTRUCTIONS:")
    lines.append("  Step 1: Identify the transformation rule from ALL examples.")
    lines.append("  Step 2: Verify your rule produces the correct output for each example.")
    lines.append("  Step 3: Apply the rule to the Test Input.")
    lines.append(f"  Step 4: Output ONLY the {rows}×{cols} result grid — digits only, one row per line.")
    lines.append("          The output MUST differ from the Test Input by at least one cell.")
    lines.append("          Do not include any explanation, labels, or extra text after the grid.")

    return "\n".join(lines)


def build_classify_prompt(puzzle: dict) -> str:
    """
    Short prompt asking Qwen to classify puzzle type and operation count.
    Designed for fast inference (small output ~50 tokens).
    """
    lines = []
    lines.append("Analyze this ARC puzzle. Reply in EXACTLY this format — nothing else:")
    lines.append("TYPE: <geometric|color-mapping|object-manipulation|pattern-completion|counting|compositional>")
    lines.append("OPS: <integer 1-5, how many distinct operations are needed>")
    lines.append("HIDDEN: <yes|no, whether invisible steps like counting/measuring are required>")
    lines.append("CONFIDENCE: <low|medium|high>")
    lines.append("REASONING: <one sentence>")
    lines.append("")
    lines.append("EXAMPLES:")
    for i, ex in enumerate(puzzle["train"]):
        lines.append(f"Input {i+1}:\n{grid_to_str(ex['input'])}")
        lines.append(f"Output {i+1}:\n{grid_to_str(ex['output'])}\n")
    lines.append(f"Test Input:\n{grid_to_str(puzzle['test'][0]['input'])}")
    return "\n".join(lines)


def parse_classification(text: str) -> dict:
    """Parse structured classification output into a dict."""
    result = {"type": "unknown", "ops": 3, "hidden": False, "confidence": "low", "reasoning": ""}
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("TYPE:"):
            result["type"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("OPS:"):
            try: result["ops"] = int(line.split(":", 1)[1].strip()[0])
            except: pass
        elif line.startswith("HIDDEN:"):
            result["hidden"] = "yes" in line.lower()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
    return result


def classify_puzzle(pid, puzzle, model, tok, device) -> dict:
    """Run a fast classification pass on one puzzle. Returns classification dict."""
    prompt = build_classify_prompt(puzzle)
    # Small token budget — just need ~50 tokens for the structured reply
    grid = _generate(prompt, model, tok, device, token_budget=128, do_sample=False)
    # _generate parses grids; we need raw text instead
    try:
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(messages, tokenize=True,
                                      add_generation_prompt=True, return_tensors="pt")
        if isinstance(enc, dict):
            enc = {k: v.to(device) for k, v in enc.items()}
            input_ids = enc["input_ids"]
        else:
            input_ids = enc.to(device)
    except Exception:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    import torch
    with torch.inference_mode():
        out = model.generate(input_ids, max_new_tokens=128, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    result = parse_classification(text)
    result["puzzle_id"] = pid
    print(f"[Qwen] {pid} → type={result['type']} ops={result['ops']} hidden={result['hidden']} conf={result['confidence']}", flush=True)
    return result


def build_variation_prompt(puzzle: dict, prev_grid: list) -> str:
    """
    Prompt for attempt_2+: given the previous attempt, produce a variation.
    Uses the same transformation logic but tries a slight different interpretation.
    """
    stats = analyze_puzzle(puzzle)
    test_input = puzzle["test"][0]["input"]
    rows, cols = len(test_input), len(test_input[0]) if test_input else 0

    lines = []
    lines.append("You are solving an ARC puzzle. A previous attempt is shown below.")
    lines.append("Produce a VARIATION — same transformation logic, but with a slightly different interpretation.")
    lines.append("")
    lines.append("CONSTRAINTS:")
    lines.append(f"  1. Output must be EXACTLY {rows}×{cols}")
    if not stats["examples_introduce_colors"]:
        lines.append(f"  2. NEVER introduce a color not already in the Test Input. "
                     f"Allowed: {stats['test_input_colors']}.")
    else:
        lines.append(f"  2. Output may only use colors: {stats['allowed_colors']}")
    lines.append("  3. Output must differ from the Previous Attempt by at least one cell")
    if stats["max_example_out_regions"] == 1:
        lines.append("  4. Output must be a single connected region — no separate disconnected patches.")
    lines.append("")
    lines.append("EXAMPLES:")
    for i, ex in enumerate(puzzle["train"]):
        lines.append(f"Input {i+1}:\n{grid_to_str(ex['input'])}")
        lines.append(f"Output {i+1}:\n{grid_to_str(ex['output'])}\n")
    lines.append(f"Test Input:\n{grid_to_str(test_input)}")
    lines.append(f"\nPrevious Attempt:\n{grid_to_str(prev_grid)}")
    lines.append("")
    lines.append("Apply a slight variation of the same rule.")
    lines.append(f"Output ONLY the {rows}×{cols} result grid — digits only, one row per line, no labels.")

    return "\n".join(lines)


# ── Single-puzzle inference ────────────────────────────────────────────────────

def _generate(prompt, model, tok, device, token_budget, do_sample=False, temperature=0.8):
    """Tokenize prompt, generate, decode, parse grid. Returns grid or None."""
    import torch
    try:
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(messages, tokenize=True,
                                      add_generation_prompt=True, return_tensors="pt")
        if isinstance(enc, dict):
            enc = {k: v.to(device) for k, v in enc.items()}
            input_ids = enc["input_ids"]
        else:
            input_ids = enc.to(device)
    except Exception:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

    with torch.inference_mode():
        out = model.generate(
            input_ids, max_new_tokens=token_budget,
            do_sample=do_sample, temperature=temperature if do_sample else 1.0,
            top_p=0.95, pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return str_to_grid(text)


def _min_perturb(grid, ref_grid):
    """Flip the first cell that differs from ref, or first cell if identical. Returns new grid."""
    g = [row[:] for row in grid]
    for r in range(len(g)):
        for c in range(len(g[r])):
            ref_val = ref_grid[r][c] if r < len(ref_grid) and c < len(ref_grid[r]) else -1
            if g[r][c] == ref_val:
                g[r][c] = (g[r][c] + 1) % 10
                return g
    if g: g[0][0] = (g[0][0] + 1) % 10
    return g


def infer_puzzle(pid, puzzle, model, tok, n_attempts, max_new_tokens, device,
                 stop_flag=None):
    """
    Generate up to n_attempts grids for one puzzle.

    Slot layout:
      attempt_1  — copy of test input (baseline; establishes the starting point)
      attempt_2  — first LLM transformation (greedy, must change ≥1 cell)
      attempt_3+ — sampling variations on the previous attempt

    n_attempts can be:
      int  0-12  — fixed count
      None       — variable mode: run until stop_flag is set or 100 attempts reached

    stop_flag: threading.Event or None — set externally (SIGINT) to stop early.
    """
    test_input = puzzle["test"][0]["input"]
    rows = len(test_input)
    cols = len(test_input[0]) if test_input else 1
    token_budget = max(max_new_tokens, rows * (cols + 1) + 512)

    max_a = 100 if n_attempts is None else n_attempts
    results = []

    if max_a == 0:
        return results

    # ── Attempt 1: copy of test input (baseline) ────────────────────────────────
    results.append([row[:] for row in test_input])
    print(f"[Qwen] {pid} attempt 1 → baseline (copy of input)", flush=True)

    # ── Attempt 2: first LLM pass (greedy) ──────────────────────────────────────
    if max_a >= 2 and not (stop_flag and stop_flag.is_set()):
        grid2 = _generate(build_prompt(puzzle), model, tok, device,
                          token_budget, do_sample=False)
        if grid2 is None or grids_equal(grid2, test_input):
            grid2 = _min_perturb([row[:] for row in test_input], test_input)
        results.append(grid2)
        print(f"[Qwen] {pid} attempt 2/{max_a} → {len(grid2)}×{len(grid2[0])} (LLM greedy)", flush=True)

    # ── Attempts 3+: sampling variations ────────────────────────────────────────
    attempt_num = 3
    while attempt_num <= max_a:
        if stop_flag and stop_flag.is_set():
            print(f"[Qwen] {pid} stopped early at attempt {len(results)}", flush=True)
            break
        prev = results[-1]
        grid = _generate(build_variation_prompt(puzzle, prev), model, tok, device,
                         token_budget, do_sample=True, temperature=0.8)
        if grid is None or grids_equal(grid, prev):
            grid = _min_perturb([row[:] for row in prev], prev)
        results.append(grid)
        print(f"[Qwen] {pid} attempt {attempt_num}/{max_a} → {len(grid)}×{len(grid[0])}", flush=True)
        attempt_num += 1

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3 ARC inference")
    parser.add_argument("--challenges",    required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--model",         default="Qwen/Qwen3-4B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--lora",          default=None,
                        help="Path to LoRA adapter (optional, e.g. ARChitects weights)")
    parser.add_argument("--ids",           nargs="*",
                        help="Specific puzzle IDs (default: all)")
    parser.add_argument("--quant",         default="auto",
                        choices=["auto", "4bit", "8bit", "none"],
                        help="Quantization (auto=detect by VRAM)")
    parser.add_argument("--attempts",      default="3",
                        help="Attempts per puzzle: int 0-12, or 'variable' (run until interrupted, max 100). "
                             "Attempt 1 is always a copy of the input (baseline).")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Minimum token budget for generation (auto-scales with grid size)")
    parser.add_argument("--classify-only", action="store_true",
                        help="Run classification pass only — no inference. Saves classifications.json.")
    parser.add_argument("--sort-by-difficulty", action="store_true",
                        help="Classify first, then solve easiest (ops=1) to hardest (ops=5).")
    args = parser.parse_args()

    # Detect hardware
    hw = detect_hardware()
    quant = "none" if args.quant == "none" else args.quant

    # Check bitsandbytes available for quantization
    if quant in ("4bit", "8bit") or (quant == "auto" and hw["quant"]):
        try:
            import bitsandbytes
        except ImportError:
            print("[Qwen] WARNING: bitsandbytes not installed — falling back to float16 full precision", flush=True)
            print("[Qwen]   Install with: pip install bitsandbytes", flush=True)
            quant = "none"
            if hw["vram_gb"] < 8:
                print("[Qwen] WARNING: VRAM may be insufficient without quantization", flush=True)

    # Load challenges
    with open(args.challenges) as f:
        all_challenges = json.load(f)

    ids = args.ids if args.ids else list(all_challenges.keys())
    challenges = {k: all_challenges[k] for k in ids if k in all_challenges}
    print(f"[Qwen] Puzzles: {len(challenges)} | model: {args.model} | quant: {quant} | attempts: {args.attempts}", flush=True)

    # Load model
    model, tok = load_model_and_tokenizer(args.model, args.lora, hw, quant)
    device = next(model.parameters()).device

    # Classification pass (always runs if --classify-only or --sort-by-difficulty)
    classifications = {}
    if args.classify_only or args.sort_by_difficulty:
        print(f"[Qwen] Classifying {len(challenges)} puzzles…", flush=True)
        for pid, puzzle in challenges.items():
            classifications[pid] = classify_puzzle(pid, puzzle, model, tok, device)
        clf_path = Path(args.output).with_name(
            Path(args.output).stem + "_classifications.json")
        clf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(clf_path, "w") as f:
            json.dump(classifications, f, indent=2)
        print(f"[Qwen] Classifications saved to: {clf_path}", flush=True)
        if args.classify_only:
            return

    # Parse attempts: int or "variable"
    variable_mode = (args.attempts.strip().lower() == "variable")
    if variable_mode:
        n_attempts = None   # signals variable mode to infer_puzzle
        print("[Qwen] Variable mode: up to 100 attempts per puzzle, Ctrl+C to stop cleanly", flush=True)
    else:
        try:
            n_attempts = max(0, min(100, int(args.attempts)))
        except ValueError:
            print(f"[Qwen] WARNING: invalid --attempts '{args.attempts}', defaulting to 3", flush=True)
            n_attempts = 3

    # SIGINT (Ctrl+C) handler — sets flag so current puzzle finishes its batch then saves
    import threading
    stop_flag = threading.Event()
    _orig_sigint = signal.getsignal(signal.SIGINT)
    def _handle_sigint(sig, frame):
        if not stop_flag.is_set():
            print("\n[Qwen] Interrupt received — finishing current puzzle then saving...", flush=True)
            stop_flag.set()
        else:
            print("\n[Qwen] Second interrupt — force exit", flush=True)
            signal.signal(signal.SIGINT, _orig_sigint)
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _handle_sigint)

    # Sort by estimated difficulty if requested
    puzzle_order = list(challenges.items())
    if args.sort_by_difficulty and classifications:
        puzzle_order.sort(key=lambda kv: classifications.get(kv[0], {}).get("ops", 3))
        print("[Qwen] Solving easiest → hardest by estimated op count", flush=True)

    # Run inference — save incrementally so Ctrl+C preserves completed puzzles
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission = {}

    for i, (pid, puzzle) in enumerate(puzzle_order):
        if stop_flag.is_set():
            print(f"[Qwen] Stopping before puzzle {pid} — saving {len(submission)} results", flush=True)
            break
        clf = classifications.get(pid, {})
        clf_note = f" [type={clf.get('type','?')} ops={clf.get('ops','?')}]" if clf else ""
        print(f"[Qwen] [{i+1}/{len(puzzle_order)}] {pid}{clf_note}", flush=True)

        grids = infer_puzzle(pid, puzzle, model, tok, n_attempts, args.max_new_tokens, device,
                             stop_flag=stop_flag)

        # Build attempt dict — all attempts saved, not just top 2
        attempt_dict = {f"attempt_{j+1}": g for j, g in enumerate(grids)}

        # Score every attempt with constraint filter
        test_input = puzzle["test"][0]["input"]
        scores = CF.score_all_attempts(attempt_dict, test_input, puzzle["train"])
        attempt_dict["_constraint_scores"] = scores

        submission[pid] = [attempt_dict]

        # Rank and log best attempt
        ranked = CF.rank_attempts(grids, test_input, puzzle["train"])
        if ranked:
            best_v, best_c = ranked[0][1], ranked[0][2]
            best_idx = grids.index(ranked[0][0]) + 1
            print(f"[Qwen]   best: attempt_{best_idx} — violations={best_v}, changes={best_c}", flush=True)

        # Incremental save so nothing is lost on interrupt
        with open(out, "w") as f:
            json.dump(submission, f)

    print(f"[Qwen] Done. {len(submission)} puzzles saved to: {out}", flush=True)


if __name__ == "__main__":
    main()
