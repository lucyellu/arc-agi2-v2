#!/usr/bin/env python3
"""
ARC constraint filter â€” color delta analysis and attempt scoring.

Core idea: count cells per color (0-9) in training input vs output.
If the delta direction is consistent across ALL examples, that becomes a hard
constraint. Any attempt that violates it is almost certainly wrong.

This eliminates a large fraction of wrong answers before any LLM reasoning
and narrows the search space for both TRM and Qwen3.
"""


def count_colors(grid):
    """Return dict {color_id: cell_count} for colors 0-9."""
    counts = {c: 0 for c in range(10)}
    for row in grid:
        for val in row:
            v = int(val)
            if 0 <= v <= 9:
                counts[v] += 1
    return counts


def grid_size(grid):
    """Return (rows, cols). Returns (0, 0) for empty grids."""
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0) if rows else 0
    return rows, cols


def _pixel_sim_grids(grid_a, grid_b):
    """Fraction of cells at same position with same value. Returns None if shapes differ."""
    ra, rb = len(grid_a), len(grid_b)
    if ra != rb or ra == 0:
        return None
    total = matching = 0
    for r in range(ra):
        if len(grid_a[r]) != len(grid_b[r]):
            return None
        for c in range(len(grid_a[r])):
            total += 1
            if int(grid_a[r][c]) == int(grid_b[r][c]):
                matching += 1
    return matching / total if total > 0 else 0.0


def min_pixel_sim_to_input(examples):
    """Min fraction of cells preserved from input→output across training pairs.
    Returns None if any pair has mismatched shapes (size-changing transformation)."""
    sims = []
    for ex in examples:
        s = _pixel_sim_grids(ex['output'], ex['input'])
        if s is None:
            return None
        sims.append(s)
    return min(sims) if sims else None


def analyze_color_signature(examples):
    """
    Derive a color change signature from training examples.

    For each color 0-9, determines whether its cell count:
      'zero'         â€” never changes (most common, very constraining)
      'increase'     â€” always goes up
      'decrease'     â€” always goes down
      'nondecreasing'â€” sometimes stays same, sometimes increases
      'nonincreasing'â€” sometimes stays same, sometimes decreases
      'inconsistent' â€” direction varies; don't constrain this color

    Also returns size_change: 'same'|'increase'|'decrease'|'inconsistent'

    Returns: (signature dict, size_change str)
    """
    color_deltas = {c: [] for c in range(10)}
    size_deltas = []

    for ex in examples:
        inp, out = ex['input'], ex['output']
        in_counts  = count_colors(inp)
        out_counts = count_colors(out)
        for c in range(10):
            color_deltas[c].append(out_counts[c] - in_counts[c])
        in_r, in_c   = grid_size(inp)
        out_r, out_c = grid_size(out)
        size_deltas.append((out_r * out_c) - (in_r * in_c))

    signature = {}
    for c, ds in color_deltas.items():
        signs = set()
        for d in ds:
            if d > 0:    signs.add(1)
            elif d < 0:  signs.add(-1)
            else:        signs.add(0)
        if signs == {0}:
            signature[c] = 'zero'
        elif signs == {1}:
            signature[c] = 'increase'
        elif signs == {-1}:
            signature[c] = 'decrease'
        elif signs == {0, 1}:
            signature[c] = 'nondecreasing'
        elif signs == {0, -1}:
            signature[c] = 'nonincreasing'
        else:
            signature[c] = 'inconsistent'

    size_signs = set()
    for d in size_deltas:
        if d > 0:    size_signs.add(1)
        elif d < 0:  size_signs.add(-1)
        else:        size_signs.add(0)

    if size_signs == {0}:
        size_change = 'same'
    elif size_signs == {1}:
        size_change = 'increase'
    elif size_signs == {-1}:
        size_change = 'decrease'
    else:
        size_change = 'inconsistent'

    return signature, size_change


def score_attempt(attempt, test_input, signature, size_change, min_pix_sim=None):
    """
    Count how many constraint rules this attempt violates.

    Returns: (violation_score, detail_list)
      violation_score â€” weighted count; 0 = fully valid
      detail_list     â€” human-readable descriptions of each violation

    Violations are weighted by magnitude (e.g. 5 extra cells of a 'zero'
    color scores 5, not 1) so larger errors are ranked worse.
    """
    if attempt is None:
        return 999, ['attempt is None']

    inp_counts = count_colors(test_input)
    out_counts = count_colors(attempt)
    violations = 0
    details = []

    _names = ['black', 'blue', 'red', 'green', 'yellow',
              'grey', 'magenta', 'orange', 'azure', 'maroon']

    for c, rule in signature.items():
        d = out_counts[c] - inp_counts[c]
        name = _names[c]
        if rule == 'zero' and d != 0:
            violations += abs(d)
            details.append(f'color {c} ({name}): must stay unchanged, changed by {d:+d}')
        elif rule == 'increase' and d <= 0:
            violations += 1
            details.append(f'color {c} ({name}): must increase, changed by {d:+d}')
        elif rule == 'decrease' and d >= 0:
            violations += 1
            details.append(f'color {c} ({name}): must decrease, changed by {d:+d}')
        elif rule == 'nondecreasing' and d < 0:
            violations += 1
            details.append(f'color {c} ({name}): must not decrease, changed by {d:+d}')
        elif rule == 'nonincreasing' and d > 0:
            violations += 1
            details.append(f'color {c} ({name}): must not increase, changed by {d:+d}')

    # Size constraint
    in_r, in_c   = grid_size(test_input)
    out_r, out_c = grid_size(attempt)
    size_d = (out_r * out_c) - (in_r * in_c)
    if size_change == 'same' and size_d != 0:
        violations += abs(size_d)
        details.append(f'size: must stay {in_r}x{in_c}, got {out_r}x{out_c}')
    elif size_change == 'increase' and size_d <= 0:
        violations += 1
        details.append(f'size: must increase, but {out_r}x{out_c} <= {in_r}x{in_c}')
    elif size_change == 'decrease' and size_d >= 0:
        violations += 1
        details.append(f'size: must decrease, but {out_r}x{out_c} >= {in_r}x{in_c}')

    # Pixel overlap with input: if training always preserves some cells, 0.0% is impossible
    if min_pix_sim is not None and min_pix_sim > 0:
        att_sim = _pixel_sim_grids(attempt, test_input)
        if att_sim is not None and att_sim == 0.0:
            violations += 5
            details.append(f'pixel_sim_to_input=0.0 (training min={min_pix_sim:.3f})')

    return violations, details


def count_changed_cells(attempt, test_input):
    """
    Count cells that differ between attempt and test_input.
    Used as a tie-breaker: prefer fewer changes (inaction bias).
    """
    if attempt is None:
        return 999
    changed = 0
    for r in range(min(len(attempt), len(test_input))):
        for c in range(min(len(attempt[r]), len(test_input[r]))):
            if attempt[r][c] != test_input[r][c]:
                changed += 1
    return changed


def rank_attempts(attempts, test_input, examples):
    """
    Score and rank a list of candidate grids.

    Returns list of (grid, violation_score, changed_cells, details) sorted
    best-first. Ties in violation_score broken by fewest changed cells
    (inaction bias: do the minimum necessary).
    """
    signature, size_change = analyze_color_signature(examples)
    min_ps = min_pixel_sim_to_input(examples)
    scored = []
    for g in attempts:
        v, details = score_attempt(g, test_input, signature, size_change, min_pix_sim=min_ps)
        changes = count_changed_cells(g, test_input)
        scored.append((g, v, changes, details))
    scored.sort(key=lambda x: (x[1], x[2]))
    return scored


def score_all_attempts(attempt_dict, test_input, examples):
    """
    Score every attempt in an attempt dict (e.g. {"attempt_1": grid, ...}).
    Returns a dict of the same keys: {"attempt_1": {"violations": N, "details": [...]}, ...}
    Skips non-attempt keys (e.g. _constraint_scores).
    """
    signature, size_change = analyze_color_signature(examples)
    min_ps = min_pixel_sim_to_input(examples)
    result = {}
    for key, grid in attempt_dict.items():
        if not key.startswith('attempt_'):
            continue
        v, details = score_attempt(grid, test_input, signature, size_change, min_pix_sim=min_ps)
        changes = count_changed_cells(grid, test_input)
        result[key] = {
            'violations': v,
            'changed_cells': changes,
            'details': details,
        }
    return result


def signature_hint(examples):
    """
    Build a human-readable string describing the inferred color change rules.
    Inject this into LLM prompts to hard-constrain the search space.
    """
    signature, size_change = analyze_color_signature(examples)
    _names = ['black', 'blue', 'red', 'green', 'yellow',
              'grey', 'magenta', 'orange', 'azure', 'maroon']
    parts = []
    for c, rule in sorted(signature.items()):
        if rule == 'inconsistent':
            continue
        # Only mention colors that actually appear in training
        name = _names[c]
        if rule == 'zero':
            parts.append(f"color {c} ({name}): cell count must stay exactly the same")
        elif rule == 'increase':
            parts.append(f"color {c} ({name}): cell count must increase")
        elif rule == 'decrease':
            parts.append(f"color {c} ({name}): cell count must decrease")
        elif rule == 'nondecreasing':
            parts.append(f"color {c} ({name}): cell count must not decrease")
        elif rule == 'nonincreasing':
            parts.append(f"color {c} ({name}): cell count must not increase")
    if size_change != 'inconsistent':
        parts.append(f"grid size: must {'stay the same' if size_change == 'same' else size_change}")
    if not parts:
        return "  (no consistent color pattern detected across examples)"
    return '\n'.join(f"  - {p}" for p in parts)
