"""
Puzzle complexity / difficulty scorer for ARC-AGI.

compute_puzzle_complexity(puzzle_data) -> int
  Lower score = easier.

Factors
-------
- max grid area (rows x cols) across all training + test grids
- number of unique colors across all training examples
- size-change penalty (output size differs from input = harder)
- training-example scarcity penalty (fewer examples = harder)
"""


def _grid_area(grid):
    rows = len(grid)
    cols = max((len(r) for r in grid), default=0) if rows else 0
    return rows * cols


def compute_puzzle_complexity(puzzle_data):
    """Return an integer complexity score; lower = easier."""
    examples = puzzle_data.get("train", [])
    tests    = puzzle_data.get("test",  [])

    max_area     = 0
    all_cells    = []
    size_changes = []

    for ex in examples:
        inp, out = ex["input"], ex["output"]
        a_in  = _grid_area(inp)
        a_out = _grid_area(out)
        max_area = max(max_area, a_in, a_out)
        size_changes.append(a_in != a_out)
        for row in inp:
            all_cells.extend(int(v) for v in row)
        for row in out:
            all_cells.extend(int(v) for v in row)

    for t in tests:
        inp = t["input"]
        max_area = max(max_area, _grid_area(inp))
        for row in inp:
            all_cells.extend(int(v) for v in row)

    num_colors          = len(set(v for v in all_cells if 0 <= v <= 9))
    size_change_penalty = 50 if any(size_changes) else 0
    train_penalty       = max(0, (3 - len(examples))) * 20

    score = max_area + num_colors * 3 + size_change_penalty + train_penalty
    return score
