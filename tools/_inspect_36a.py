import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
chal = json.loads((ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_challenges.json").read_text())
sol = json.loads((ROOT / "data" / "arc-prize-2026-arc-agi-2" / "arc-agi_evaluation_solutions.json").read_text())

p = chal["36a08778"]
s = sol["36a08778"]
print(f"train pairs: {len(p['train'])}")
for i, ex in enumerate(p["train"]):
    inp, out = ex["input"], ex["output"]
    print(f"  train[{i}]: in={len(inp)}x{len(inp[0])}  out={len(out)}x{len(out[0])}")
print(f"test inputs: {len(p['test'])}  solutions: {len(s)}")
for i, (t, g) in enumerate(zip(p["test"], s)):
    print(f"  test[{i}]: in={len(t['input'])}x{len(t['input'][0])}  out={len(g)}x{len(g[0])}")

# colors used
def colors(grid):
    return sorted({v for r in grid for v in r})
print(f"train[0] in colors: {colors(p['train'][0]['input'])}  out colors: {colors(p['train'][0]['output'])}")
print(f"test[0] in colors:  {colors(p['test'][0]['input'])}  sol colors: {colors(s[0])}")
print(f"test[1] in colors:  {colors(p['test'][1]['input'])}  sol colors: {colors(s[1])}")

# any size change?
for i, ex in enumerate(p["train"]):
    inp, out = ex["input"], ex["output"]
    sc = (len(inp) != len(out)) or (len(inp[0]) != len(out[0]))
    print(f"  train[{i}] size_change={sc}")
