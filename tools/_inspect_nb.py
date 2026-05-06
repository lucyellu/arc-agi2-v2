import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
path = Path(sys.argv[1])
nb = json.loads(path.read_text(encoding="utf-8"))
print(f"# {path}")
print(f"cells: {len(nb['cells'])}")
print(f"metadata: {json.dumps(nb.get('metadata', {}), indent=2)[:600]}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    print(f"\n--- cell {i} ({c['cell_type']}) [{len(src)} chars] ---")
    head = src[:1200]
    print(head)
    if len(src) > 1200:
        print(f"... [+{len(src) - 1200} more chars]")
