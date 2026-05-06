import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
nb_path = Path(sys.argv[1])
indices = [int(a) for a in sys.argv[2:]] if len(sys.argv) > 2 else None
nb = json.loads(nb_path.read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if indices is not None and i not in indices:
        continue
    src = "".join(c["source"])
    print(f"### cell {i} ({c['cell_type']}) [{len(src)} chars] ###")
    print(src)
    print()
