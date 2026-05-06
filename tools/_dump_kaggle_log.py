import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
log = Path(sys.argv[1])
text = log.read_text(encoding="utf-8")
data = json.loads(text)
for entry in data:
    t = entry["time"]
    s = entry["stream_name"]
    d = entry["data"].rstrip("\n")
    print(f"{t:7.2f} [{s}] {d}")
