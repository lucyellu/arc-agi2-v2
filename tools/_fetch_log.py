import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from kaggle import KaggleApi

api = KaggleApi()
api.authenticate()

owner, slug = "lucyellu", "separated172-36a08778"
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
out_dir.mkdir(parents=True, exist_ok=True)

api.kernels_output_cli(kernel=f"{owner}/{slug}", path=str(out_dir), force=True, quiet=True)
