"""Fetch a Kaggle kernel's log via the lower-level API.

Workaround for Kaggle CLI 2.1.0 bug on Windows: kernels_output() writes the
log via open(..., "w") which defaults to cp1252 and crashes on the unsloth
sloth emoji.
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from kaggle import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
)

owner = sys.argv[1]
slug = sys.argv[2]
out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"{slug}.log")
out_path.parent.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()

with api.build_kaggle_client() as kaggle:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = owner
    req.kernel_slug = slug
    response = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)

log = response.log or ""
out_path.write_text(log, encoding="utf-8")
print(f"Wrote {out_path}: {len(log)} chars, {out_path.stat().st_size} bytes")
