"""Fetch outputs / log for a specific older kernel version, since
`kaggle kernels output` only fetches the latest. Bypasses the cp1252 issue too.
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
out_dir = Path(sys.argv[3])
version_number = int(sys.argv[4]) if len(sys.argv) > 4 else None

out_dir.mkdir(parents=True, exist_ok=True)

api = KaggleApi()
api.authenticate()

with api.build_kaggle_client() as kaggle:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = owner
    req.kernel_slug = slug
    if version_number is not None:
        req.version_number = version_number
    response = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)

# Save log
log = response.log or ""
log_path = out_dir / f"{slug}_v{version_number or 'latest'}.log"
log_path.write_text(log, encoding="utf-8")
print(f"Wrote log: {log_path} ({len(log)} chars)")

# List files
import requests

for item in response.files or []:
    name = item.file_name
    target = out_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {name}")
    r = requests.get(item.url, stream=True)
    target.write_bytes(r.content)
print(f"Saved {len(response.files or [])} files to {out_dir}")
