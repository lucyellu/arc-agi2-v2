"""Bypass kaggle CLI's broken Windows encoding handling for the log fetch."""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from kaggle import KaggleApi

api = KaggleApi()
api.authenticate()

owner, slug = "lucyellu", "separated172-36a08778"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kernel.log")
out.parent.mkdir(parents=True, exist_ok=True)

# kernels_output is the lower-level call. The CLI version writes the log via
# text-mode open() which crashes on the unsloth sloth emoji under cp1252.
# Here we ask only for the log content and write it as bytes ourselves.
response = api.kernels_output(f"{owner}/{slug}", str(out.parent), force=True, quiet=True)
# But kernels_output also writes through python text I/O — fall back to the
# direct API call to fetch the kernel output from process_response.
log_text = api.kernels_output_cli  # method shadow probe
print(f"out path: {out}")
# Actually the simplest path: kernels_output returns dict-like; the CLI uses
# api.process_response(api.kernels_output_with_http_info(...)). Let's get
# the raw log via api.kernels_log_cli or similar.
# Try kernels_status and look for log_url.

# Easiest: kernels_output() saves files and returns a (files, log) pair;
# the CLI then writes log to disk. Since the CLI write fails, just iterate
# api.kernels_output() return.
# Inspect what kernels_output gave us:
print(f"return type: {type(response)}")
if isinstance(response, tuple):
    files, log = response
    print(f"files: {len(files)}  log_len: {len(log) if log else 0}")
    out.write_bytes(log.encode("utf-8", errors="replace") if isinstance(log, str) else log)
    print(f"wrote log to {out} ({out.stat().st_size} bytes)")
else:
    print(f"unexpected return shape: {response}")
