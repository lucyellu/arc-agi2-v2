"""
Simple HTTP server for run_viewer.html with /api/list-latest endpoint.
Serves static files from cwd + exposes folder listing for auto-load.
"""
import http.server
import json
import os
import urllib.parse
from pathlib import Path

PORT = 8080
TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
LATEST_DIR = ROOT / "results" / "submissions" / "latest"
DATA_DIR = ROOT / "data" / "arc-prize-2026-arc-agi-2"

# Extra files always bundled alongside the run outputs (challenges, solutions)
BUNDLED_FILES = [
    DATA_DIR / "arc-agi_evaluation_challenges.json",
    DATA_DIR / "arc-agi_evaluation_solutions.json",
]

class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress request logs

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/list-latest":
            self._serve_latest_listing()
        elif parsed.path.startswith("/api/file-bundled/"):
            name = urllib.parse.unquote(parsed.path[len("/api/file-bundled/"):])
            self._serve_file(DATA_DIR / name)
        elif parsed.path.startswith("/api/file/"):
            rel = urllib.parse.unquote(parsed.path[len("/api/file/"):])
            self._serve_file(LATEST_DIR / rel)
        else:
            super().do_GET()

    def _serve_latest_listing(self):
        files = []
        # Bundled challenge/solution files first (so they load before run data)
        for bf in BUNDLED_FILES:
            if bf.exists():
                files.append({"name": bf.name, "size": bf.stat().st_size, "src": "bundled"})
        # Run outputs from latest/
        if LATEST_DIR.exists():
            for f in sorted(LATEST_DIR.iterdir()):
                if f.is_file() and f.suffix == ".json" and f.stat().st_size > 0:
                    files.append({"name": f.name, "size": f.stat().st_size, "src": "latest"})
        self._json_response({"files": files})

    def _file_path_for(self, name: str, src: str) -> Path:
        if src == "bundled":
            return DATA_DIR / name
        return LATEST_DIR / name

    def _serve_file(self, path: Path):
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    os.chdir(TOOLS_DIR)
    with http.server.HTTPServer(("", PORT), ViewerHandler) as httpd:
        print(f"Serving on http://localhost:{PORT}")
        httpd.serve_forever()
