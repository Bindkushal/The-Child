#!/usr/bin/env python3
"""
brain/server.py
===============
Run this to open the SENN Brain visualization in your browser.

Usage:
  Terminal 1:  python brain/server.py     ← starts viewer
  Terminal 2:  python train.py            ← starts training

The visualizer auto-updates every 500ms from live_state.json
"""

import http.server
import socketserver
import json
import os
import webbrowser
import threading
from pathlib import Path

PORT = 7700
BRAIN_DIR  = Path(__file__).parent
# Look for live_state.json in current dir or parent
STATE_PATHS = [
    Path.cwd() / "live_state.json",
    BRAIN_DIR.parent / "live_state.json",
    BRAIN_DIR / "live_state.json",
]


class BrainHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BRAIN_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            self._serve_state()
        else:
            super().do_GET()

    def _serve_state(self):
        data = "{}"
        for p in STATE_PATHS:
            if p.exists():
                try:
                    data = p.read_text()
                    break
                except Exception:
                    pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, fmt, *args):
        pass  # silence HTTP logs


def run():
    os.chdir(BRAIN_DIR)
    with socketserver.TCPServer(("", PORT), BrainHandler) as httpd:
        httpd.allow_reuse_address = True
        url = f"http://localhost:{PORT}/brain.html"
        print(f"\n  🧠  SENN BRAIN  →  {url}")
        print(f"  Now run training in another terminal:  python train.py\n")
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Brain stopped.")


if __name__ == "__main__":
    run()