#!/usr/bin/env python3
"""Serve the live demo on http://localhost:8000 (microphone access is allowed on localhost).

Same idea as the server.py in Edge Impulse's WebAssembly browser guide: make sure .wasm files
are sent as application/wasm, and disable caching so a rebuilt model is picked up.

  python demo/serve.py            # then open http://localhost:8000
"""
import http.server
import socketserver
from pathlib import Path

PORT = 8000
ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".js": "text/javascript", ".json": "application/json"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"Live demo: http://localhost:{PORT}  (Ctrl+C to stop)")
        httpd.serve_forever()
