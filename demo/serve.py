#!/usr/bin/env python3
"""Serve the live demo on http://localhost:8000 (microphone access is allowed on localhost).

Same idea as the server.py in Edge Impulse's WebAssembly browser guide: make sure .wasm files
are sent as application/wasm, and disable caching so a rebuilt model is picked up.

  python demo/serve.py            # then open the printed address (http://localhost:8000 if free)
  python demo/serve.py 9000       # choose a port
"""
import http.server
import socketserver
import sys
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


def open_server(first_port: int) -> socketserver.TCPServer:
    # if the port is taken (another app, or reserved by Windows), try the next ones
    for port in range(first_port, first_port + 20):
        try:
            return socketserver.TCPServer(("127.0.0.1", port), Handler)
        except OSError:
            print(f"port {port} is busy, trying {port + 1}")
    sys.exit(f"no free port between {first_port} and {first_port + 19}; pass one: python demo/serve.py 9000")


if __name__ == "__main__":
    with open_server(int(sys.argv[1]) if len(sys.argv) > 1 else PORT) as httpd:
        print(f"Live demo: http://localhost:{httpd.server_address[1]}  (Ctrl+C to stop)")
        httpd.serve_forever()
