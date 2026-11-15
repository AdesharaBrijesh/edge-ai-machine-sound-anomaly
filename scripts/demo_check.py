#!/usr/bin/env python3
"""Verify the WebAssembly deployment in a real browser and capture dashboard screenshots.

Serves demo/ on a temporary local port, then drives headless Chrome:
  * demo/?evaluate=1  -> every held-out clip is classified by the WASM model in the browser;
                         the results table is parsed into results/demo_eval.json
  * demo/?offline=... -> dashboard filled from one clip, saved as docs/report/figures/dashboard.png
  * demo/?evaluate=1  -> evaluation table screenshot, docs/report/figures/dashboard_eval.png

  python scripts/demo_check.py [--clip abnormal_1_id00.wav]
"""
from __future__ import annotations

import argparse
import functools
import html
import http.server
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
FIGS = ROOT / "docs" / "report" / "figures"
CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm",
                      ".js": "text/javascript", ".json": "application/json"}

    def log_message(self, *args):  # keep the console quiet
        pass


def chrome(args: list[str], timeout: int = 180) -> str:
    exe = next((c for c in CHROME_CANDIDATES if c.exists()), None)
    if exe is None:
        sys.exit("Chrome/Edge not found")
    base = [str(exe), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
            "--autoplay-policy=no-user-gesture-required", "--virtual-time-budget=90000"]
    out = subprocess.run(base + args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    return out.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="abnormal_1_id00.wav")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not (DEMO / "model" / "edge-impulse-standalone.js").exists():
        sys.exit("demo/model is empty; run scripts/ei_pipeline.py --skip-upload --skip-retrain --targets wasm")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port),
                                             functools.partial(Handler, directory=str(DEMO)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{args.port}/"
    try:
        dom = chrome(["--dump-dom", base + "?evaluate=1"])
        match = re.search(r'<pre id="eval-json"[^>]*>(.*?)</pre>', dom, re.S)
        if not match or not match.group(1).strip():
            title = re.search(r"<title>(.*?)</title>", dom)
            sys.exit(f"evaluation did not finish (page title: {title.group(1) if title else '?'})")
        result = json.loads(html.unescape(match.group(1)))
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "demo_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"browser (WASM) evaluation: window accuracy {result['windowAccuracy']:.3f}, clip accuracy "
              f"{result['clipAccuracy']:.2f}, {result['inferences']} inferences, {result['meanLatencyMs']:.2f} ms each")
        for row in result["clips"]:
            print(f"  {row['file']:24s} true {row['label']:9s} -> {row['majority']:10s} "
                  f"{row['windowAccuracy']:.0%} of windows, anomaly {row['meanAnomaly']:.2f}")

        FIGS.mkdir(parents=True, exist_ok=True)
        chrome(["--window-size=1280,1000", f"--screenshot={FIGS / 'dashboard.png'}", base + f"?offline={args.clip}"])
        chrome(["--window-size=1280,1500", f"--screenshot={FIGS / 'dashboard_eval.png'}", base + "?evaluate=1"])
        for name in ("dashboard.png", "dashboard_eval.png"):
            print(f"  screenshot {'ok' if (FIGS / name).exists() else 'MISSING'}: docs/report/figures/{name}")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
