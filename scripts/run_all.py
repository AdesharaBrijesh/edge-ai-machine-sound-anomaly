#!/usr/bin/env python3
"""Rebuild the whole Edge Impulse side of the project in the project named in .env.

  python scripts/run_all.py                 # everything
  python scripts/run_all.py --from train    # resume from a step

Steps: target -> upload -> create -> configure -> features -> train -> test -> perf -> build.
Uploads are resumable (--skip-existing), so re-running after a network error is safe.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = [sys.executable, "-u", "-P"]
BI = str(ROOT / "scripts" / "build_impulse.py")
UP = str(ROOT / "scripts" / "upload_to_ei.py")

STEPS = [
    ("target", [[*PY, BI, "target", "--device", "arduino-nano-33-ble"]]),
    ("upload", [
        [*PY, UP, "--source", str(ROOT / "data" / "prepared"), "--skip-existing"],
        [*PY, UP, "--dir", str(ROOT / "data" / "mic" / "training"), "--category", "training", "--skip-existing"],
        [*PY, UP, "--dir", str(ROOT / "data" / "mic" / "testing"), "--category", "testing", "--skip-existing"],
        [*PY, UP, "--dir", str(ROOT / "data" / "augmented" / "training"), "--category", "training", "--skip-existing"],
    ]),
    ("create", [[*PY, BI, "create", "--anomaly", "none"]]),
    ("configure", [[*PY, BI, "configure"]]),
    ("features", [[*PY, BI, "features"]]),
    # final model: fits the free plan's 25 compute minutes. With more compute, v1's settings
    # (--epochs 100 --lr 0.001 --batch-size 32, no budget) gave ~5 points more test accuracy.
    ("train", [[*PY, BI, "train", "--anomaly", "none", "--epochs", "150", "--lr", "0.003",
                "--batch-size", "256", "--time-budget", "240"]]),
    ("test", [[*PY, BI, "test"]]),
    ("perf", [[*PY, BI, "perf"]]),
    ("build", [[*PY, str(ROOT / "scripts" / "ei_pipeline.py"), "--skip-upload", "--skip-retrain", "--targets", "cpp,wasm"]]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="target", choices=[name for name, _ in STEPS])
    ap.add_argument("--to", dest="stop", default="build", choices=[name for name, _ in STEPS])
    args = ap.parse_args()
    names = [name for name, _ in STEPS]
    for name, commands in STEPS[names.index(args.start): names.index(args.stop) + 1]:
        print(f"\n===== {name} =====", flush=True)
        for cmd in commands:
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                sys.exit(f"step '{name}' failed; fix it and resume with: python scripts/run_all.py --from {name}")
    print("\nAll steps done.")


if __name__ == "__main__":
    main()
