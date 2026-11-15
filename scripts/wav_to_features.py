#!/usr/bin/env python3
"""Write one window of a 16 kHz WAV file as comma-separated raw features.

This is the input format of example-standalone-inferencing-linux's `custom` app (the same
numbers Studio's "Raw features" copy button gives): int16-range sample values, exactly
EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE of them (16000 for a 1000 ms window).

  python scripts/wav_to_features.py demo/clips/abnormal_1_id00.wav --start 2.0 --out features.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="window start in seconds")
    ap.add_argument("--window-ms", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("features.txt"))
    args = ap.parse_args()

    rate, data = wavfile.read(args.wav)
    if rate != 16000 or data.dtype != np.int16 or data.ndim != 1:
        sys.exit("expected a mono 16 kHz int16 WAV")
    n = rate * args.window_ms // 1000
    start = int(args.start * rate)
    window = data[start:start + n]
    if len(window) < n:
        sys.exit(f"window runs past the end of the file ({len(data) / rate:.2f} s)")
    args.out.write_text(",".join(str(int(v)) for v in window) + "\n", encoding="ascii")
    print(f"wrote {n} features from {args.wav.name} @ {args.start:.2f}s to {args.out}")


if __name__ == "__main__":
    main()
