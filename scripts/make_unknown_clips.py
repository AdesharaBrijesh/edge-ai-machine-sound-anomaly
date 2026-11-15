#!/usr/bin/env python3
"""Create synthetic "unknown" sounds for testing the anomaly score (never used for training).

Writes demo/clips/unknown_<name>.wav (10 s, 16 kHz, int16) and adds them to demo/clips/clips.json
with label "unknown", so the dashboard lists them and `?evaluate=1` reports their anomaly scores.

  python scripts/make_unknown_clips.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "demo" / "clips"
SR, SECONDS, RMS = 16000, 10.0, 0.05     # similar loudness to the dataset clips


def scaled(x: np.ndarray) -> np.ndarray:
    return x * RMS / (np.sqrt(np.mean(x ** 2)) + 1e-12)


def main() -> None:
    t = np.arange(int(SR * SECONDS)) / SR
    rng = np.random.default_rng(3)
    sounds = {
        "white_noise": rng.standard_normal(len(t)),
        "two_tone_hum": np.sin(2 * np.pi * 1000 * t) + 0.6 * np.sin(2 * np.pi * 3000 * t),
        "frequency_sweep": chirp(t, f0=200, t1=SECONDS, f1=6000, method="logarithmic"),
    }
    index_path = CLIPS / "clips.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    index = [c for c in index if c.get("label") != "unknown"]
    for k, (name, x) in enumerate(sounds.items(), start=1):
        file = f"unknown_{k}_{name}.wav"
        wavfile.write(CLIPS / file, SR, np.clip(np.round(scaled(x) * 32767), -32768, 32767).astype(np.int16))
        index.append({"file": file, "label": "unknown", "machine_id": name.replace("_", " "),
                      "source": "synthetic", "duration_s": SECONDS})
        print(f"wrote demo/clips/{file}")
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
