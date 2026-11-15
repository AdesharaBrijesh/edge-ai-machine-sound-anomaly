#!/usr/bin/env python3
"""Mix recorded room noise into training clips at a random SNR.

A numpy version of edgeimpulse/example-transform-mix-audio: for each selected clean clip a
random background recording is scaled so that signal-to-noise = U(min, max) dB and added.
The classifier then learns that a pump with a noisy room behind it is still that pump.

  python scripts/augment_noise.py --per-class 100 --min-snr 5 --max-snr 20

Clean clips:  data/prepared/training/{normal,abnormal}.*.wav
Noise:        data/mic/training/background.*.wav   (record_mic_path.py background)
Output:       data/augmented/training/<label>.mix-<name>-snr<NN>.wav   (training only)
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402


def load(path: Path) -> np.ndarray:
    rate, data = wavfile.read(path)
    assert rate == 16000 and data.dtype == np.int16, f"{path}: expected 16 kHz int16"
    return data.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--min-snr", type=float, default=5.0)
    ap.add_argument("--max-snr", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    noises = [load(p) for p in sorted((DATA / "mic" / "training").glob("background.*.wav"))]
    if not noises:
        sys.exit("No background recordings in data/mic/training (run record_mic_path.py background).")
    out = DATA / "augmented" / "training"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rng = random.Random(args.seed)
    made = 0
    for label in ("normal", "abnormal"):
        clean_paths = sorted((DATA / "prepared" / "training").glob(f"{label}.*.wav"))
        for path in rng.sample(clean_paths, min(args.per_class, len(clean_paths))):
            clean = load(path)
            noise = rng.choice(noises)
            if len(noise) < len(clean):
                noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
            start = rng.randrange(0, len(noise) - len(clean) + 1)
            noise = noise[start:start + len(clean)]
            snr = rng.uniform(args.min_snr, args.max_snr)
            p_clean = np.mean(clean ** 2)
            p_noise = np.mean(noise ** 2) + 1e-9
            mixed = clean + noise * np.sqrt(p_clean / (p_noise * 10 ** (snr / 10)))
            peak = np.max(np.abs(mixed))
            if peak > 32767:                     # avoid clipping by scaling the whole mix
                mixed *= 32767 / peak
            name = f"{label}.mix-{path.stem.split('.', 1)[1]}-snr{snr:02.0f}.wav"
            wavfile.write(out / name, 16000, np.round(mixed).astype(np.int16))
            made += 1
    print(f"Wrote {made} noise-mixed training clips to {out} (SNR {args.min_snr}-{args.max_snr} dB, "
          f"{len(noises)} noise recordings)")


if __name__ == "__main__":
    main()
