#!/usr/bin/env python3
"""Record the "mic path": play clips through the laptop speaker while capturing them with the
laptop's own microphone, and record plain background audio.

Why: in the live demo the model hears machine sounds through a speaker and the laptop mic,
which colours the sound (weak bass, room echo, mic processing). If only the background class
were recorded with the laptop mic, the model could learn the shortcut "laptop-mic sound means
background". Re-recording labelled machine clips through the same path removes that shortcut.

  python scripts/record_mic_path.py check                    # devices, noise floor, pickup test
  python scripts/record_mic_path.py replay                   # data/playback -> data/mic/{training,testing}
  python scripts/record_mic_path.py background --minutes 8   # 10 s chunks, every 5th one to testing

Output files are named <label>.mic-<name>.wav so upload_to_ei.py can read the label.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

DEVICE_SR = 48000          # play/record at a rate every Windows device accepts...
TARGET_SR = 16000          # ...and store at the impulse's rate
OUT = DATA / "mic"
PLAYBACK = DATA / "playback"


def rms_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def record(seconds: float, device=None) -> np.ndarray:
    rec = sd.rec(int(seconds * DEVICE_SR), samplerate=DEVICE_SR, channels=1, dtype="float32", device=device)
    sd.wait()
    return rec[:, 0]


def play_and_record(x48: np.ndarray, tail_s: float, devices) -> np.ndarray:
    out = np.concatenate([x48, np.zeros(int(tail_s * DEVICE_SR))]).astype(np.float32)
    rec = sd.playrec(out, samplerate=DEVICE_SR, channels=1, dtype="float32", device=devices)
    sd.wait()
    return rec[:, 0]


def align(reference: np.ndarray, recorded: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Crop `recorded` to `reference` length at the latency offset. Returns (clip, lag, corr)."""
    n = len(reference)
    max_lag = len(recorded) - n
    seg = reference[: 3 * TARGET_SR]
    window = recorded[: len(seg) + max(max_lag, 0)]
    c = correlate(window, seg, mode="valid", method="fft")
    lag = int(np.argmax(np.abs(c)))
    match = window[lag: lag + len(seg)]
    corr = float(abs(c[lag]) / (np.linalg.norm(seg) * np.linalg.norm(match) + 1e-12))
    return recorded[lag: lag + n], lag, corr


def save16(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, TARGET_SR, np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16))


def load16(path: Path) -> np.ndarray:
    rate, data = wavfile.read(path)
    if rate != TARGET_SR or data.dtype != np.int16:
        sys.exit(f"{path} must be 16 kHz int16 (run prepare_dataset.py)")
    return data.astype(np.float64) / 32768.0


def cmd_check(args) -> None:
    di, do = sd.default.device
    print(f"input : {sd.query_devices(args.input if args.input is not None else di)['name']}")
    print(f"output: {sd.query_devices(args.output if args.output is not None else do)['name']}")
    quiet = record(3, args.input)
    print(f"noise floor (3 s, stay quiet): {rms_dbfs(quiet):.1f} dBFS")
    t = np.arange(int(2 * DEVICE_SR)) / DEVICE_SR
    tone = 0.1 * np.sin(2 * np.pi * 1000 * t)                     # quiet 1 kHz beep, -20 dBFS
    rec = play_and_record(tone, 0.5, (args.input, args.output))
    spec = np.abs(np.fft.rfft(rec * np.hanning(len(rec))))
    freqs = np.fft.rfftfreq(len(rec), 1 / DEVICE_SR)
    band = (freqs > 950) & (freqs < 1050)
    ref = (freqs > 1200) & (freqs < 2000)
    snr = 20 * np.log10(spec[band].max() / (np.median(spec[ref]) + 1e-12))
    print(f"1 kHz beep picked up {snr:.0f} dB above the room ({rms_dbfs(rec):.1f} dBFS overall)")
    if snr < 15:
        print("WARNING: weak pickup. Raise the speaker volume (~80%), unmute, or disable mic echo "
              "cancellation / 'audio enhancements' in Windows sound settings.")
    else:
        print("OK: the mic hears the speaker; replay recording will work.")


def cmd_replay(args) -> None:
    jobs = []
    for kind, category in (("train", "training"), ("test", "testing")):
        for path in sorted((PLAYBACK / kind).glob("*/*.wav")):
            jobs.append((path, category, path.parent.name))
    if not jobs:
        sys.exit("No playback clips; run prepare_dataset.py first.")
    clips = [load16(p) for p, _, _ in jobs]
    peak = np.percentile(np.concatenate([np.abs(c) for c in clips]), 99.9)
    gain = min(args.peak / max(peak, 1e-6), 30.0)          # one global gain keeps relative loudness
    print(f"{len(jobs)} clips, global playback gain x{gain:.1f}; keep the room quiet (~{len(jobs) * 11 / 60:.0f} min)")

    weak = 0
    for i, ((path, category, label), clip) in enumerate(zip(jobs, clips), start=1):
        dest = OUT / category / f"{label}.mic-{path.name.split('.', 1)[1]}"
        if dest.exists() and not args.overwrite:
            continue
        x48 = resample_poly(clip * gain, DEVICE_SR // TARGET_SR, 1)
        rec48 = play_and_record(np.clip(x48, -1, 1), args.tail, (args.input, args.output))
        rec16 = resample_poly(rec48.astype(np.float64), 1, DEVICE_SR // TARGET_SR)
        aligned, lag, corr = align(clip, rec16)
        save16(dest, aligned)
        flag = "" if corr >= 0.15 else "  <- weak match"
        weak += corr < 0.15
        print(f"[{i:2d}/{len(jobs)}] {category}/{dest.name}  lag {lag / TARGET_SR * 1000:.0f} ms  "
              f"corr {corr:.2f}  level {rms_dbfs(aligned):.1f} dBFS{flag}")
    if weak:
        print(f"WARNING: {weak} recordings barely match their source; check volume / room noise.")


def cmd_background(args) -> None:
    chunks = int(round(args.minutes * 60 / 10))
    stamp = time.strftime("%Y%m%d-%H%M")
    print(f"Recording {chunks} x 10 s of background. Mix in normal room sounds: silence, talking, "
          "typing, footsteps, fan/AC.")
    for n in range(1, chunks + 1):
        rec48 = record(10.0, args.input)
        rec16 = resample_poly(rec48.astype(np.float64), 1, DEVICE_SR // TARGET_SR)
        category = "testing" if n % 5 == 0 else "training"
        dest = OUT / category / f"background.mic-bg-{stamp}-{n:03d}.wav"
        save16(dest, rec16)
        print(f"[{n:3d}/{chunks}] {category}/{dest.name}  level {rms_dbfs(rec16):.1f} dBFS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "replay", "background"])
    ap.add_argument("--input", type=int, help="input device index (default: system default)")
    ap.add_argument("--output", type=int, help="output device index (default: system default)")
    ap.add_argument("--minutes", type=float, default=8.0, help="background minutes")
    ap.add_argument("--peak", type=float, default=0.8, help="playback peak level (0-1) after global gain")
    ap.add_argument("--tail", type=float, default=0.6, help="extra seconds recorded after each clip")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    {"check": cmd_check, "replay": cmd_replay, "background": cmd_background}[args.mode](args)


if __name__ == "__main__":
    main()
