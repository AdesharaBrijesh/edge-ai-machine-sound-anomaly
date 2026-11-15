#!/usr/bin/env python3
"""Record the "mic path": play clips through the laptop speaker while capturing them with the
laptop's own microphone, and record plain background audio.

Why: in the live demo the model hears machine sounds through a speaker and the laptop mic,
which colours the sound (weak bass, room echo, mic processing). If only the background class
were recorded with the laptop mic, the model could learn the shortcut "laptop-mic sound means
background". Re-recording labelled machine clips through the same path removes that shortcut.

  python scripts/record_mic_path.py check                    # does the mic hear the speaker? (noise burst)
  python scripts/record_mic_path.py replay                   # data/playback -> data/mic/{training,testing}
  python scripts/record_mic_path.py background --minutes 8   # 10 s chunks, every 5th one to testing
  python scripts/record_mic_path.py playlist                 # one WAV per label to play from a phone
  python scripts/record_mic_path.py phone --label normal --minutes 4   # record while the phone plays it

Windows note: laptop microphones usually run "audio enhancements" (echo cancellation, noise
suppression) that erase sound coming from the laptop's own speaker and steady machine noise.
Turn them off for the microphone (Settings > System > Sound > Microphone > Audio enhancements: Off).

Output files are named <label>.mic-<name>.wav so upload_to_ei.py can read the label.
"""
from __future__ import annotations

import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import correlate, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

TARGET_SR = 16000          # stored at the impulse's rate; devices run at their native rate
OUT = DATA / "mic"
PLAYBACK = DATA / "playback"


def rms_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def resample(x: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return np.asarray(x, dtype=np.float64)
    ratio = Fraction(to_sr, from_sr).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator)


def device_rate(device, kind: str) -> int:
    return int(sd.query_devices(device, kind)["default_samplerate"])


def input_channels(device) -> int:
    return max(1, min(2, int(sd.query_devices(device, "input")["max_input_channels"])))


def record(seconds: float, device=None) -> np.ndarray:
    """Record `seconds` from the input device; returns mono audio at TARGET_SR."""
    sr = device_rate(device, "input")
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=input_channels(device), dtype="float32", device=device)
    sd.wait()
    return resample(rec.mean(axis=1), sr, TARGET_SR)


def play_and_record(x16: np.ndarray, tail_s: float, devices) -> np.ndarray:
    """Play 16 kHz audio while recording; separate streams so any device pair works.
    Returns the mono recording at TARGET_SR (starts slightly before the playback)."""
    in_dev, out_dev = devices
    in_sr, out_sr = device_rate(in_dev, "input"), device_rate(out_dev, "output")
    chunks: list[np.ndarray] = []

    def callback(indata, frames, t, status):
        chunks.append(indata.mean(axis=1).copy())

    with sd.InputStream(device=in_dev, samplerate=in_sr, channels=input_channels(in_dev), dtype="float32",
                        callback=callback):
        time.sleep(0.15)
        sd.play(np.clip(resample(x16, TARGET_SR, out_sr), -1, 1).astype(np.float32), out_sr,
                device=out_dev, blocking=True)
        time.sleep(tail_s)
    rec = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    return resample(rec, in_sr, TARGET_SR)


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


def devices(args) -> tuple[int, int]:
    di, do = sd.default.device
    return (args.input if args.input is not None else di, args.output if args.output is not None else do)


def pink_noise(seconds: float, rms: float = 0.1, seed: int = 0) -> np.ndarray:
    """Broadband 1/f noise at TARGET_SR: a stand-in for machine sound when testing pickup."""
    n = int(seconds * TARGET_SR)
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1 / TARGET_SR)
    freqs[0] = 1.0
    x = np.fft.irfft(spec / np.sqrt(freqs), n)
    return rms * x / np.sqrt(np.mean(x ** 2))


def cmd_check(args) -> None:
    in_dev, out_dev = devices(args)
    print(f"input : [{in_dev}] {sd.query_devices(in_dev)['name']}")
    print(f"output: [{out_dev}] {sd.query_devices(out_dev)['name']}")
    quiet = record(2.0, in_dev)
    rec = play_and_record(pink_noise(2.0), 0.3, (in_dev, out_dev))
    during = rec[int(0.5 * TARGET_SR): int(2.0 * TARGET_SR)]
    rise = rms_dbfs(during) - rms_dbfs(quiet)
    print(f"room {rms_dbfs(quiet):.1f} dBFS -> {rms_dbfs(during):.1f} dBFS while the speaker plays noise "
          f"(+{rise:.1f} dB)")
    if rise < 12:
        print("WEAK: the mic barely hears the speaker. Turn Windows 'Audio enhancements' off for the microphone, "
              "raise the speaker volume, or use `phone` mode.")
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
    in_dev, out_dev = devices(args)
    print(f"{len(jobs)} clips, global playback gain x{gain:.1f}; keep the room quiet (~{len(jobs) * 11 / 60:.0f} min)")

    weak = 0
    for i, ((path, category, label), clip) in enumerate(zip(jobs, clips), start=1):
        dest = OUT / category / f"{label}.mic-{path.name.split('.', 1)[1]}"
        if dest.exists() and not args.overwrite:
            continue
        rec16 = play_and_record(clip * gain, args.tail, (in_dev, out_dev))
        aligned, lag, corr = align(clip, rec16)
        save16(dest, aligned)
        flag = "" if corr >= 0.15 else "  <- weak match"
        weak += corr < 0.15
        print(f"[{i:2d}/{len(jobs)}] {category}/{dest.name}  lag {lag / TARGET_SR * 1000:.0f} ms  "
              f"corr {corr:.2f}  level {rms_dbfs(aligned):.1f} dBFS{flag}")
    if weak:
        print(f"WARNING: {weak} recordings barely match their source; check volume / room noise.")


def record_chunks(args, label: str, category_of, minutes: float, tag: str) -> None:
    in_dev, _ = devices(args)
    chunks = int(round(minutes * 60 / 10))
    stamp = time.strftime("%Y%m%d-%H%M")
    for n in range(1, chunks + 1):
        rec16 = record(10.0, in_dev)
        category = category_of(n)
        dest = OUT / category / f"{label}.mic-{tag}-{stamp}-{n:03d}.wav"
        save16(dest, rec16)
        print(f"[{n:3d}/{chunks}] {category}/{dest.name}  level {rms_dbfs(rec16):.1f} dBFS")


def cmd_background(args) -> None:
    print("Recording background. Mix in normal room sounds: silence, talking, typing, footsteps, fan/AC.")
    record_chunks(args, "background", lambda n: "testing" if n % 5 == 0 else "training", args.minutes, "bg")


def cmd_phone(args) -> None:
    if not args.label:
        sys.exit("phone mode needs --label normal|abnormal (and --category training|testing)")
    print(f"Play the {args.label} playlist from your phone ~20 cm from the laptop now; recording {args.minutes} min.")
    record_chunks(args, args.label, lambda n: args.category, args.minutes, "phone")


def cmd_playlist(args) -> None:
    """Join playback clips into one WAV per label, to play from a phone in `phone` mode."""
    for kind in ("train", "test"):
        for label_dir in sorted((PLAYBACK / kind).glob("*")):
            clips = [load16(p) for p in sorted(label_dir.glob("*.wav"))]
            if clips:
                dest = PLAYBACK / f"playlist_{kind}_{label_dir.name}.wav"
                save16(dest, np.concatenate(clips) * 0.97)
                print(f"{dest.relative_to(DATA.parent)}  {sum(len(c) for c in clips) / TARGET_SR / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "replay", "background", "phone", "playlist"])
    ap.add_argument("--input", type=int, help="input device index (default: system default)")
    ap.add_argument("--output", type=int, help="output device index (default: system default)")
    ap.add_argument("--minutes", type=float, default=8.0, help="minutes to record (background / phone)")
    ap.add_argument("--label", choices=["normal", "abnormal"], help="phone mode label")
    ap.add_argument("--category", choices=["training", "testing"], default="training", help="phone mode category")
    ap.add_argument("--peak", type=float, default=0.8, help="playback peak level (0-1) after global gain")
    ap.add_argument("--tail", type=float, default=1.5,
                    help="extra seconds recorded after each clip (covers ~1 s of speaker latency)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    {"check": cmd_check, "replay": cmd_replay, "background": cmd_background,
     "phone": cmd_phone, "playlist": cmd_playlist}[args.mode](args)


if __name__ == "__main__":
    main()
