#!/usr/bin/env python3
"""Prepare DCASE 2020 Task 2 machine-sound clips for Edge Impulse.

Free-plan stand-in for the reference guide's transformation block
(edgeimpulse/industrial-solutions-guide): it turns the raw dataset into validated,
labelled, split files that the ingestion API accepts as-is.

Steps
  1. Find every clip named {normal|anomaly}_id_XX_NNNNNNNN.wav under --src.
  2. Pick a balanced subset per class, spread evenly across machine IDs, and hold a few
     clips back for the live demo (they are never uploaded).
  3. Split training/testing 80/20, stratified by class and machine ID, or hold out a whole
     machine ID with --holdout-id to test on an unseen pump.
  4. Validate the format (mono, 16 kHz, 16-bit PCM) and convert anything that differs.
  5. Write <label>.<machine>-id<ID>-<origin>-<index>.wav (Edge Impulse reads the label from
     the text before the first dot), choose clips to replay through the laptop speaker for
     the mic-path set, and record everything in data/manifest.csv.

Usage:  python scripts/prepare_dataset.py --machine pump --max-per-class 300
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT  # noqa: E402

PATTERN = re.compile(r"^(normal|anomaly)_id_(\d{2})_(\d+)\.wav$", re.IGNORECASE)
LABEL_OF = {"normal": "normal", "anomaly": "abnormal"}
TARGET_SR = 16000


def collect(src: Path) -> list[dict]:
    clips = []
    for path in sorted(src.rglob("*.wav")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        kind, mid, idx = match.groups()
        clips.append({"src": path, "label": LABEL_OF[kind.lower()], "mid": mid,
                      "origin": path.parent.name, "idx": int(idx)})
    return clips


def by_id(clips: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        groups[clip["mid"]].append(clip)
    return [groups[mid] for mid in sorted(groups)]


def interleave(groups: list[list[dict]], rng: random.Random) -> list[dict]:
    """Shuffle each group, then take one clip from each in turn so machine IDs stay balanced."""
    pools = [rng.sample(group, len(group)) for group in groups]
    ordered = []
    while any(pools):
        for pool in pools:
            if pool:
                ordered.append(pool.pop())
    return ordered


def order_for(label: str, clips: list[dict], rng: random.Random, normal_source: str) -> list[dict]:
    mine = [c for c in clips if c["label"] == label]
    if label == "normal" and normal_source == "test-first":
        # Prefer the normal clips DCASE ships next to the anomalies (test/), then train/.
        first = [c for c in mine if c["origin"] == "test"]
        rest = [c for c in mine if c["origin"] != "test"]
        return interleave(by_id(first), rng) + interleave(by_id(rest), rng)
    return interleave(by_id(mine), rng)


def assign_split(upload: list[dict], test_frac: float, holdout_id: str | None, rng: random.Random) -> None:
    for group in by_id(upload):
        if holdout_id is not None:
            for clip in group:
                clip["split"] = "testing" if clip["mid"] == holdout_id else "training"
            continue
        shuffled = rng.sample(group, len(group))
        n_test = round(len(shuffled) * test_frac)
        for i, clip in enumerate(shuffled):
            clip["split"] = "testing" if i < n_test else "training"


def write_wav(src: Path, dest: Path) -> tuple[float, str]:
    """Copy src to dest as mono 16 kHz int16. Returns (duration_s, note)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(src), "rb") as wf:
            channels, rate, width, frames = wf.getnchannels(), wf.getframerate(), wf.getsampwidth(), wf.getnframes()
        if (channels, rate, width) == (1, TARGET_SR, 2):
            shutil.copyfile(src, dest)
            return frames / rate, ""
    except (wave.Error, EOFError):
        pass  # unusual header -> fall through to a full decode

    rate, data = wavfile.read(src)
    notes = []
    if data.ndim > 1:
        data = data.mean(axis=1)
        notes.append("mono")
    if np.issubdtype(data.dtype, np.floating):
        data = np.clip(data, -1.0, 1.0) * 32767.0
        notes.append("float->int16")
    elif data.dtype == np.int32:
        data = data / 65536.0
        notes.append("int32->int16")
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) * 256.0
        notes.append("uint8->int16")
    if rate != TARGET_SR:
        data = resample_poly(np.asarray(data, dtype=np.float64), TARGET_SR, rate)
        notes.append(f"{rate}->{TARGET_SR}Hz")
        rate = TARGET_SR
    out = np.clip(np.round(data), -32768, 32767).astype(np.int16)
    wavfile.write(dest, rate, out)
    return len(out) / rate, ",".join(notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default="pump")
    ap.add_argument("--src", type=Path, help="extracted dataset folder (default data/raw/dev_data_<machine>)")
    ap.add_argument("--max-per-class", type=int, default=300, help="clips uploaded per class (balanced)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--holdout-id", help="put every clip of this machine ID (e.g. 06) in testing")
    ap.add_argument("--demo-per-class", type=int, default=5, help="held-out clips for the live demo")
    ap.add_argument("--playback-train", type=int, default=24, help="training clips per class to replay (mic path)")
    ap.add_argument("--playback-test", type=int, default=6, help="testing clips per class to replay (mic path)")
    ap.add_argument("--normal-source", choices=["test-first", "any"], default="test-first")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = args.src or DATA / "raw" / f"dev_data_{args.machine}"
    clips = collect(src)
    if not clips:
        sys.exit(f"No DCASE clips found under {src}. Run scripts/download_dataset.py first.")
    print(f"Found {len(clips)} clips under {src}")
    for (label, origin), n in sorted(Counter((c["label"], c["origin"]) for c in clips).items()):
        print(f"  {label:9s} {origin:6s} {n}")

    rng = random.Random(args.seed)
    ordered = {label: order_for(label, clips, rng, args.normal_source) for label in ("normal", "abnormal")}
    available = min(len(ordered[label]) - args.demo_per_class for label in ordered)
    n_upload = min(args.max_per_class, available)
    if n_upload < args.max_per_class:
        print(f"Only {available} clips available per class after demo hold-out; using {n_upload}.")

    out_prepared, out_playback, out_demo = DATA / "prepared", DATA / "playback", ROOT / "demo" / "clips"
    for folder in (out_prepared, out_playback, out_demo):
        if folder.exists():
            shutil.rmtree(folder)

    rows, demo_index = [], []
    for label, seq in ordered.items():
        demo = seq[:args.demo_per_class]
        upload = seq[args.demo_per_class:args.demo_per_class + n_upload]
        assign_split(upload, args.test_frac, args.holdout_id, rng)

        for k, clip in enumerate(demo, start=1):
            name = f"{label}_{k}_id{clip['mid']}.wav"
            duration, note = write_wav(clip["src"], out_demo / name)
            demo_index.append({"file": name, "label": label, "machine_id": clip["mid"],
                               "source": clip["src"].name, "duration_s": round(duration, 2)})
            rows.append(dict(file=f"demo/clips/{name}", label=label, machine_id=clip["mid"], origin=clip["origin"],
                             src_index=clip["idx"], split="demo", playback="", duration_s=round(duration, 2), note=note))

        playback_pick = {
            "train": interleave(by_id([c for c in upload if c["split"] == "training"]), rng)[:args.playback_train],
            "test": interleave(by_id([c for c in upload if c["split"] == "testing"]), rng)[:args.playback_test],
        }
        playback_of = {id(c): kind for kind, picked in playback_pick.items() for c in picked}

        for clip in upload:
            name = f"{label}.{args.machine}-id{clip['mid']}-{clip['origin']}-{clip['idx']:08d}.wav"
            dest = out_prepared / clip["split"] / name
            duration, note = write_wav(clip["src"], dest)
            kind = playback_of.get(id(clip), "")
            if kind:
                pb = out_playback / kind / label / name
                pb.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dest, pb)
            rows.append(dict(file=dest.relative_to(ROOT).as_posix(), label=label, machine_id=clip["mid"],
                             origin=clip["origin"], src_index=clip["idx"], split=clip["split"], playback=kind,
                             duration_s=round(duration, 2), note=note))

    (out_demo / "clips.json").write_text(json.dumps(demo_index, indent=2), encoding="utf-8")
    manifest = DATA / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Summary + sanity checks
    print(f"\nWrote {manifest.relative_to(ROOT)} ({len(rows)} rows)")
    table = Counter((r["split"], r["label"]) for r in rows)
    for split in ("training", "testing", "demo"):
        line = "  ".join(f"{label}={table[(split, label)]}" for label in ("normal", "abnormal"))
        print(f"  {split:9s} {line}")
    per_id = Counter((r["split"], r["machine_id"]) for r in rows if r["split"] in ("training", "testing"))
    print("  per machine ID:", dict(sorted(per_id.items())))
    print("  playback:", dict(Counter((r["playback"], r["label"]) for r in rows if r["playback"])))
    converted = [r for r in rows if r["note"]]
    print(f"  converted files: {len(converted)}" + (f" (e.g. {converted[0]['note']})" if converted else ""))
    durations = sorted({r["duration_s"] for r in rows})
    print(f"  durations: {durations[0]}s .. {durations[-1]}s")
    total_h = sum(r["duration_s"] for r in rows if r["split"] in ("training", "testing")) / 3600
    print(f"  audio to upload: {total_h:.2f} h")

    names = [Path(r["file"]).name for r in rows if r["split"] in ("training", "testing")]
    sources = [(r["machine_id"], r["origin"], r["src_index"], r["label"]) for r in rows]
    assert len(names) == len(set(names)), "duplicate output names"
    assert len(sources) == len(set(sources)), "a source clip was used twice"
    print("  checks: no duplicate names, no source clip used twice")


if __name__ == "__main__":
    main()
