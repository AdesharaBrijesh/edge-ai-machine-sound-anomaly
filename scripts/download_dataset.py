#!/usr/bin/env python3
"""Download and extract one machine type of the DCASE 2020 Task 2 development dataset.

Source: Koizumi et al., DCASE 2020 Challenge Task 2 development dataset, Zenodo record 3678171
(https://zenodo.org/records/3678171), licensed CC BY-NC-SA 4.0. Clips are single-channel,
16 kHz, ~10 s WAV files built from MIMII (pump, fan, slider, valve) and ToyADMOS.

Usage:  python scripts/download_dataset.py --machine pump
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

MACHINES = ["pump", "fan", "slider", "valve", "ToyCar", "ToyConveyor"]
URL = "https://zenodo.org/records/3678171/files/dev_data_{machine}.zip?download=1"
CHUNK = 1 << 20


def download(url: str, dest: Path) -> None:
    head = requests.head(url, allow_redirects=True, timeout=60)
    head.raise_for_status()
    total = int(head.headers.get("content-length", 0))
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        print(f"{dest.name}: already complete ({total / 1e6:.0f} MB)")
        return

    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        resume = have and resp.status_code == 206
        mode = "ab" if resume else "wb"
        done = have if resume else 0
        next_report = done + 50 * CHUNK
        with dest.open(mode) as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    print(f"  {done / 1e6:.0f} MB{pct}")
                    next_report += 50 * CHUNK

    size = dest.stat().st_size
    if total and size != total:
        sys.exit(f"Size mismatch for {dest.name}: got {size}, expected {total}. Re-run to resume.")
    print(f"{dest.name}: downloaded {size / 1e6:.0f} MB")


def extract(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"{out_dir.name}: already extracted")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad:
            sys.exit(f"Corrupt member in {zip_path.name}: {bad}")
        zf.extractall(out_dir)  # zipfile strips absolute paths and '..' components
    marker.write_text("ok\n", encoding="utf-8")
    wavs = sum(1 for _ in out_dir.rglob("*.wav"))
    print(f"{out_dir.name}: extracted {wavs} wav files")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default="pump", choices=MACHINES)
    ap.add_argument("--out", type=Path, default=DATA / "raw")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.out / f"dev_data_{args.machine}.zip"
    download(URL.format(machine=args.machine), zip_path)
    extract(zip_path, args.out / f"dev_data_{args.machine}")


if __name__ == "__main__":
    main()
