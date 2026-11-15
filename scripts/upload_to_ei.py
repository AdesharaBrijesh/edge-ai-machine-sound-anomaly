#!/usr/bin/env python3
"""Upload labelled WAV files to the Edge Impulse project through the ingestion API.

Free-plan stand-in for the guide's organization-dataset import: files go straight into the
project's training or testing set.

  python scripts/upload_to_ei.py                                   # data/prepared/{training,testing}
  python scripts/upload_to_ei.py --dir data/mic/training --category training
  python scripts/upload_to_ei.py --verify-only                     # just compare counts

Labels come from the text before the first dot of each filename and are always sent with
x-label; without it Edge Impulse would derive a separate label from every filename.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, INGESTION_API, EIClient, get_config  # noqa: E402

BATCH = 50
CATEGORIES = ("training", "testing")


def label_of(path: Path) -> str:
    return path.name.split(".", 1)[0]


def post_batch(session: requests.Session, key: str, category: str, label: str, batch: list[Path]) -> dict:
    last_error = ""
    for attempt in range(1, 5):
        handles = [p.open("rb") for p in batch]
        try:
            resp = session.post(
                f"{INGESTION_API}/{category}/files",
                headers={"x-api-key": key, "x-label": label, "x-disallow-duplicates": "1"},
                files=[("data", (p.name, h, "audio/wav")) for p, h in zip(batch, handles)],
                timeout=600,
            )
            if resp.status_code < 500:
                try:
                    return {"status": resp.status_code, **resp.json()}
                except ValueError:
                    return {"status": resp.status_code, "success": False, "error": resp.text[:300]}
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        finally:
            for h in handles:
                h.close()
        wait = 5 * attempt
        print(f"    retry {attempt} in {wait}s ({last_error})")
        time.sleep(wait)
    return {"status": 0, "success": False, "error": last_error}


def upload(groups: dict[tuple[str, str], list[Path]], key: str) -> Counter:
    totals: Counter = Counter()
    session = requests.Session()
    for (category, label), paths in sorted(groups.items()):
        print(f"{category}/{label}: {len(paths)} files")
        for start in range(0, len(paths), BATCH):
            batch = paths[start:start + BATCH]
            result = post_batch(session, key, category, label, batch)
            files = result.get("files")
            if not files:
                totals["failed"] += len(batch)
                print(f"  ! batch failed: {result.get('error', result)}")
                continue
            for item in files:
                err = str(item.get("error", ""))
                if item.get("success"):
                    totals["uploaded"] += 1
                elif "exist" in err.lower() or "duplicate" in err.lower():
                    totals["duplicate"] += 1
                else:
                    totals["failed"] += 1
                    print(f"  ! {item.get('fileName')}: {err}")
            print(f"  {min(start + BATCH, len(paths))}/{len(paths)}")
    return totals


def remote_counts(client: EIClient) -> dict[str, Counter]:
    counts: dict[str, Counter] = {}
    for category in CATEGORIES:
        counts[category] = Counter()
        offset = 0
        while True:
            res = client.get("raw-data", category=category, limit=1000, offset=offset)
            samples = res.get("samples", [])
            counts[category].update(s.get("label", "?") for s in samples)
            if len(samples) < 1000:
                break
            offset += 1000
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DATA / "prepared", help="folder containing training/ and testing/")
    ap.add_argument("--dir", type=Path, help="upload every .wav in this folder instead of --source")
    ap.add_argument("--category", choices=[*CATEGORIES, "split"], help="category for --dir")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    key, project_id = get_config()
    client = EIClient(key, project_id)

    if not args.verify_only:
        groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
        if args.dir:
            if not args.category:
                sys.exit("--dir needs --category")
            for path in sorted(args.dir.glob("*.wav")):
                groups[(args.category, label_of(path))].append(path)
        else:
            for category in CATEGORIES:
                for path in sorted((args.source / category).glob("*.wav")):
                    groups[(category, label_of(path))].append(path)
        if not groups:
            sys.exit("Nothing to upload.")
        totals = upload(groups, key)
        print(f"\nUploaded {totals['uploaded']}, already present {totals['duplicate']}, failed {totals['failed']}")
        if totals["failed"]:
            sys.exit(1)

    print("\nSamples now in the project:")
    for category, counter in remote_counts(client).items():
        print(f"  {category:9s} total={sum(counter.values()):4d}  " + "  ".join(f"{k}={v}" for k, v in sorted(counter.items())))


if __name__ == "__main__":
    main()
