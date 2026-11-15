#!/usr/bin/env python3
"""Automation: new data -> retrain -> build deployments -> download.

Free-plan version of the reference guide's "Automation" section (organization data pipeline +
project train/deploy pipeline):
  1. upload new WAV files from data/incoming/{training,testing}/ (label = text before the first dot)
  2. retrain the impulse with its saved settings (regenerates DSP features, retrains every learn block)
  3. build the C++ library and the WebAssembly package (EON compiler, int8)
  4. download them to deploy/<timestamp>/, unpack the WASM build into demo/model/ and the C++
     library into deployment/cpp-linux/work/model/

  python scripts/ei_pipeline.py                         # every step
  python scripts/ei_pipeline.py --skip-upload --skip-retrain --targets wasm
  python scripts/ei_pipeline.py --list-targets
Runs locally or in GitHub Actions (.github/workflows/retrain.yml) with EI_API_KEY / EI_PROJECT_ID.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import edgeimpulse_api as ei

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_impulse import Studio  # noqa: E402
from common import DATA, ROOT  # noqa: E402

INCOMING = DATA / "incoming"
DEPLOY = ROOT / "deploy"
DEMO_MODEL = ROOT / "demo" / "model"
CPP_MODEL = ROOT / "deployment" / "cpp-linux" / "work" / "model"
TARGETS = {"cpp": "zip", "wasm": None}  # wasm format name is looked up from the project's targets


def find_wasm_format(studio: Studio) -> str:
    targets = studio.deployment.list_deployment_targets_for_project(studio.pid).to_dict().get("targets", [])
    wasm = [t for t in targets if "wasm" in t["format"].lower() or "webassembly" in t["name"].lower()]
    browser = [t for t in wasm if "browser" in (t["format"] + t["name"]).lower()]
    pick = (browser or wasm or [None])[0]
    if not pick:
        sys.exit("No WebAssembly deployment target found; run with --list-targets.")
    return pick["format"]


def build_and_download(studio: Studio, fmt: str, engine: str, out_dir: Path) -> Path:
    req = ei.BuildOnDeviceModelRequest.from_dict({"engine": engine, "modelType": "int8"})
    job = studio.ok(studio.jobs.build_on_device_model_job(studio.pid, type=fmt, build_on_device_model_request=req),
                    f"build {fmt}")
    studio.wait(job.id, f"build {fmt}", timeout=3600)
    resp = studio.deployment.download_build(studio.pid, type=fmt, model_type="int8", engine=engine,
                                            _preload_content=False)
    if resp.status != 200:
        raise RuntimeError(f"download {fmt} failed: HTTP {resp.status}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{fmt}-int8-{engine}.zip"
    dest.write_bytes(resp.data)
    print(f"  saved {dest.relative_to(ROOT)} ({len(resp.data) / 1e6:.1f} MB)")
    return dest


def safe_extract(zip_path: Path, dest: Path, keep=lambda name: True, flatten: bool = False) -> int:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not keep(name) or name.startswith(("/", "\\")) or ".." in Path(name).parts:
                continue
            target = dest / (Path(name).name if flatten else name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--skip-retrain", action="store_true")
    ap.add_argument("--targets", default="cpp,wasm", help="comma list of: cpp, wasm")
    ap.add_argument("--engine", default="tflite-eon")
    ap.add_argument("--list-targets", action="store_true")
    args = ap.parse_args()

    studio = Studio()

    if args.list_targets:
        for t in studio.deployment.list_deployment_targets_for_project(studio.pid).to_dict().get("targets", []):
            print(f"{t['format']:40s} {t['name']}")
        return

    if not args.skip_upload:
        for category in ("training", "testing"):
            folder = INCOMING / category
            if folder.exists() and any(folder.glob("*.wav")):
                print(f"Uploading new {category} data from {folder.relative_to(ROOT)}")
                subprocess.run([sys.executable, "-P", str(ROOT / "scripts" / "upload_to_ei.py"),
                                "--dir", str(folder), "--category", category], check=True)
            else:
                print(f"No new {category} data in {folder.relative_to(ROOT)}")

    if not args.skip_retrain:
        job = studio.ok(studio.jobs.start_retrain_job(studio.pid), "retrain")
        lines = studio.wait(job.id, "retrain", timeout=3600)
        metrics = {v: studio.metrics_from_log(lines, v) for v in ("float32", "int8")}
        studio.save("retrain_latest", metrics)
        for variant, m in metrics.items():
            if m:
                print(f"  retrained {variant}: validation accuracy {m.get('accuracy')}")

    out_dir = DEPLOY / time.strftime("%Y%m%d-%H%M%S")
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        if target == "cpp":
            zip_path = build_and_download(studio, "zip", args.engine, out_dir)
            n = safe_extract(zip_path, CPP_MODEL)
            print(f"  C++ library: {n} files -> {CPP_MODEL.relative_to(ROOT)}")
        elif target == "wasm":
            fmt = find_wasm_format(studio)
            zip_path = build_and_download(studio, fmt, args.engine, out_dir)
            n = safe_extract(zip_path, DEMO_MODEL, keep=lambda name: name.endswith((".js", ".wasm")), flatten=True)
            print(f"  WebAssembly ({fmt}): {n} files -> {DEMO_MODEL.relative_to(ROOT)}")
        else:
            sys.exit(f"unknown target {target}")
    (out_dir / "build-info.json").write_text(json.dumps(
        {"project": studio.pid, "engine": args.engine, "targets": args.targets, "built": time.ctime()}, indent=2),
        encoding="utf-8")
    print(f"Done: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
