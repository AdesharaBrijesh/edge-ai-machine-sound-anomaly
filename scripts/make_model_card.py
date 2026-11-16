#!/usr/bin/env python3
"""Write demo/model_card.json (shown on the dashboard's Model card tab) from results/*.json.

  python scripts/make_model_card.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(name: str) -> dict:
    path = RES / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_block(ev: dict) -> dict:
    if not ev:
        return {}
    return {
        "window_accuracy": ev["overall"]["window_accuracy"],
        "clip_accuracy": ev["overall"]["clip_accuracy"],
        "mic_clip_accuracy": (ev.get("mic") or {}).get("clip_accuracy"),
        "recall": {k: v["recall"] for k, v in ev["per_class"].items()},
        "confusion": ev["confusion"],
    }


def main() -> None:
    v2, v1 = load("cpp_eval_v2.json"), load("cpp_eval_v1.json")
    train, perf, perf1 = load("train_main.json"), load("performance.json"), load("performance_v1.json")
    rows = list(csv.DictReader((ROOT / "data" / "manifest.csv").open(encoding="utf-8"))) \
        if (ROOT / "data" / "manifest.csv").exists() else []
    splits = Counter(r["split"] for r in rows)
    mic = len(list((ROOT / "data" / "mic").glob("*/*.wav")))
    aug = len(list((ROOT / "data" / "augmented" / "training").glob("*.wav")))
    uploaded = splits.get("training", 0) + splits.get("testing", 0) + mic + aug
    conf = (train.get("float32") or {}).get("confusion_matrix") or []
    val_windows = sum(sum(r) for r in conf)

    def mem(key: str):
        value = (perf.get("classifier_int8") or {}).get(key)
        return value if value else (perf1.get("classifier_int8") or {}).get(key)

    card = {
        "test": {"v2": test_block(v2), "v1": test_block(v1)},
        "train": {"val_int8": (train.get("int8") or {}).get("accuracy"),
                  "val_float32": (train.get("float32") or {}).get("accuracy")},
        "device": {"int8_ms": perf.get("nn_ms"),
                   "float32_ms": (perf.get("classifier_float32") or {}).get("latency_ms"),
                   "ram_kb": mem("ram_kb"), "flash_kb": mem("flash_kb")},
        "data": {"clips": uploaded or None, "pump_units": 4, "mic_clips": mic or None, "augmented": aug or None,
                 "test_clips": (v2.get("overall") or {}).get("clips"),
                 "train_windows": round(val_windows / 0.2 * 0.8) if val_windows else None},
    }
    out = ROOT / "demo" / "model_card.json"
    out.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
