#!/usr/bin/env python3
"""Summarise deployment/cpp-linux/eval.sh output (per-window CSV from the C++ impulse).

  python scripts/eval_cpp_results.py results/cpp_eval_v2.csv [--json results/cpp_eval_v2.json]

The true label is the file-name prefix (abnormal.*, normal.*, background.*), as uploaded.
Reports window accuracy (argmax), accuracy of confident windows (top score >= threshold),
clip accuracy by majority vote, a confusion matrix, the DCASE vs laptop-mic split and timing.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarise(path: Path, threshold: float = 0.6) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = [k for k in rows[0] if k not in ("file", "window", "anomaly", "dsp_ms", "classification_ms")]
    clips: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        clips[r["file"]].append(r)

    def truth(file: str) -> str:
        return Path(file).name.split(".")[0]

    def source(file: str) -> str:
        return "mic" if "/mic/" in file.replace("\\", "/") else "dcase"

    confusion = {t: Counter() for t in labels}
    groups: dict[str, Counter] = defaultdict(Counter)
    clip_rows = []
    anomaly_by_label: dict[str, list[float]] = defaultdict(list)
    for file, windows in clips.items():
        t = truth(file)
        votes = Counter()
        for w in windows:
            scores = {lab: float(w[lab]) for lab in labels}
            pred = max(scores, key=scores.get)
            votes[pred] += 1
            confusion[t][pred] += 1
            for g in ("all", source(file)):
                groups[g]["windows"] += 1
                groups[g]["correct"] += pred == t
                if scores[pred] >= threshold:
                    groups[g]["confident"] += 1
                    groups[g]["confident_correct"] += pred == t
            if w.get("anomaly"):
                anomaly_by_label[t].append(float(w["anomaly"]))
        majority = votes.most_common(1)[0][0]
        for g in ("all", source(file)):
            groups[g]["clips"] += 1
            groups[g]["clips_correct"] += majority == t
        clip_rows.append({"file": Path(file).name, "label": t, "majority": majority,
                          "window_accuracy": votes[t] / len(windows)})

    def rates(c: Counter) -> dict:
        return {"windows": c["windows"], "window_accuracy": c["correct"] / c["windows"],
                "confident_share": c["confident"] / c["windows"],
                "confident_accuracy": c["confident_correct"] / max(1, c["confident"]),
                "clips": c["clips"], "clip_accuracy": c["clips_correct"] / c["clips"]}

    per_class = {t: {"windows": sum(confusion[t].values()),
                     "recall": confusion[t][t] / max(1, sum(confusion[t].values()))} for t in labels}
    dsp = [float(r["dsp_ms"]) for r in rows]
    nn = [float(r["classification_ms"]) for r in rows]
    return {
        "source_csv": str(path).replace("\\", "/"),
        "labels": labels,
        "threshold": threshold,
        "overall": rates(groups["all"]),
        "dcase": rates(groups["dcase"]) if groups["dcase"]["windows"] else None,
        "mic": rates(groups["mic"]) if groups["mic"]["windows"] else None,
        "per_class": per_class,
        "confusion": {t: {p: confusion[t][p] for p in labels} for t in labels},
        "anomaly_mean_by_label": {t: sum(v) / len(v) for t, v in anomaly_by_label.items()} or None,
        "mean_dsp_ms": sum(dsp) / len(dsp),
        "mean_classification_ms": sum(nn) / len(nn),
        "clips": clip_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--json", type=Path, help="write the summary here (default: next to the CSV)")
    ap.add_argument("--threshold", type=float, default=0.6)
    args = ap.parse_args()
    s = summarise(args.csv, args.threshold)
    out = args.json or args.csv.with_suffix(".json")
    out.write_text(json.dumps(s, indent=2), encoding="utf-8")

    o = s["overall"]
    print(f"{args.csv.name}: {o['windows']} windows, {o['clips']} clips")
    print(f"  window accuracy {o['window_accuracy']:.1%} | confident (>= {s['threshold']}) "
          f"{o['confident_share']:.0%} of windows, {o['confident_accuracy']:.1%} correct | "
          f"clip majority {o['clip_accuracy']:.1%}")
    for g in ("dcase", "mic"):
        if s[g]:
            print(f"  {g:5s}: windows {s[g]['window_accuracy']:.1%}, clips {s[g]['clip_accuracy']:.1%} ({s[g]['clips']})")
    print("  recall: " + ", ".join(f"{t} {v['recall']:.1%}" for t, v in s["per_class"].items()))
    print("  confusion (rows = true): " + "; ".join(
        f"{t}: " + " ".join(f"{p}={n}" for p, n in row.items()) for t, row in s["confusion"].items()))
    if s["anomaly_mean_by_label"]:
        print("  mean anomaly: " + ", ".join(f"{t} {v:.2f}" for t, v in s["anomaly_mean_by_label"].items()))
    print(f"  timing on this PC: DSP {s['mean_dsp_ms']:.2f} ms, classifier {s['mean_classification_ms']:.2f} ms")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
