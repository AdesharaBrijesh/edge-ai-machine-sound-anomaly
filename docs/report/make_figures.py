#!/usr/bin/env python3
"""Figures and numbers for the report (docs/report/figures/*.png + report_data.json).

Every figure is optional: it is produced only when its inputs exist, so the report can be
rebuilt at any stage. Colours follow the dataviz reference palette (categorical slots
blue/orange/aqua for class identity, a single-hue blue ramp for magnitude).

  python docs/report/make_figures.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from scipy.signal import stft, welch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
RESULTS = ROOT / "results"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"normal": "#2a78d6", "abnormal": "#eb6834", "background": "#1baf7a"}
FINAL = "#8f5bd6"   # fourth categorical colour: the final (time-budgeted) training run
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + BLUE_RAMP)


def style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": SURFACE, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def load(path: Path) -> np.ndarray:
    rate, data = wavfile.read(path)
    assert rate == 16000
    return data.astype(np.float64) / 32768.0


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float = 0.0,
                   fmax: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    fmax = fmax or sr / 2
    hz_to_mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    mel_to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, centre):
            fb[m - 1, k] = (k - left) / max(centre - left, 1)
        for k in range(centre, right):
            fb[m - 1, k] = (right - k) / max(right - centre, 1)
    return fb, mel_to_hz(mels[1:-1])


def log_mel(x: np.ndarray, sr: int = 16000, n_mels: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, t, z = stft(x, fs=sr, nperseg=512, noverlap=256)
    fb, centres = mel_filterbank(n_mels, 512, sr)
    power = fb @ (np.abs(z) ** 2)
    return 10 * np.log10(power + 1e-10), t, centres


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}")


def fig_spectrograms(rows: list[dict]) -> None:
    pick = {}
    for label in ("normal", "abnormal"):
        cand = [r for r in rows if r["label"] == label and r["split"] == "training" and r["machine_id"] == "00"]
        if cand:
            pick[label] = ROOT / cand[0]["file"]
    if len(pick) < 2:
        return
    specs = {label: log_mel(load(path)) for label, path in pick.items()}
    vmax = max(s[0].max() for s in specs.values())
    vmin = vmax - 50
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True, constrained_layout=True)
    for ax, (label, (spec, t, centres)) in zip(axes, specs.items()):
        im = ax.imshow(spec, origin="lower", aspect="auto", cmap=SEQ, vmin=vmin, vmax=vmax,
                       extent=[t[0], t[-1], 0, len(centres)])
        ticks = [0, 16, 32, 48, 63]
        ax.set_yticks(ticks, [f"{centres[i] / 1000:.1f}" for i in ticks])
        ax.set_title(f"{label.capitalize()} pump (ID 00)", loc="left")
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (kHz, mel scale)")
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("Energy (dB)", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "spectrograms.png")


def fig_mic_path() -> None:
    mic = sorted((ROOT / "data" / "mic" / "training").glob("normal.mic-*.wav"))
    if not mic:
        return
    rec = mic[0]
    original = next((ROOT / "data" / "prepared" / "training").glob(f"normal.{rec.name.split('mic-', 1)[1]}"), None)
    if original is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 2.6), constrained_layout=True)
    curves = {}
    for label, path, colour in (("Original dataset clip", original, SERIES["normal"]),
                                ("Same clip: laptop speaker → laptop mic", rec, SERIES["abnormal"])):
        f, p = welch(load(path), fs=16000, nperseg=2048)
        db = 10 * np.log10(p + 1e-14)
        db -= db[(f > 100)].max()
        curves[label] = (f, db, colour)
        ax.plot(f[1:], db[1:], color=colour, linewidth=2, label=label)
    ax.set_xscale("log")
    ax.set_xlim(50, 8000)
    ticks = [50, 100, 200, 500, 1000, 2000, 5000, 8000]
    ax.set_xticks(ticks, ["50", "100", "200", "500", "1k", "2k", "5k", "8k"])
    ax.minorticks_off()
    ax.set_ylim(-45, 5)
    ax.grid(True, axis="y")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Level (dB, peak = 0)")
    ax.set_title("What the laptop microphone hears: the speaker path removes most of the bass", loc="left")
    ax.legend(loc="lower left", fontsize=8, labelcolor=INK2)
    save(fig, "mic_path.png")


EPOCH_RE = re.compile(r"loss: ([0-9.]+) - accuracy: ([0-9.]+) - val_loss: ([0-9.]+) - val_accuracy: ([0-9.]+)")


def fig_training_curves(log: Path, name: str = "training_curves.png") -> None:
    if not log.exists():
        return
    rows = [tuple(map(float, m.groups())) for m in EPOCH_RE.finditer(log.read_text(encoding="utf-8"))]
    if len(rows) < 3:
        return
    arr = np.array(rows)
    epochs = np.arange(1, len(arr) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(6.5, 2.4), constrained_layout=True)
    for ax, (tr, va), title in ((ax_acc, (1, 3), "Accuracy"), (ax_loss, (0, 2), "Loss")):
        for col, label, colour in ((tr, "training", SERIES["normal"]), (va, "validation", SERIES["abnormal"])):
            ax.plot(epochs, arr[:, col], color=colour, linewidth=2, label=label)
            ax.annotate(label, (epochs[-1], arr[-1, col]), xytext=(4, 0), textcoords="offset points",
                        color=INK2, fontsize=8, va="center")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.grid(True, axis="y")
        ax.set_xlim(1, epochs[-1] * 1.18)
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(loc="lower right", fontsize=8, labelcolor=INK2)
    save(fig, name)


def fig_training_story(runs_spec: list[tuple[str, Path, str]]) -> None:
    """Validation accuracy per epoch for the runs that led to the final model."""
    runs = []
    for label, path, colour in runs_spec:
        if path.exists():
            vals = [float(m.group(4)) for m in EPOCH_RE.finditer(path.read_text(encoding="utf-8"))]
            if len(vals) >= 3:
                runs.append((label, vals, colour))
    if len(runs) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.1), constrained_layout=True)
    longest = max(len(v) for _, v, _ in runs)
    ax.plot([1, longest], [0.5, 0.5], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(2, 0.47, "chance", color=MUTED, fontsize=8, va="top")
    ends = []
    for label, vals, colour in runs:
        epochs = np.arange(1, len(vals) + 1)
        ax.plot(epochs, vals, color=colour, linewidth=1.6, label=label)
        ends.append([vals[-1], f"{label}: best {max(vals):.0%}", colour])
    # direct labels at the right edge, nudged apart so they never overlap
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06)
    for y, text, colour in ends:
        ax.annotate(text, (longest, y), xytext=(6, 0), textcoords="offset points", color=colour, fontsize=8, va="center")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(1, longest * 1.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(True, axis="y")
    ax.set_title("Same data and network: the shuffle, then the learning rate, made it train", loc="left")
    ax.set_ylim(0.3, 1.0)
    save(fig, "shuffle_fix.png")


def fig_anomaly_scores(path: Path) -> None:
    """Mean GMM anomaly score per demo clip (computed by the WebAssembly model in the browser)."""
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = data.get("clips") or []
    if not clips:
        return
    colours = {"normal": SERIES["normal"], "abnormal": SERIES["abnormal"], "unknown": SERIES["background"]}
    order = {"normal": 0, "abnormal": 1, "unknown": 2}
    clips = sorted(clips, key=lambda c: (order.get(c["label"], 3), c["file"]))
    names = [c["file"].replace(".wav", "").replace("_", " ") for c in clips]
    values = [c["meanAnomaly"] for c in clips]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(clips) + 0.9), constrained_layout=True)
    y = np.arange(len(clips))[::-1]
    ax.barh(y, values, height=0.7, color=[colours.get(c["label"], MUTED) for c in clips],
            edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, names, fontsize=8)
    thr = data.get("threshold")
    if isinstance(thr, (int, float)):
        ax.axvline(thr, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"threshold {thr:g}", (thr, y[0] + 0.6), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in ("normal", "abnormal", "unknown")]
    ax.legend(handles, ["normal pump", "abnormal pump", "synthetic unknown sound"], loc="lower right",
              fontsize=8, labelcolor=INK2)
    ax.set_xlabel("Mean anomaly score (GMM)")
    ax.grid(True, axis="x")
    ax.set_title("Unknown sounds score far above every pump clip", loc="left")
    save(fig, "anomaly_scores.png")


def cpp_run() -> dict:
    """Terminal output of ./build/custom on the three sample windows (deployment/cpp-linux/sample-output.txt)."""
    path = ROOT / "deployment" / "cpp-linux" / "sample-output.txt"
    if not path.exists():
        return {}
    lines = [l.rstrip() for l in path.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith(("#", "Built:"))]
    return {"output_lines": lines}


def dataset_stats(rows: list[dict]) -> dict:
    counts = Counter((r["split"], r["label"]) for r in rows)
    mic = Counter()
    for category in ("training", "testing"):
        for path in (ROOT / "data" / "mic" / category).glob("*.wav"):
            mic[(category, path.name.split(".", 1)[0])] += 1
    aug = len(list((ROOT / "data" / "augmented" / "training").glob("*.wav")))
    return {"dcase": {f"{s}/{l}": n for (s, l), n in counts.items()},
            "mic": {f"{s}/{l}": n for (s, l), n in mic.items()}, "augmented": aug}


def main() -> None:
    style()
    rows = list(csv.DictReader((ROOT / "data" / "manifest.csv").open(encoding="utf-8")))
    print("Figures:")
    fig_spectrograms(rows)
    fig_mic_path()
    fig_training_curves(RESULTS / "train_main.log")
    fig_training_story([
        ("default shuffle, lr 0.005", RESULTS / "train_stuck.log", SERIES["abnormal"]),
        ("full shuffle, lr 0.005", RESULTS / "train_lr005.log", SERIES["background"]),
        ("full shuffle, lr 0.001, batch 32 (v1)", RESULTS / "train_lr001_mfeonly.log", SERIES["normal"]),
        ("batch 256, lr 0.003, 240 s budget (final)", RESULTS / "train_main.log", FINAL),
    ])
    data = {"dataset": dataset_stats(rows), "cpp_run": cpp_run()}
    for path in sorted(RESULTS.glob("*.json")):
        if not path.name.endswith("_full.json"):
            data[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    (Path(__file__).resolve().parent / "report_data.json").write_text(json.dumps(data, indent=2, default=str),
                                                                      encoding="utf-8")
    print("  report_data.json")


if __name__ == "__main__":
    main()
