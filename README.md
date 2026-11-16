# Pump Sound Anomaly Detection with Edge Impulse (Edge AI)

Predictive maintenance by ear: a tiny neural network listens to an industrial pump and flags
**abnormal** operation (leakage, contamination, clogging, damage) and tells both apart from
a quiet room (**background**). It runs entirely on the edge device; here the laptop stands in for a microcontroller.

The project follows the workflow of Edge Impulse's *Industrial Productivity Solutions Guide*
(data → DSP → model → EON Tuner → testing → C++ deployment → automation → collaboration). It
uses the free Developer plan, so the guide's Enterprise-only parts are rebuilt with scripts.

| | |
|---|---|
| Classes | `normal`, `abnormal`, `background` (a GMM anomaly block was tried in v1; see Results) |
| Data | DCASE 2020 Task 2 dev set, pump (MIMII), 16 kHz mono, 10 s clips + laptop-mic recordings |
| Impulse | 1000 ms window / 500 ms stride → MFE (61 × 40) → 1D CNN, int8 |
| Deployment | C++ library (EON, int8) built and run in WSL Ubuntu; WebAssembly monitoring dashboard |
| Target estimate | Arduino Nano 33 BLE Sense (Cortex-M4F 64 MHz, onboard microphone) |

Results: see [Results](#results) (filled in from `results/*.json`).

## Repository layout

```
scripts/
  download_dataset.py   DCASE 2020 Task 2 machine zip from Zenodo (resume + size check), extract
  prepare_dataset.py    balanced subset, stratified split, format checks, manifest, demo/playback sets
  upload_to_ei.py       ingestion API upload (explicit x-label), count verification
  record_mic_path.py    replay clips through the laptop speaker into its mic + background recording
  augment_noise.py      mix room noise into training clips at random SNR
  build_impulse.py      create/configure/train/test the impulse through the Studio API
  ei_pipeline.py        upload new data -> retrain -> build C++ + WASM -> download (automation)
  wav_to_features.py    raw-features file for the C++ app
  run_all.py            rebuild the whole Edge Impulse project (target -> upload -> train -> build)
  eval_cpp_results.py   summarise the C++ test-set evaluation (accuracy, recall, confusion)
  make_model_card.py    results/*.json -> demo/model_card.json for the dashboard
  demo_check.py         headless browser evaluation + dashboard screenshots
deployment/cpp-linux/   build.sh / run.sh for example-standalone-inferencing-linux (WSL);
                        eval.sh + batch_eval.cpp score every test clip with the C++ library
demo/                   four-tab WebAssembly dashboard, served by demo/serve.py
  app.js                monitor, alarm logic, analysis, stress test, model card
  dsp.js                FFT, MFE features, band removal, noise mixing, WAV decoding
  charts.js             canvas charts with hover tooltips (no libraries, works offline)
.github/workflows/      scheduled retrain + rebuild every 2 days
docs/                   report builder, demo script, viva sheet, screenshot checklist
results/                metrics from Edge Impulse and from the C++ evaluation
```

## Reproduce

Requirements: Python 3.10+, an Edge Impulse account (free), and WSL/Linux for the C++ build.

```bash
pip install -r requirements.txt edgeimpulse-api
cp .env.example .env                     # fill in EI_API_KEY and EI_PROJECT_ID

python scripts/download_dataset.py --machine pump
python scripts/prepare_dataset.py --machine pump --max-per-class 300
python scripts/upload_to_ei.py

# laptop-mic data (speaker unmuted, ~80% volume, quiet room)
python scripts/record_mic_path.py check
python scripts/record_mic_path.py replay
python scripts/record_mic_path.py background --minutes 8
python scripts/augment_noise.py
python scripts/upload_to_ei.py --dir data/mic/training --category training
python scripts/upload_to_ei.py --dir data/mic/testing --category testing
python scripts/upload_to_ei.py --dir data/augmented/training --category training

python scripts/run_all.py                                       # target, upload, impulse, features, train, build

python scripts/ei_pipeline.py --skip-upload --skip-retrain      # C++ + WASM builds
bash deployment/cpp-linux/build.sh                              # in WSL
python scripts/wav_to_features.py demo/clips/abnormal_1_id00.wav --out deployment/cpp-linux/features-abnormal.txt
bash deployment/cpp-linux/run.sh deployment/cpp-linux/features-abnormal.txt
bash deployment/cpp-linux/eval.sh                               # in WSL: score the whole test set
python scripts/eval_cpp_results.py results/cpp_eval.csv

python scripts/make_model_card.py
python demo/serve.py                                            # prints the address (8000, or 8001 if busy)
```

## Live dashboard

`python demo/serve.py`, then open the printed address in Chrome or Edge (`python demo/serve.py 9000`
picks a port). The model runs as WebAssembly in the page; nothing is sent anywhere.

| Tab | What it does |
|---|---|
| 01 Monitor | Replay a held-out clip, the live microphone, or your own audio file; optional noise mixed in at a chosen SNR. A verdict every 0.25 s, a machine state with hysteresis (ALARM when ≥ 6 of the last 8 windows are abnormal, cleared at ≤ 2), an event log with CSV export, the MFE features the network sees, and an uplink counter (16 B per state change vs 32 KB/s of audio). |
| 02 Analyze a clip | Every 0.5 s window, majority vote, MFE of the whole clip, and an occlusion test: each frequency band is removed (FFT → zero → inverse FFT) and the model re-run to show which bands drive the decision. |
| 03 Noise stress test | All held-out clips with white noise or a room recording at 7 SNR levels, classified in the browser. |
| 04 Model card | Signal path with the real tensor shapes, test results, confusion matrix, device estimates, and a benchmark of 300 inferences on the viewing machine. |

Keys: `1`–`0` play clips, `M` microphone, `Esc` stop.

## Results

Final model (v2, public Edge Impulse project [1165553](https://studio.edgeimpulse.com/public/1165553/live)), int8. The test set (141 held-out clips, 2,679
one-second windows) is scored with the exported C++ library itself (`deployment/cpp-linux/eval.sh`
+ `scripts/eval_cpp_results.py`), because the free plan's compute was used up by training.

| | Final (v2) | v1 (same network, 100 epochs) |
|---|---|---|
| Validation accuracy, int8 | 81.9 % | 91.4 % |
| Test windows correct | 78.7 % | 83.7 % |
| Test clips correct (majority of 19 windows) | 78.7 % | 83.7 % |
| Recall abnormal / normal / background | 81.4 % / 73.1 % / 100.0 % | 78.9 % / 86.4 % / 100.0 % |
| Classifier on Arduino Nano 33 BLE Sense (est.) | 7 ms int8 vs 150 ms float32 | 10 ms int8 |
| C++ on the laptop, per window | 1.1 ms DSP + 1.0 ms inference | |

Why v2 is a little weaker than v1: the second project had 25 compute minutes, so v2 was trained
with batch 256 for a 240 s budget (58 epochs) instead of 100 epochs. Why v2 has no anomaly
block: v1's second processing block (Spectral Analysis → GMM) overflowed the SDK's feature buffer
at run time ("Would write outside feature buffer"), and a GMM on the MFE features crashed
Edge Impulse's profiler. The full story, with the debugging lessons, is in the report.

Measured in the dashboard (10 held-out clips, all in the browser):

- **Which frequencies matter (occlusion test):** for abnormal clip 1 of pump 00, removing
  300–600 Hz lowers p(abnormal) by 0.33 and 1–1.5 kHz by 0.09; bands above 4 kHz have no effect.
  The model listens to the low-frequency fault signature.
- **Noise stress test:** window accuracy stays within 5 points of clean (77.4 %) down to 0 dB
  white noise (72.6 %), and light noise (20 dB) even raises it to 82.6 %; with a room recording it
  falls slowly to 68.9 % at −5 dB. Noise only ever turns *normal* into *abnormal*: abnormal recall
  is 100 % at every level, so the monitor fails safe (false alarms, never missed faults).

## Data and attribution

Audio comes from the **DCASE 2020 Challenge Task 2 development dataset** (Koizumi et al.,
Zenodo record 3678171), built from **MIMII** (Purohit et al., Hitachi) and **ToyADMOS** (NTT). It is
licensed **CC BY-NC-SA 4.0** and is not redistributed here; the scripts download it. The
Edge Impulse project that contains the uploaded clips is shared for non-commercial academic use
under the same license.

Related work used for comparison: Manivannan, *Industry 4.0 Predictive Maintenance* (public
Edge Impulse projects 88093 / 88799 / 88839); the DCASE 2020 Task 2 baseline
(`y-kawagu/dcase2020_task2_baseline`).
