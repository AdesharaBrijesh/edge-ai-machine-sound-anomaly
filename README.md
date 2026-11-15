# Pump Sound Anomaly Detection with Edge Impulse (Edge AI)

Predictive maintenance by ear: a tiny neural network listens to an industrial pump and flags
**abnormal** operation (leakage, contamination, clogging, damage) and **unknown sounds**. It
runs entirely on the edge device; here the laptop stands in for a microcontroller.

The project follows the workflow of Edge Impulse's *Industrial Productivity Solutions Guide*
(data → DSP → model → EON Tuner → testing → C++ deployment → automation → collaboration). It
uses the free Developer plan, so the guide's Enterprise-only parts are rebuilt with scripts.

| | |
|---|---|
| Classes | `normal`, `abnormal`, `background` + GMM anomaly score for never-heard sounds |
| Data | DCASE 2020 Task 2 dev set, pump (MIMII), 16 kHz mono, 10 s clips + laptop-mic recordings |
| Impulse | 1000 ms window / 500 ms stride → MFE → 1D CNN; Spectral Analysis → GMM |
| Deployment | C++ library (EON, int8) built and run in WSL Ubuntu; WebAssembly live dashboard |
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
deployment/cpp-linux/   build.sh / run.sh for example-standalone-inferencing-linux (WSL)
demo/                   live dashboard (replay + microphone), served by demo/serve.py
.github/workflows/      scheduled retrain + rebuild every 2 days
docs/STUDIO_STEPS.md    screenshot checklist and manual Studio fallback
results/                metrics exported from Edge Impulse
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

python scripts/build_impulse.py target --device "nano 33"
python scripts/build_impulse.py all

python scripts/ei_pipeline.py --skip-upload --skip-retrain      # C++ + WASM builds
bash deployment/cpp-linux/build.sh                              # in WSL
python scripts/wav_to_features.py demo/clips/abnormal_1_id00.wav --out deployment/cpp-linux/features-abnormal.txt
bash deployment/cpp-linux/run.sh deployment/cpp-linux/features-abnormal.txt

python demo/serve.py                                            # http://localhost:8000
```

## Results

_To be filled in after training._

## Data and attribution

Audio comes from the **DCASE 2020 Challenge Task 2 development dataset** (Koizumi et al.,
Zenodo record 3678171), built from **MIMII** (Purohit et al., Hitachi) and **ToyADMOS** (NTT). It is
licensed **CC BY-NC-SA 4.0** and is not redistributed here; the scripts download it. The
Edge Impulse project that contains the uploaded clips is shared for non-commercial academic use
under the same license.

Related work used for comparison: Manivannan, *Industry 4.0 Predictive Maintenance* (public
Edge Impulse projects 88093 / 88799 / 88839); the DCASE 2020 Task 2 baseline
(`y-kawagu/dcase2020_task2_baseline`).
