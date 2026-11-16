# Edge Impulse Studio: screenshot checklist and manual fallback

The scripts build everything through the Edge Impulse API, so Studio already shows the finished
project. This page lists the screenshots the report and slides need, and how to do each step by
hand if an API step ever fails.

Open the project at `https://studio.edgeimpulse.com/studio/<EI_PROJECT_ID>`.

## Screenshots

Save each one as PNG in `docs/screenshots/` with exactly this name. The right column shows
which figure of the reference guide it mirrors.

| File name | Where in Studio | What must be visible | Guide figure |
|---|---|---|---|
| `01-dashboard.png` | Dashboard | Project info panel with **Target device** (Arduino Nano 33 BLE Sense) | Fig. 2 |
| `02-data-acquisition.png` | Data acquisition | Sample list, labels, the training/testing split ratio | Fig. 9 |
| `03-data-explorer.png` | Data acquisition → Data explorer tab | The clustered dataset view (optional) | – |
| `04-impulse-design.png` | Impulse design → Create impulse | Input block, MFE, Classifier | Fig. 15 |
| `05-mfe-parameters.png` | Impulse design → MFE | Parameters panel and the spectrogram (DSP result) | Fig. 16 |
| `06-mfe-features.png` | MFE → Generate features tab | Feature explorer and on-device performance | Figs. 3a, 17 |
| `07-anomaly-explorer.png` | Anomaly detection (**first project, 1164931**) | v1 GMM on the Spectral features | – |
| `08-classifier.png` | Classifier | Accuracy, loss, confusion matrix, on-device performance (Quantized int8) | Figs. 3b, 20 |
| `10-model-testing.png` | Model testing (**first project, 1164931**) | v1 accuracy and confusion matrix | Fig. 21 |
| `11-eon-tuner.png` | EON Tuner | The configuration page (no run: no compute left) | Fig. 13 |
| `12-deployment.png` | Deployment | C++ library selected, EON compiler on, int8 vs float32 table | Fig. 22 |
| `13-versioning.png` | Versioning | The saved version(s) | – |
| `14-pipeline.png` | Data acquisition → Data sources | The pipeline with Retrain + Create version + Create deployment | Figs. 32–33 |
| `15-collaborators.png` | Dashboard | Collaborators panel and the README | Fig. 34 |
| `16-launch-in-browser.png` | Dashboard → Run this model → Launch in browser | Live classification from the laptop mic (optional) | – |

Tip: zoom the browser to 80–90% so each panel fits, and use Win+Shift+S to capture a region.

## Manual fallback (only if a script step fails)

Run `python scripts/build_impulse.py info` first. It shows what already exists.

1. **Target device:** Dashboard → Project info → Target device → *Arduino Nano 33 BLE Sense*.
2. **Data:** Data acquisition → Add data → Upload data. Select every file in `data/prepared/training`,
   choose **Training**, and pick "Infer from filename" for the label. Repeat for
   `data/prepared/testing` with **Testing**, and for `data/mic/...`.
3. **Impulse:** Impulse design → Create impulse.
   - Time series data: window size 1000 ms, window increase 500 ms, frequency 16000 Hz, zero-pad on.
   - Add processing block **Audio (MFE)**, then add processing block **Spectral Analysis**.
   - Add learning block **Classification**, with only MFE ticked as its input.
   - Add learning block **Anomaly Detection (GMM)**, with only Spectral features ticked as its input.
   - Click **Save impulse**.
4. **MFE:**
   - Frame length 0.032, frame stride 0.016, filter number 40, FFT length 512.
   - Low frequency 0, high frequency 8000, noise floor −72 dB.
   - Save parameters → Generate features.
5. **Spectral features:** filter none, analysis FFT, FFT length 128, take log on, overlap on → Save parameters →
   Generate features (tick "Calculate feature importance").
6. **Classifier:**
   - 100 training cycles, learning rate 0.005, batch size 32, auto-weight classes on.
   - Data augmentation on: noise, time masking and frequency masking all set to *low*.
   - Architecture: Reshape (40 columns) → 1D conv 16 (kernel 3) → Dropout 0.25 → 1D conv 32 (kernel 3) → Dropout 0.25 → Flatten.
   - Start training.
7. **Anomaly detection:** 5 components, select all axes → Start training.
8. **Model testing:** Classify all.
9. **EON Tuner:** target Nano 33 BLE Sense, latency budget 500 ms → Run EON Tuner.
10. **Native pipeline (the guide's Figs. 32–33):**
    - Data acquisition → Data sources → **+ Add new data source** → *Don't import data*.
    - Tick **Retrain model**, **Create new version** and **Create new deployment** (C++ library).
    - Set it to run daily → **Create pipeline**.
11. **Make the project public:** Dashboard → Project visibility → Public.
