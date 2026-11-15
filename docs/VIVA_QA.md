# Viva preparation: likely questions and short answers

## Concepts

**What is an "impulse" in Edge Impulse?**
The whole processing pipeline that runs on the device: an input block (cuts the signal into
windows), processing/DSP blocks (turn each window into features), and learning blocks
(classify the features). Ours: 1 s audio windows at 16 kHz → MFE → 1D CNN, plus
Spectral Analysis → GMM anomaly detector.

**Why Edge AI instead of the cloud?**
Latency (alarm in under a second), no bandwidth for continuous audio, privacy of factory
audio, keeps working offline, and low power: only the result is transmitted.

**Why sound for predictive maintenance?**
Faults (leaks, clogging, bearing wear, impeller damage) change the acoustic signature before
failure. A microphone is cheap, non-invasive, and does not touch the machine.

## Data

**Where does the data come from?**
DCASE 2020 Challenge Task 2 development set (built on Hitachi's MIMII), pump subset:
real factory pumps, 16 kHz, 10 s clips, four pump units, normal and anomalous (leakage,
contamination, clogging). We also recorded our own data with the laptop microphone.

**How did you avoid data leakage?**
The split is by clip, not by window, so all windows of a clip stay in one set. It is stratified
by class and pump unit. Ten clips are held out completely for the live demo. Limitation: the
same four pump units appear in both training and testing; `--holdout-id 06` tests an unseen unit.

**Why a background class, and why re-record clips through the laptop?**
Without a "no machine" class, the model must call every sound normal or abnormal. If only the
background were recorded with the laptop mic, the model could learn "laptop-mic sound means
background" (shortcut learning). Re-recording pump clips through the speaker → mic path removes
that shortcut and teaches the colouring of the real deployment microphone (bass loss below
about 300 Hz).

**What did you learn about the laptop microphone?**
Windows "audio enhancements" (echo cancellation and noise suppression) erased the speaker sound
completely, and suppress steady machine noise in general. A monitoring device needs a raw
microphone path; we used the raw driver stream for recording and turned the enhancements off
for the demo.

## DSP and model

**Why MFE and not MFCC?**
MFCCs decorrelate the spectrum for speech (vocal-tract envelope). Machine sounds are
non-voice: MFE keeps the mel-band energies directly, preserving harmonic and broadband
structure. Edge Impulse recommends MFE for non-voice audio.

**Why 1 s windows with a 500 ms stride?**
1 s covers many pump rotations and gives 61 × 40 = 2,440 features (16,000 samples = 32 KB
raw, which fits a microcontroller). The 500 ms stride gives 19 training windows per 10 s clip,
and a decision twice per second.

**Why a 1D CNN?**
Convolution along time over 40 mel channels finds local spectro-temporal patterns cheaply; it
is small (a few thousand parameters) and fast on Cortex-M with CMSIS-NN kernels.

**What is the GMM anomaly score?**
A Gaussian mixture model (5 components) is fitted to the spectral features of all training
windows. A new window's anomaly score is how unlikely it is under that model. Unknown sounds
(hair dryer, drill) score high even though the classifier must still choose a known class.

**Why did training first fail, and how did you fix it?**
Loss stuck at 0.693 (ln 2), and every window was predicted as the same class. Edge Impulse
feeds windows grouped by class, and the default script shuffles only 4 × batch = 128 windows,
so batches were single-class. Switching to expert mode and fully shuffling the training set
each epoch fixed it. We proved the features were fine first: a logistic regression on
downloaded MFE statistics already reached 78 % on held-out clips. A second issue followed:
Studio's default learning rate (0.005) made validation accuracy swing between about 57 % and
80 %, then collapse after 75 epochs. 0.001 trained stably. Edge Impulse keeps the best epoch.

**Why does the classifier receive the spectral features if it does not use them?**
The Edge Impulse SDK runs every processing block and writes their outputs one after another
into one buffer sized by the classifier's input. With MFE only (2,440 values), the anomaly
block's 69 spectral values overflowed it at run time ("Would write outside feature buffer").
Feeding both blocks to the classifier sizes the buffer to 2,509. A Cropping1D layer inside
the network passes only the MFE part on, so the model itself is unchanged.

## Results and deployment

**What accuracy did you get?**
Validation {val acc} (int8); test {test acc} on held-out clips. For comparison: the public
Edge Impulse pump project reaches 93.5 % test accuracy, and the DCASE autoencoder baseline has
72.9 % AUC (an unsupervised task, so harder).

**What does int8 quantisation do?**
Weights and activations become 8-bit integers instead of 32-bit floats: about 4× smaller and
faster on MCUs, usually with under 1 % accuracy loss. Edge Impulse reports both variants.

**What is the EON compiler?**
Edge Impulse's compiler that turns the network into plain C++ instead of interpreting it with
TensorFlow Lite Micro at run time. It uses 25–55 % less RAM and about 35 % less flash, with
the same accuracy. Trade-off: the model is compiled into the firmware and cannot be swapped
without reflashing.

**Can it run in real time on the target?**
For continuous monitoring, DSP + inference must take less than the 500 ms stride. Estimated
on the Arduino Nano 33 BLE Sense: {total ms} (DSP {dsp ms} + NN {nn ms}), {ram} KB RAM,
{flash} KB flash.

**How did you deploy without hardware?**
Two ways. (1) The C++ library, built with example-standalone-inferencing-linux in WSL Ubuntu,
exactly as in the guide (`APP_CUSTOM=1 make`, then `./build/custom features.txt`). (2) A
WebAssembly build running in the browser with the live microphone: the same generated code,
compiled for the browser.

**How is retraining automated?**
`ei_pipeline.py` uploads new labelled data, starts a retrain (features + all learning blocks),
builds the C++ and WASM packages, and downloads them. GitHub Actions runs it every 2 days.
Edge Impulse's native project pipeline (retrain → new version → C++ deployment) mirrors the
guide's figures 32–33.

**What would you do next?**
Flash an Arduino Nano 33 BLE Sense and use its PDM microphone. Test on an unseen pump unit.
Add an unsupervised normal-only model for faults never recorded. Add more machine types.
Use performance calibration to reduce false alarms in continuous mode.
