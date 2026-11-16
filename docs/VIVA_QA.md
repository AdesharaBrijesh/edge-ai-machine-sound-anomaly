# Viva preparation: likely questions and short answers

## Concepts

**What is an "impulse" in Edge Impulse?**
The whole processing pipeline that runs on the device: an input block (cuts the signal into
windows), processing/DSP blocks (turn each window into features), and learning blocks
(classify the features). Ours (final): 1 s audio windows at 16 kHz → MFE (61 × 40) → 1D CNN,
int8. The first version (v1) also had Spectral Analysis → GMM anomaly detector.

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
A Gaussian mixture model (5 components) is fitted to the features of all training windows; a
new window's anomaly score is how unlikely it is under that model, so unknown sounds score high
even though the classifier must still pick a known class. v1 had one; the final model does not
(see "Why does the final model have no anomaly block?").

**Why did training first fail, and how did you fix it?**
Loss stuck at 0.693 (ln 2), and every window was predicted as the same class. Edge Impulse
feeds windows grouped by class, and the default script shuffles only 4 × batch = 128 windows,
so batches were single-class. Switching to expert mode and fully shuffling the training set
each epoch fixed it. We proved the features were fine first: a logistic regression on
downloaded MFE statistics already reached 78 % on held-out clips. A second issue followed:
Studio's default learning rate (0.005) made validation accuracy swing between about 57 % and
80 %, then collapse after 75 epochs. 0.001 trained stably. Edge Impulse keeps the best epoch.

**Why did the two-block design (MFE + Spectral Analysis) fail?**
We tried two processing blocks first. The deployed SDK writes every block's output into one
buffer sized by the classifier's input, so the anomaly block's 69 spectral values overflowed
it at run time ("Would write outside feature buffer"). Concatenating both blocks into the
classifier fixed the buffer but broke int8: one input scale for MFE values (0–0.8) and
spectral statistics (up to ~400) dropped int8 accuracy from 91 % to 6 %. Lesson: one
processing block per impulse. (Patching nn_input_frame_size to 2,509 in the exported v1 library
made v1's C++ build run, which is how v1 was evaluated.)

**What is per-tensor input quantisation and why did it matter?**
An int8 model stores its input as 8-bit integers with one scale and zero point for the whole
input tensor. If some features are hundreds of times larger than others, the small ones round
to nearly the same integer and their information is lost. Keep features in comparable ranges
(normalise) before feeding one int8 network.

## Results and deployment

**What accuracy did you get?**
Final model (int8): 81.9 % validation; on the 141 held-out test clips (2,679 windows), 78.7 % of
windows and 78.7 % of clips correct; background recall 100 %. v1 (same network, 100 epochs):
83.7 % / 83.7 %. For comparison: the public Edge Impulse pump project reaches 93.5 % (two
classes, whole clips), and the DCASE autoencoder baseline has 72.9 % AUC (unsupervised, harder).

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
on the Arduino Nano 33 BLE Sense: the int8 classifier takes 7 ms (float32: 150 ms), 7.9 KB
RAM, 34.6 KB flash, so almost the whole 500 ms budget is left for the MFE block. On the laptop
the C++ build needs about 1 ms DSP + 1 ms inference per window.

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

## Final model, evaluation and dashboard

**What did Edge Impulse do, and what did you build yourselves?**
Edge Impulse: data storage and labelling (ingestion API), the MFE processing block, cloud GPU
training of the Keras model, int8 quantisation and the EON compiler, on-device latency/memory
estimates for the Nano 33 BLE, and the exported C++ and WebAssembly libraries. Ours: dataset
selection and splitting, laptop-mic recording and noise augmentation, all API automation
(build_impulse.py, run_all.py, ei_pipeline.py, GitHub Actions), the expert-mode training
script (full shuffle, time budget), the C++ build and batch test-set evaluation, and the
four-tab browser dashboard (alarm logic, occlusion test, stress test, benchmark).

**Why does the final model have no anomaly block?**
v1's Spectral Analysis → GMM block broke the deployed library (feature-buffer overflow). In the
final project we fitted the GMM to a slice of the MFE features instead; it trained, but Edge
Impulse's profiling step crashed ("cannot reshape array of size 35,558,120 into shape
(14573, 61)") and used a quarter of the remaining compute. So the final model is classifier-only,
and an unknown sound (e.g. white noise) is forced into a class. Re-adding GMM/K-means is the
first next step.

**Why did you measure test accuracy with the C++ library instead of Edge Impulse's Model testing?**
Model testing is a cloud job and the free compute was used up. Testing the deployed artefact is
arguably stronger anyway: eval.sh builds the standalone Linux app with a batch front end
(batch_eval.cpp) that slides the 1 s window with 0.5 s stride over every test clip. Cross-check:
for v1 the local result (83.7 % of clips) matches Edge Impulse's own Model testing exactly.

**Why is the final model weaker than v1?**
Compute, not method. The second project had 25 minutes in total; v1's schedule (100 epochs,
20 s each) would need ~35. We raised the batch to 256 (per-step time was mostly overhead) and
lr to 0.003, and a callback stopped training after 240 s, keeping the best epoch (58 epochs).
The validation curves of both runs overlap up to that point.

**What does the occlusion test show, and how does it work?**
For each 1 s window, one frequency band is deleted (FFT → zero the band → inverse FFT) and the
model is run again; the drop in the predicted class's probability measures how much the model
relies on that band (152 extra inferences per 10 s clip). For abnormal clip 1 (pump 00),
removing 300–600 Hz lowers p(abnormal) by 0.33; bands above 4 kHz have no effect. The model
uses the low-frequency fault signature, which is also why laptop speakers (−15 to −20 dB below
300 Hz) make the live demo harder.

**What does the noise stress test show?**
The ten held-out clips mixed with white noise or a room recording at 7 levels (clean to −5 dB
SNR), 1,330 inferences per run in the browser. Accuracy stays within 5 points of clean (77 %)
down to 0 dB white noise; light noise (20–30 dB) even raises it to 83 %, consistent with the
noise-augmented training data (5–20 dB). Errors only go one way: noise makes normal pumps sound
abnormal, abnormal recall stays 100 %. For maintenance that is fail-safe: false alarms, never
missed faults.

**How does the alarm logic work?**
Per window the verdict is the average of the last 3 inferences (below 0.6 → uncertain). The
machine state looks at the last 8 decisions (2 s): ALARM when ≥ 6 are abnormal, cleared only when
≤ 2 are (hysteresis), WATCH in between, NO MACHINE when background dominates. One odd window
never raises an alarm. Every state change goes to the event log; an edge device would send only
that 16-byte message instead of 32 KB/s of audio.

**Any debugging story from the dashboard?**
(1) The page hung on "Loading model": in a fast browser the WebAssembly runtime finished before
the wrapper script attached its "ready" callback, so the signal was missed; the wrapper now also
checks Module.calledRun. (2) Charts grew or shrank on every update when browser zoom was not
100 %: setting canvas.height for high-DPI rendering rewrote the height attribute that was also
read as the CSS height, compounding the device-pixel ratio. Storing the intended height once
fixed it.
