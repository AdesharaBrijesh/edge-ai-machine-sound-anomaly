// Content of the project report. Numbers come from report_data.json (see make_figures.py).
'use strict';
const fs = require('fs');
const path = require('path');
const R = require('./build_report.js');
const { D, get, pct, fmt, first, p, h1, h2, h3, bullets, numbered, pageBreak, code, note, table, spacer, fig, shot, TEAM } = R;
const { Document, Packer, Paragraph, TextRun, AlignmentType, Header, Footer, PageNumber, TableOfContents, LevelFormat, HeadingLevel } = R.docx;

// ---------- metric accessors (tolerant to missing data) ----------
const train = (variant, key) => D(`train_main.${variant}.${key}`);
const testAcc = () => first(D('test_main.accuracy.accuracyScore'), D('test_main.log_metrics.accuracy'));
const dcase = (k) => D(`dataset.dcase.${k}`) || 0;
const mic = (k) => D(`dataset.mic.${k}`) || 0;
const perf = (k) => D(`performance.${k}`);
const tuner = () => D('eon_tuner') || {};
const cpp = () => D('cpp_run') || {};
const demo = () => D('demo') || {};

function confusionRows(metrics) {
  if (!metrics) return null;
  const cm = first(metrics.confusion_matrix, metrics.confusionMatrix);
  const labels = first(metrics.class_names, metrics.labels, metrics.classes);
  if (!cm || !labels) return null;
  return cm.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    return [labels[i], ...row.map((v) => `${v} (${Math.round((v / total) * 100)} %)`)];
  });
}
function confusionTable(metrics, title) {
  const rows = confusionRows(metrics);
  const labels = metrics && first(metrics.class_names, metrics.labels, metrics.classes);
  if (!rows) return [p(`${title}: not available yet.`)];
  const w = Math.floor(R.TEXT_W / (labels.length + 1));
  return [p([{ text: title, bold: true }]), table(['actual \\ predicted', ...labels], rows, Array(labels.length + 1).fill(w)), spacer()];
}

function testConfusion() {
  const acc = D('test_main.accuracy');
  const cm = acc && acc.confusionMatrixValues;
  if (!cm) return [p('Test confusion matrix: not available yet.')];
  const labels = ['abnormal', 'background', 'normal'].filter((l) => cm[l]);
  const cols = [...labels, 'uncertain'];
  const rows = labels.map((l) => {
    const total = cols.reduce((s, c) => s + (cm[l][c] || 0), 0) || 1;
    return [l, ...cols.map((c) => `${cm[l][c] || 0} (${Math.round(((cm[l][c] || 0) / total) * 100)} %)`)];
  });
  const uncertain = labels.reduce((s, l) => s + (cm[l].uncertain || 0), 0);
  const good = get(acc, 'totalSummary.good') || 0;
  const bad = get(acc, 'totalSummary.bad') || 0;
  const decided = good + bad - uncertain;
  const w = Math.floor(R.TEXT_W / (cols.length + 1));
  return [
    p([{ text: 'Test confusion matrix (int8, 1 s windows)', bold: true }]),
    table(['actual \\ predicted', ...cols], rows, Array(cols.length + 1).fill(w)),
    R.caption('“uncertain” = top class below the 0.6 confidence threshold; Edge Impulse counts these windows as errors'),
    p(`Of ${good + bad} test windows, ${uncertain} fall below the confidence threshold. Among the windows the model does decide on, ${pct(decided ? good / decided : undefined)} are correct.`),
  ];
}

function build() {
  const team = TEAM.members || [{ name: '[Member 1 name]', id: '[Roll no.]' }, { name: '[Member 2 name]', id: '[Roll no.]' },
    { name: '[Member 3 name]', id: '[Roll no.]' }];
  const course = TEAM.course || '[Course name] — Edge AI';
  const professor = TEAM.professor || '[Professor name]';
  const institute = TEAM.institute || '[Institute / Department]';
  const date = TEAM.date || 'November 2026';
  const eiLink = TEAM.edge_impulse_url || 'https://studio.edgeimpulse.com/public/1164931/latest';
  const ghLink = 'https://github.com/AdesharaBrijesh/edge-ai-machine-sound-anomaly';

  const cover = [
    new Paragraph({ children: [], spacing: { before: 2400 } }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun({ text: 'Edge AI Predictive Maintenance', size: 48, bold: true, color: R.ACCENT })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 },
      children: [new TextRun({ text: 'Industrial Pump Sound Anomaly Detection with Edge Impulse', size: 32, color: '404040' })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
      children: [new TextRun({ text: 'Project report', size: 24, italics: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
      children: [new TextRun({ text: course, size: 24 })] }),
    ...team.map((m) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
      children: [new TextRun({ text: `${m.name}  ·  ${m.id}`, size: 24 })] })),
    new Paragraph({ children: [], spacing: { after: 480 } }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
      children: [new TextRun({ text: `Guided by: ${professor}`, size: 22 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
      children: [new TextRun({ text: institute, size: 22 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: date, size: 22 })] }),
    pageBreak(),
  ];

  const valAcc8 = train('int8', 'accuracy');
  const valAcc32 = train('float32', 'accuracy');
  const testA = testAcc();

  const body = [
    h1('Abstract'),
    p(`Industrial pumps usually announce a fault through their sound long before they stop. This project builds an Edge AI system that listens to a pump and decides, one second at a time, whether it sounds normal, abnormal, or whether no machine is running (background). A Gaussian-mixture anomaly score additionally flags sounds the model has never heard. The model is designed in Edge Impulse following the workflow of Edge Impulse’s Industrial Productivity Solutions Guide (data, DSP, learning block, EON Tuner, testing, deployment, automation, collaboration), using recordings of real pumps from the DCASE 2020 Task 2 / MIMII dataset plus our own laptop-microphone recordings. The int8 model reaches ${pct(valAcc8)} validation accuracy; on held-out recordings it classifies ${pct(D('test_clip_level.clip_accuracy'))} of 10-second clips correctly by majority vote (${pct(testA)} of individual one-second windows). On an Arduino Nano 33 BLE Sense (Cortex-M4F, 64 MHz) the int8 classifier is estimated at ${fmt(perf('nn_ms'), 0, ' ms')} per window with ${fmt(perf('classifier_int8.ram_kb'), 1, ' KB')} RAM, about ${perf('classifier_float32.latency_ms') && perf('nn_ms') ? Math.round(perf('classifier_float32.latency_ms') / perf('nn_ms')) : '—'}× faster than the float32 version at the same accuracy. Without any IoT hardware, the same impulse is deployed twice on a laptop acting as the edge device: as a C++ library compiled and run on Linux (exactly as in the reference guide) and as a WebAssembly module in a live browser dashboard that listens to the microphone. Retraining and redeployment are automated with the Edge Impulse API and a scheduled GitHub Actions workflow.`),

    h1('1  Introduction'),
    p('Unplanned downtime of pumps, fans and valves is expensive, and most mechanical faults (leaks, clogging, contamination, bearing or impeller damage) change the sound of a machine before they cause a failure. Streaming that audio to the cloud for analysis is costly in bandwidth, raises privacy concerns in a factory, and fails when the network does. Edge AI moves the analysis to a small device mounted on the machine: the microcontroller records the sound, extracts features and runs a compact neural network locally, sending only the result (normal / abnormal / unknown) upstream.'),
    p('Edge Impulse is a platform for building such pipelines. Its central concept is the impulse: an input block that windows the raw signal, one or more processing (DSP) blocks that turn each window into features, and learning blocks that classify those features. The whole impulse is compiled into dependency-free C++ that runs on microcontrollers, Linux devices or in a browser.'),
    p('Our professor’s reference, the Edge Impulse Industrial Productivity Solutions Guide, demonstrates this workflow on motor-current data from a sensorless AC drive. This project applies the same workflow to a different and very practical sensing modality, airborne sound, and to a different machine, an industrial water pump.'),

    h1('2  Objectives'),
    ...bullets([
      'Classify one-second windows of pump sound as normal, abnormal or background (no machine running).',
      'Flag sounds unlike anything in the training data (unknown events) with an anomaly score.',
      'Fit the model on a low-power microcontroller and run in real time (inference faster than the 500 ms window stride).',
      'Deploy and demonstrate the model without IoT hardware: compiled C++ on Linux and a live WebAssembly dashboard on a laptop.',
      'Reproduce every stage of the reference guide, replacing Enterprise-only features with free equivalents, and automate retraining and redeployment.',
    ]),

    h1('3  Reference workflow and how this project maps to it'),
    p('The free Edge Impulse Developer plan (GPU training, 60-minute jobs, three private projects, three collaborators) covers the impulse workflow but not the Enterprise features the guide uses for data management (organization datasets backed by S3, transformation blocks, organization data pipelines) or its LightGBM learning block. Table 1 shows how each part of the guide is covered here.'),
    table(['Reference guide', 'This project'], [
      ['Organization dataset + S3 bucket + transformation block (txt → JSON)', 'prepare_dataset.py: selects, validates, splits and renames the raw WAV clips (same role as the transformation block)'],
      ['CSV Wizard / data forwarder (direct ingestion)', 'upload_to_ei.py via the ingestion API; record_mic_path.py records directly from the laptop microphone'],
      ['Spectral Analysis + Flatten processing blocks', 'MFE (mel filterbank energy) for the classifier + Spectral Analysis for the anomaly block'],
      ['LightGBM learning block (Enterprise)', '1D convolutional neural network + Gaussian-mixture (GMM) anomaly detection'],
      ['On-device performance estimates (nRF5340 DK)', 'Same feature, target Arduino Nano 33 BLE Sense (has an on-board microphone)'],
      ['EON Tuner, model testing', 'Same'],
      ['C++ library + example-standalone-inferencing-linux', 'Same, compiled and run in WSL2 Ubuntu on the laptop'],
      ['Organization data pipeline (import + transform)', 'ei_pipeline.py on a GitHub Actions schedule (every 2 days)'],
      ['Project pipeline: retrain, version, deploy', 'Same native Edge Impulse feature (Data sources → pipeline)'],
      ['Collaborators, project README', 'Same; project and code are public'],
    ], [4300, 4726]),
    R.caption('Table 1 — Reference guide sections and their equivalents in this project'),

    h1('4  Dataset'),
    h2('4.1  Pump recordings (DCASE 2020 Task 2 / MIMII)'),
    p('The machine sounds come from the development set of the DCASE 2020 Challenge Task 2, “Unsupervised detection of anomalous sounds for machine condition monitoring”, which is built from Hitachi’s MIMII dataset. Each clip is a 10-second, 16 kHz, single-channel recording of a real water pump in a factory, with real factory background noise mixed in. Anomalous clips were recorded after deliberately introducing faults such as leakage, contamination and clogging. Four different pump units are included (machine IDs 00, 02, 04, 06). The pump subset contains 3,749 normal and 456 anomalous clips (1.0 GB); the data is licensed CC BY-NC-SA 4.0.'),
    ...fig('spectrograms.png', 'Figure 1 — Log-mel spectrograms of a normal and an abnormal clip from the same pump: the fault adds broadband transients and changes the tonal pattern.', 600),
    h2('4.2  Subset and split'),
    p('prepare_dataset.py draws a balanced subset of 300 normal and 300 abnormal clips, spread evenly over the four pump units, and splits it 80/20 into training and testing, stratified by class and unit. Ten further clips (five per class) are held back entirely for the live demonstration and are never uploaded. Every file is checked for format (mono, 16 kHz, 16-bit PCM, 10 s) and renamed <label>.<machine>-id<ID>-<origin>-<index>.wav so the label travels with the file.'),
    table(['Set', 'normal', 'abnormal', 'background', 'Source'], [
      ['Training (dataset)', dcase('training/normal'), dcase('training/abnormal'), '—', 'DCASE 2020 pump clips'],
      ['Testing (dataset)', dcase('testing/normal'), dcase('testing/abnormal'), '—', 'DCASE 2020 pump clips'],
      ['Training (laptop mic)', mic('training/normal'), mic('training/abnormal'), mic('training/background'), 'Replayed clips + room recordings'],
      ['Testing (laptop mic)', mic('testing/normal'), mic('testing/abnormal'), mic('testing/background'), 'Replayed clips + room recordings'],
      ['Noise-augmented (training)', '', '', '', `${D('dataset.augmented') || 0} clips mixed with room noise`],
      ['Live-demo hold-out', dcase('demo/normal'), dcase('demo/abnormal'), '—', 'never uploaded'],
    ], [2400, 1100, 1100, 1300, 3126]),
    R.caption('Table 2 — Number of 10-second clips per set and class'),
    ...shot('02-data-acquisition.png', 'Figure 2 — The dataset in Edge Impulse (Data acquisition)', 'Data acquisition page with the train/test split'),

    h2('4.3  Laptop-microphone recordings'),
    p('The model must eventually listen through a real microphone. A model trained only on the clean dataset plus a background class recorded with the laptop microphone can learn a shortcut: “anything that sounds like the laptop microphone is background”. To remove that shortcut we re-recorded 60 pump clips through the laptop’s own speaker and microphone (record_mic_path.py), so normal and abnormal examples exist with the same acoustic colouring as the background, and recorded about eight minutes of the room (silence, speech, typing, footsteps) as the background class. Each re-recording is aligned to its source by cross-correlation; the median correlation with the original clip was 0.47 (range 0.26–0.68).'),
    ...fig('mic_path.png', 'Figure 3 — Average spectrum of a clip before and after the laptop speaker-to-microphone path: the low frequencies (below about 300 Hz) lose 15–20 dB.', 600),
    note('Practical finding: Windows applies voice-oriented “audio enhancements” (echo cancellation and noise suppression) to laptop microphone arrays. On this laptop they erased the speaker sound almost completely (the level fell from −94 to −103 dBFS while the speaker played noise), whereas the raw kernel-streaming device heard it clearly (+20 dB). A machine-monitoring product must use a raw microphone path, as a dedicated MEMS microphone on a microcontroller does; for the demo the enhancements were switched off.'),
    h2('4.4  Noise augmentation'),
    p('augment_noise.py mixes the recorded room noise into dataset training clips at a random signal-to-noise ratio between 5 and 20 dB (the same idea as Edge Impulse’s mix-noise transformation block), so the classifier keeps working when people talk or a fan runs nearby.'),

    h1('5  Data pipeline'),
    ...numbered([
      'download_dataset.py downloads the pump archive from Zenodo (with resume and size check) and extracts it into its own folder.',
      'prepare_dataset.py selects, validates, splits and renames clips and writes manifest.csv (the transformation step).',
      'upload_to_ei.py sends the files to the Edge Impulse ingestion API in batches, always with an explicit x-label header, then reads the sample counts back from the project to verify them against the manifest.',
      'record_mic_path.py and augment_noise.py create the laptop-microphone and noise-augmented sets, uploaded the same way.',
    ]),
    p('All uploads were verified: the project holds exactly the clips listed in the manifest, and no source clip appears in both the training and the testing set.'),

    h1('6  Impulse design'),
    p('The impulse (Figure 4) was created and configured through the Edge Impulse Studio API by build_impulse.py, following Edge Impulse’s official Python API example, and can equally be inspected and edited in Studio.'),
    ...shot('04-impulse-design.png', 'Figure 4 — The impulse: audio input, MFE and Spectral Analysis processing blocks, classifier and anomaly detection', 'Impulse design → Create impulse'),
    h2('6.1  Input block'),
    p('Audio at 16 kHz is cut into 1,000 ms windows every 500 ms (zero-padding enabled). One-second windows give the network several rotations of the pump and keep the raw input at 16,000 samples (32 KB as int16), small enough for a microcontroller. A 10-second clip yields 19 overlapping training windows.'),
    h2('6.2  MFE processing block (for the classifier)'),
    p('Mel-filterbank energy (MFE) features suit non-voice audio: they keep the spectral envelope on a perceptual frequency scale without the speech-oriented decorrelation of MFCCs. Each window becomes 61 frames × 40 mel bands = 2,440 features.'),
    table(['Parameter', 'Value', 'Why'], [
      ['Frame length / stride', '32 ms / 16 ms', '512-sample FFT frames with 50 % overlap'],
      ['Filters', '40', 'Enough resolution for machine harmonics'],
      ['FFT length', '512', 'Matches the frame length'],
      ['Frequency range', '0–8000 Hz', 'Full band at 16 kHz'],
      ['Noise floor', '−72 dB', 'Keeps quiet machine detail; clips silence'],
    ], [2600, 2200, 4226]),
    R.caption('Table 3 — MFE parameters'),
    h2('6.3  Spectral Analysis block (for the anomaly detector)'),
    p('A second processing block computes a 128-point FFT spectrum (log power, overlapping frames) plus statistical features. This compact representation feeds the Gaussian-mixture anomaly detector, mirroring the guide’s use of two processing blocks whose outputs feed different learners.'),
    h2('6.4  Learning blocks'),
    table(['Layer', 'Configuration'], [
      ['Input', '2,509 features (MFE + spectral); a cropping layer passes only the 2,440 MFE values on, reshaped to 61 × 40'],
      ['1D convolution + max-pool', '16 filters, kernel 3'],
      ['Dropout', '0.25'],
      ['1D convolution + max-pool', '32 filters, kernel 3'],
      ['Dropout', '0.25'],
      ['Flatten → Dense (softmax)', 'one output per class'],
      ['Training', '100 epochs, Adam, learning rate 0.001, batch 32, 20 % validation, class weights, full shuffle each epoch, best epoch kept'],
    ], [3200, 5826]),
    R.caption('Table 4 — Classifier architecture and training settings'),
    p('The anomaly block is a Gaussian mixture model (5 components) fitted to the spectral features of all training windows. At inference it reports how unlikely a new window is under that model; windows far from everything seen in training (a hair dryer, a drill, a different machine) receive a high anomaly score even though the classifier must still pick one of its known classes.'),

    h1('7  Training results'),
    note('Debugging lesson: the first training runs stayed at exactly 50 % accuracy with a loss of 0.693 (= ln 2) for all 100 epochs, and every window was predicted as the same class. The features were not the problem: a simple logistic regression on per-band MFE statistics, computed from the features downloaded through the API, separated the classes with 78 % accuracy on held-out clips. The cause was data order. Edge Impulse delivers the training windows grouped by class, and the default training script shuffles with a buffer of only 4 × batch size (128 windows), so almost every batch contained a single class and the network only chased the label of the current batch. Switching the learning block to expert mode and shuffling the complete training set every epoch, one changed line, fixed it. A second run then showed that Studio’s default learning rate of 0.005 was too high for this data: validation accuracy swung between about 57 % and 80 % and collapsed to one class after 75 epochs. Lowering it to 0.001 gave the stable final model (Figure 5).'),
    ...fig('shuffle_fix.png', 'Figure 5 — Validation accuracy per epoch: default shuffle (stuck at chance), full shuffle with learning rate 0.005 (unstable) and the final run (full shuffle, 0.001)', 600),
    table(['Metric', 'float32', 'int8 (deployed)'], [
      ['Validation accuracy', pct(valAcc32), pct(valAcc8)],
      ['Validation loss', fmt(train('float32', 'loss'), 3), fmt(train('int8', 'loss'), 3)],
    ], [3400, 2800, 2826]),
    R.caption('Table 5 — Validation results of the classifier'),
    ...confusionTable(D('train_main.int8'), 'Validation confusion matrix (int8)'),
    ...fig('training_curves.png', 'Figure 6 — Final model: training and validation accuracy and loss per epoch', 600),
    ...shot('08-classifier.png', 'Figure 7 — Classifier page in Edge Impulse: accuracy, confusion matrix and on-device performance', 'Classifier page after training'),

    h1('8  Model testing'),
    p(`Model testing runs the int8 impulse on the held-out test set (clips never seen in training). Overall accuracy is ${pct(testA)}.`),
    ...testConfusion(),
    ...shot('10-model-testing.png', 'Figure 8 — Model testing results in Edge Impulse', 'Model testing → Classify all'),
    table(['System', 'Task', 'Result'], [
      ['This project (int8, MFE + 1D CNN)', 'normal / abnormal / background, 1 s windows', `${pct(testA)} test accuracy`],
      ['Public Edge Impulse pump project (Manivannan)', 'normal / abnormal, MFE + NN', '93.5 % test accuracy (97.8 % validation)'],
      ['DCASE 2020 baseline (autoencoder)', 'unsupervised, normal-only training', '72.9 % AUC on pump'],
    ], [3600, 3000, 2426]),
    R.caption('Table 6 — Comparison with related work (the DCASE baseline solves a harder, unsupervised task)'),

    h1('9  Anomaly detection'),
    p('The GMM anomaly score separates sounds the model knows from sounds it has never heard. Known classes receive low scores; unknown sounds (e.g. a hair dryer or a vacuum cleaner played near the microphone) exceed the threshold and the dashboard reports UNKNOWN SOUND instead of forcing a wrong class.'),
    ...fig('anomaly_scores.png', 'Figure 9 — Anomaly scores of known classes and unknown sounds', 600),
    ...shot('09-anomaly.png', 'Figure 10 — Anomaly detection (GMM) in Edge Impulse', 'Anomaly detection page'),

    h1('10  EON Tuner'),
    p('The EON Tuner searches combinations of processing parameters and network architectures under the target’s latency and memory constraints. It was run for the Arduino Nano 33 BLE Sense with a latency budget below the 500 ms window stride.'),
    ...(tuner().rows ? [table(['Configuration', 'Accuracy', 'Latency', 'RAM', 'Flash'], tuner().rows, [3400, 1400, 1400, 1400, 1426]),
      R.caption('Table 7 — Best EON Tuner configurations')] : [p('EON Tuner results: see Figure 11.')]),
    ...shot('11-eon-tuner.png', 'Figure 11 — EON Tuner results', 'EON Tuner page after the run'),

    h1('11  Deployment'),
    h2('11.1  On-device estimates and real-time feasibility'),
    table(['Block (Arduino Nano 33 BLE Sense, Cortex-M4F 64 MHz)', 'Latency', 'RAM', 'Flash'], [
      ['MFE processing block', perf('mfe_ms') ? fmt(perf('mfe_ms'), 0, ' ms') : 'Figure B3', perf('mfe_ram_kb') ? fmt(perf('mfe_ram_kb'), 1, ' KB') : '—', '—'],
      ['Classifier, int8 + EON compiler (deployed)', fmt(perf('classifier_int8.latency_ms'), 0, ' ms'), fmt(perf('classifier_int8.ram_kb'), 1, ' KB'), fmt(perf('classifier_int8.flash_kb'), 1, ' KB')],
      ['Classifier, float32 (for comparison)', fmt(perf('classifier_float32.latency_ms'), 0, ' ms'), fmt(perf('classifier_float32.ram_kb'), 1, ' KB'), fmt(perf('classifier_float32.flash_kb'), 1, ' KB')],
      ['GMM anomaly detector', fmt(perf('gmm_ms'), 0, ' ms'), fmt(perf('anomaly_float32.ram_kb'), 1, ' KB'), fmt(perf('anomaly_float32.flash_kb'), 1, ' KB')],
    ], [4626, 1400, 1500, 1500]),
    R.caption('Table 8 — Estimated on-device performance per block (Edge Impulse profiling)'),
    p(`For continuous monitoring a new window starts every 500 ms, so the MFE features and both models must finish within 500 ms. The two models together take ${fmt(perf('learn_ms'), 0, ' ms')}, leaving almost the whole budget to feature extraction. Quantising the classifier to int8 cut its latency from ${fmt(perf('classifier_float32.latency_ms'), 0, ' ms')} to ${fmt(perf('nn_ms'), 0, ' ms')} at the same validation accuracy. For comparison, the public reference pump model needs 555 ms per inference on a Cortex-M4F at 80 MHz and would fall behind a 500 ms stride.`),
    ...shot('12-deployment.png', 'Figure 12 — Deployment page: C++ library with the EON compiler', 'Deployment page'),
    note('Deployment lesson: the first C++ and WebAssembly builds failed at run time with “Would write outside feature buffer”. The Edge Impulse SDK writes the output of every processing block, one after another, into a single buffer sized by the classifier’s input. Our classifier read only the MFE features (2,440 values), so the 69 spectral features of the anomaly block overflowed it. Feeding both processing blocks to the classifier, and cropping its input back to the MFE values inside the network (a standard Cropping1D layer), sized the buffer correctly without changing the model.'),
    h2('11.2  C++ library on Linux (as in the reference guide)'),
    p('The C++ library (EON compiler, int8) was built with Edge Impulse’s example-standalone-inferencing-linux project inside WSL2 Ubuntu on the laptop, exactly as in the guide, and run on one window of raw features taken from a held-out abnormal clip:'),
    ...code(['$ APP_CUSTOM=1 make -j$(nproc)', '$ ./build/custom features-abnormal.txt', ...((cpp().output_lines) || ['(output inserted after the build)'])]),
    h2('11.3  WebAssembly live dashboard'),
    p('The same impulse, built as WebAssembly, runs inside a browser page served from the laptop (demo/). In replay mode it plays a held-out clip through the speakers while classifying the exact same samples every 250 ms; in live mode it classifies the microphone stream. The page shows class probabilities, the anomaly score against an adjustable threshold, the inference time, a timeline and a live spectrogram.'),
    ...fig('dashboard.png', 'Figure 13 — Live dashboard classifying a held-out abnormal clip in the browser', 600),
    ...(demo().replay_summary ? [p(demo().replay_summary)] : []),

    h1('12  Automation'),
    p('The guide automates two pipelines; both are reproduced here:'),
    ...bullets([
      [{ text: 'Data import and retraining: ', bold: true }, 'ei_pipeline.py uploads any new labelled recordings, starts a retrain job (which regenerates the DSP features and retrains every learning block), builds the C++ library and the WebAssembly package with the EON compiler, and downloads both. A GitHub Actions workflow runs it every two days (cron 0 3 */2 * *) and stores the libraries as build artifacts.'],
      [{ text: 'Project pipeline: ', bold: true }, 'Edge Impulse’s native Data sources pipeline (“Don’t import data”) retrains the model, creates a new project version and builds a new C++ deployment on a schedule — the same configuration as the guide’s Figures 32–33.'],
    ]),
    ...shot('14-pipeline.png', 'Figure 14 — Native Edge Impulse pipeline: retrain, create version, build deployment', 'Data acquisition → Data sources'),

    h1('13  Collaboration and reproducibility'),
    p(`All team members are collaborators on the Edge Impulse project, whose README documents the data origin and processing. The project is public (${eiLink}) and the complete code, with instructions to reproduce every step, is public on GitHub (${ghLink}).`),

    h1('14  Limitations and future work'),
    ...bullets([
      'No physical microcontroller was used; on-device numbers are Edge Impulse estimates and the laptop stands in for the device. The next step is flashing the same library to an Arduino Nano 33 BLE Sense and listening with its on-board microphone.',
      'Training and test clips come from the same four pump units. A stricter test holds out a whole unit (prepare_dataset.py --holdout-id 06).',
      'The classifier needs labelled fault recordings; for faults never recorded, the GMM score (or an unsupervised model trained on normal sound only, as in DCASE) must carry the decision.',
      'Laptop speakers cannot reproduce the pump’s low frequencies, so the live demo hears a band-limited version of the machine.',
      'More machine types (fans, valves, slide rails) and Edge Impulse’s performance calibration for continuous audio would make the system broader and quieter in production.',
    ]),

    h1('15  Conclusion'),
    p('Following the reference guide end to end on a free plan and without IoT hardware, we built, evaluated and deployed an Edge AI pump-sound monitor: a 1D CNN on MFE features that distinguishes normal, abnormal and background audio, a GMM anomaly detector for unknown sounds, a C++ build running on Linux, a live WebAssembly dashboard, and an automated retrain-and-deploy loop. Recording our own microphone data revealed two practical lessons that the clean dataset hides: the model must be trained on the acoustic path it will hear in deployment, and consumer audio processing can silently erase the very sounds a monitor needs.'),

    h1('References'),
    ...numbered([
      'Edge Impulse, “Industrial Productivity Solutions Guide: Adding a sixth sense to your industrial machines” (I. Turasov).',
      'Y. Koizumi et al., “Description and discussion on DCASE2020 Challenge Task 2: Unsupervised anomalous sound detection for machine condition monitoring,” DCASE Workshop, 2020. Dataset: Zenodo record 3678171.',
      'H. Purohit et al., “MIMII Dataset: Sound dataset for malfunctioning industrial machine investigation and inspection,” DCASE Workshop, 2019.',
      'Edge Impulse documentation: ingestion API, processing blocks (MFE, Spectral Analysis), anomaly detection (GMM), EON Tuner, deployment — docs.edgeimpulse.com.',
      'Edge Impulse, example-standalone-inferencing-linux, notebooks (Python API bindings example), example-transform-mix-audio — github.com/edgeimpulse.',
      'Manivannan, “Industry 4.0 – Predictive Maintenance” (Hackster.io) and public Edge Impulse projects 88093 / 88799 / 88839.',
      'Y. Kawaguchi, DCASE 2020 Task 2 baseline system — github.com/y-kawagu/dcase2020_task2_baseline.',
    ], 'refs'),

    pageBreak(),
    h1('Appendix A — Contributions'),
    p('To be completed by the team.'),
    table(['Team member', 'Contributions'], team.map((m) => [m.name, ' ']), [3000, 6026]),

    h1('Appendix B — Additional Edge Impulse screenshots'),
    ...shot('01-dashboard.png', 'Figure B1 — Project dashboard with the target device (Arduino Nano 33 BLE Sense)', 'Dashboard → Project info'),
    ...shot('05-mfe-parameters.png', 'Figure B2 — MFE parameters and processed spectrogram', 'Impulse design → MFE → Parameters'),
    ...shot('06-mfe-features.png', 'Figure B3 — MFE feature explorer and on-device processing estimate', 'MFE → Generate features'),
    ...shot('07-spectral-features.png', 'Figure B4 — Spectral features with feature importance', 'Spectral features → Generate features'),
    ...shot('13-versioning.png', 'Figure B5 — Saved project versions', 'Versioning'),
    ...shot('15-collaborators.png', 'Figure B6 — Collaborators and project README', 'Dashboard'),
    ...shot('16-launch-in-browser.png', 'Figure B7 — Edge Impulse “Launch in browser” live classification', 'Dashboard → Run this model'),

    h1('Appendix C — Reproducing the project'),
    ...code([
      'pip install -r requirements.txt edgeimpulse-api',
      'python scripts/download_dataset.py --machine pump',
      'python scripts/prepare_dataset.py --machine pump --max-per-class 300',
      'python scripts/upload_to_ei.py',
      'python scripts/record_mic_path.py replay --input <raw mic>',
      'python scripts/record_mic_path.py background --input <raw mic> --minutes 8',
      'python scripts/augment_noise.py',
      'python scripts/build_impulse.py all',
      'python scripts/ei_pipeline.py --skip-upload --skip-retrain',
      'bash deployment/cpp-linux/build.sh   # in WSL',
      'python demo/serve.py                  # http://localhost:8000',
    ]),
  ];

  const doc = new Document({
    creator: 'Edge AI project team',
    title: 'Industrial Pump Sound Anomaly Detection with Edge Impulse',
    styles: {
      default: { document: { run: { font: R.FONT, size: 22 } } },
      paragraphStyles: [
        { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: R.FONT, size: 30, bold: true, color: R.ACCENT },
          paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
        { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: R.FONT, size: 25, bold: true, color: '2E5C8A' },
          paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
        { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: R.FONT, size: 23, bold: true, color: '404040' },
          paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 2 } },
      ],
    },
    numbering: {
      config: [
        { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
        { reference: 'steps', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
        { reference: 'refs', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '[%1]', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 540 } } } }] },
      ],
    },
    sections: [
      { properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: cover },
      {
        properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
        headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: 'Industrial Pump Sound Anomaly Detection — Edge AI project', size: 16, color: '808080' })] })] }) },
        footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: ['Page ', PageNumber.CURRENT, ' of ', PageNumber.TOTAL_PAGES], size: 16, color: '808080' })] })] }) },
        children: [
          new Paragraph({ children: [new TextRun({ text: 'Contents', bold: true, size: 30, color: R.ACCENT })], spacing: { after: 200 } }),
          new TableOfContents('Contents', { hyperlink: true, headingStyleRange: '1-2' }),
          pageBreak(),
          ...body,
        ],
      },
    ],
  });
  return Packer.toBuffer(doc).then((buf) => {
    fs.writeFileSync(R.OUT, buf);
    console.log(`wrote ${path.relative(R.ROOT, R.OUT)} (${(buf.length / 1024).toFixed(0)} KB)`);
  });
}

module.exports = { build };
