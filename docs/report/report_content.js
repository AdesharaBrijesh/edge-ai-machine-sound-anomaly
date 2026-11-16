// Content of the project report, laid out in the RRU dissertation format (see build_report.js).
// Numbers come from report_data.json (make_figures.py); cross-references use ref('label').
'use strict';
const fs = require('fs');
const path = require('path');
const R = require('./build_report.js');
const {
  D, pct, fmt, first, cite, p, bullets, numbered, pageBreak, spacer, chapter, appendix, section, subsection,
  frontTitle, code, table, fig, shot, ref, TEAM, TEXT_W, FONT, FRONT_FONT, LINE,
} = R;
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, Header, Footer, PageNumber, TableOfContents, StyleLevel,
  LevelFormat, ImageRun, NumberFormat, TabStopType, SectionType,
} = R.docx;

// ---------- metric accessors (tolerant to missing data) ----------
const train = (variant, key) => D(`train_main.${variant}.${key}`);
const trainV1 = (variant) => first(D(`train_lr001_mfeonly.${variant}.accuracy`), D(`train_lr001_mfeonly.${variant}`));
// test-set results computed with the compiled C++ impulse (deployment/cpp-linux/eval.sh)
const ev = (v, k) => D(`cpp_eval_${v}.${k}`);
const testAcc = () => ev('v2', 'overall.window_accuracy');
const dcase = (k) => D(`dataset.dcase.${k}`) || 0;
const mic = (k) => D(`dataset.mic.${k}`) || 0;
const perf = (k) => D(`performance.${k}`);
// RAM / flash: the final profile did not report memory; the network is identical to v1, so v1's figures apply
const mem = (k) => (perf(k) ? perf(k) : D(`performance_v1.${k}`));
const tuner = () => D('eon_tuner') || {};
const cpp = () => D('cpp_run') || {};

const TITLE = 'Edge AI Predictive Maintenance: Industrial Pump Sound Anomaly Detection with Edge Impulse';
const SHORT_TITLE = 'Industrial Pump Sound Anomaly Detection with Edge Impulse';

// ---------- front matter in the template's typography (Garamond, Old English Text MT) ----------
function g(text, size, opts = {}) {
  return new TextRun({ text, font: FRONT_FONT, size: size * 2, ...opts });
}
function centre(children, after = 0, before = 0) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before, after, line: 276 }, children });
}

function titlePage(team, info) {
  const logoFile = path.join(R.HERE, 'template_assets', 'rru_logo.png');
  const logo = fs.existsSync(logoFile)
    ? [centre([new ImageRun({ type: 'png', data: fs.readFileSync(logoFile), transformation: { width: 112, height: 112 },
      altText: { title: 'Rashtriya Raksha University logo', description: 'Rashtriya Raksha University logo', name: 'rru_logo' } })], 120, 240)]
    : [centre([g('[RRU logo]', 12)], 120, 240)];
  return [
    centre([g(TITLE.toUpperCase(), 14, { bold: true })], 360, 240),
    centre([g('Project Report submitted', 14)], 0),
    centre([g('to', 14)], 120),
    centre([g('RASHTRIYA RAKSHA UNIVERSITY', 16, { bold: true })], 0),
    centre([g('(An Institution of National Importance)', 12)], 240),
    centre([g('For the partial fulfilment for the award of the degree of', 14)], 120),
    centre([new TextRun({ text: info.degree, font: 'Old English Text MT', size: 56, bold: true })], 0),
    centre([g(info.specialisation, 14)], 300),
    centre([g('Submitted by', 14)], 100),
    ...team.map((m) => centre([g(`${m.name.toUpperCase()} (Enrolment No. ${m.id})`, 12, { bold: true })], 40)),
    centre([], 200),
    centre([g('Under the Guidance of', 14)], 100),
    centre([g(info.professor.toUpperCase(), 12, { bold: true })], 0),
    centre([g(`${info.designation},`, 12)], 0),
    centre([g(`${info.schoolShort},`, 12)], 0),
    centre([g('Lavad', 12)], 0),
    ...logo,
    centre([g(info.school.toUpperCase(), 12, { bold: true })], 0),
    centre([g('RASHTRIYA RAKSHA UNIVERSITY,', 12, { bold: true })], 0),
    centre([g('Lavad, Dehgam, Gandhinagar-382305, Gujarat, India', 12, { bold: true })], 0),
    centre([g(info.date, 12, { bold: true })], 0),
  ];
}

function formParagraph(children, opts = {}) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200, line: LINE }, children, ...opts });
}

function declaration(team, info) {
  return [
    new Paragraph({ style: 'FrontTitle', children: [g('DECLARATION', 14, { bold: true })] }),
    formParagraph([g(`The work embodied in this Project Report titled “${TITLE}” submitted for the partial fulfillment of the degree of ${info.degree} (${info.specialisationShort}) is the original work carried out by us. The work does not form the basis for the award of any degree, diploma, associateship, fellowship or other titles in the Rashtriya Raksha University or similar institutions of higher learning. All the ideas and references have been duly acknowledged.`, 12)]),
    spacer(), spacer(),
    ...team.map((m) => new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W, leader: 'dot' }],
      spacing: { after: 360, line: LINE },
      children: [g(`${m.name} (${m.id})`, 12), g('\t', 12)],
    })),
    new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 240 }, children: [g('(Name & Signature of the Candidates)', 12, { bold: true })] }),
    new Paragraph({ children: [g('Date:', 12, { bold: true })] }),
  ];
}

function certificate(team, info) {
  const names = team.map((m) => `${m.name} (Enrolment No. ${m.id})`);
  const list = names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}` : names[0];
  return [
    new Paragraph({ style: 'FrontTitle', pageBreakBefore: true, children: [g('CERTIFICATE', 14, { bold: true })] }),
    formParagraph([g(`This is to certify that the Project Report titled “${TITLE}” was carried out by ${list}, studying at the ${info.school} (${info.schoolShort}) for partial fulfillment of the ${info.degree} (${info.specialisationShort}) degree to be awarded by Rashtriya Raksha University. This work has been carried out under my guidance and supervision and it is up to my satisfaction. The Project Report is fit to be considered for evaluation for the degree of ${info.degree} (${info.specialisationShort}).`, 12)]),
    spacer(),
    new Paragraph({ spacing: { after: 120 }, children: [g('Date:', 12)] }),
    new Paragraph({ spacing: { after: 1200 }, children: [g('Place: Lavad', 12)] }),
    new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }],
      spacing: { after: 0 },
      children: [g('Signature and Name of Supervisor', 12, { bold: true }), g('\tSignature and Name of School Director', 12, { bold: true })],
    }),
    new Paragraph({ spacing: { after: 1200 }, children: [g(info.professor, 12)] }),
    centre([g('Round Seal of RRU', 12, { bold: true })]),
  ];
}

function acknowledgements(info) {
  return [
    new Paragraph({ style: 'FrontTitle', pageBreakBefore: true, children: [new TextRun('Acknowledgements')] }),
    p(`We are grateful to our guide, ${info.professor}, for introducing us to Edge AI and for pointing us to the Edge Impulse Industrial Productivity Solutions Guide, which shaped every stage of this project. We thank the ${info.school}, Rashtriya Raksha University, for the course and the facilities that made the work possible.`),
    p('We thank Hitachi, Ltd. and the organisers of the DCASE 2020 Challenge for releasing the MIMII pump recordings under an open licence, and Edge Impulse for the free Developer plan, its documentation and its open-source examples, on which our scripts build.'),
    p('Finally, we thank our families and classmates for their patience and encouragement.'),
  ];
}

function abstractPage(team, info) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, pageBreakBefore: true, spacing: { after: 240, line: LINE },
      children: [new TextRun({ text: `Abstract: ${TITLE}`, bold: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun('Submitted By')] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun(team.map((m) => m.name).join(', '))] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun(`Supervised By ${info.professor}, ${info.designation}, ${info.schoolShort}`)] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 }, children: [new TextRun({ text: 'Abstract', bold: true })] }),
    p([{ text: 'Objective. ', bold: true }, 'Industrial pumps change their sound before they fail. This project builds an Edge AI system that classifies one-second windows of pump sound as normal, abnormal or background, small enough for a microcontroller, following Edge Impulse’s Industrial Productivity Solutions Guide without any IoT hardware.']),
    p([{ text: 'Work done. ', bold: true }, 'Real pump recordings from the DCASE 2020 Task 2 / MIMII dataset (600 clips, four pump units) were combined with laptop-microphone re-recordings, room background and noise-augmented clips, uploaded through the Edge Impulse API and processed with mel-filterbank energy (MFE) features and a 1D convolutional network quantised to int8. The impulse was deployed as a C++ library compiled on Linux, as in the guide, and as WebAssembly in a live browser dashboard; retraining and redeployment are automated.']),
    p([{ text: 'Results and discussion. ', bold: true }, `Scored with the deployed C++ library on ${ev('v2', 'overall.clips') || 141} held-out clips, the model labels ${pct(testAcc())} of one-second windows and ${pct(ev('v2', 'overall.clip_accuracy'))} of clips correctly and recognises every background window; the same network trained longer reached ${pct(ev('v1', 'overall.window_accuracy'))}. On an Arduino Nano 33 BLE Sense it is estimated at ${fmt(perf('nn_ms'), 0, ' ms')} per window. In-browser tests show it relies on the 300–600 Hz fault signature and, under added noise, errs only towards false alarms. Debugging showed that class-ordered data needs a full shuffle and that one processing block per impulse is needed for deployment.`]),
    p([{ text: 'Conclusions. ', bold: true }, 'A complete edge audio monitor can be designed, tested and deployed with free tools and a laptop. Detecting never-heard sounds needs an anomaly block, which is left for future work.']),
  ];
}

// ---------- body ----------
function body(team) {
  const valAcc8 = train('int8', 'accuracy');
  const valAcc32 = train('float32', 'accuracy');
  const testA = testAcc();
  const eiLink = TEAM.edge_impulse_url || 'https://studio.edgeimpulse.com/public/1165553/live';
  const ghLink = 'https://github.com/AdesharaBrijesh/edge-ai-machine-sound-anomaly';
  const v1Test = (() => {
    const good = D('test_v1.accuracy.totalSummary.good');
    const bad = D('test_v1.accuracy.totalSummary.bad');
    return typeof good === 'number' && typeof bad === 'number' ? good / (good + bad) : undefined;
  })();

  return [
    // ===================================================================== 1
    chapter(1, 'Introduction', 'ch:intro'),
    section('Background'),
    p('Unplanned downtime of pumps, fans and valves is expensive, and most mechanical faults (leaks, clogging, contamination, bearing or impeller damage) change the sound of a machine before they cause a failure. Streaming that audio to the cloud for analysis is costly in bandwidth, raises privacy concerns in a factory, and fails when the network does. Edge AI moves the analysis to a small device mounted on the machine: the microcontroller records the sound, extracts features and runs a compact neural network locally, sending only the result (normal / abnormal) upstream.'),
    p(['Edge Impulse is a platform for building such pipelines', cite(5), '. Its central concept is the impulse: an input block that windows the raw signal, one or more processing (DSP) blocks that turn each window into features, and learning blocks that classify those features. The whole impulse is compiled into dependency-free C++ that runs on microcontrollers, Linux devices or in a browser.']),
    p(['Our reference, the Edge Impulse Industrial Productivity Solutions Guide', cite(1), ', demonstrates this workflow on motor-current data from a sensorless AC drive. This project applies the same workflow to a different and very practical sensing modality, airborne sound, and to a different machine, an industrial water pump.']),
    section('Objectives'),
    ...bullets([
      'Classify one-second windows of pump sound as normal, abnormal or background (no machine running).',
      `Investigate an anomaly score for sounds unlike anything in the training data (${ref('sec:anomaly')} reports what worked and what did not).`,
      'Fit the model on a low-power microcontroller and run in real time (inference faster than the 500 ms window stride).',
      'Deploy and demonstrate the model without IoT hardware: compiled C++ on Linux and a live WebAssembly dashboard on a laptop.',
      'Reproduce every stage of the reference guide, replacing Enterprise-only features with free equivalents, and automate retraining and redeployment.',
    ]),
    section('Reference workflow and how this project maps to it', 'sec:mapping'),
    p(`The free Edge Impulse Developer plan (GPU training, 60-minute jobs, three private projects, three collaborators) covers the impulse workflow but not the Enterprise features the guide uses for data management (organization datasets backed by S3, transformation blocks, organization data pipelines) or its LightGBM learning block. ${ref('tab:mapping')} shows how each part of the guide is covered here.`),
    ...table('Reference guide sections and their equivalents in this project', ['Reference guide', 'This project'], [
      ['Organization dataset + S3 bucket + transformation block (txt → JSON)', 'prepare_dataset.py: selects, validates, splits and renames the raw WAV clips (same role as the transformation block)'],
      ['CSV Wizard / data forwarder (direct ingestion)', 'upload_to_ei.py via the ingestion API; record_mic_path.py records directly from the laptop microphone'],
      ['Spectral Analysis + Flatten processing blocks', 'MFE (mel filterbank energy) block; a second Spectral Analysis block for an anomaly detector was tried first'],
      ['LightGBM learning block (Enterprise)', '1D convolutional neural network (a Gaussian-mixture anomaly block was trained in the first version)'],
      ['On-device performance estimates (nRF5340 DK)', 'Same feature, target Arduino Nano 33 BLE Sense (has an on-board microphone)'],
      ['EON Tuner, model testing', 'Tuner prepared; final model tested with the deployed C++ library (compute limit)'],
      ['C++ library + example-standalone-inferencing-linux', 'Same, compiled and run in WSL2 Ubuntu on the laptop'],
      ['Organization data pipeline (import + transform)', 'ei_pipeline.py on a GitHub Actions schedule (every 2 days)'],
      ['Project pipeline: retrain, version, deploy', 'Same native Edge Impulse feature (Data sources → pipeline)'],
      ['Collaborators, project README', 'Public Edge Impulse project and public GitHub repository with a README'],
    ], [45, 55], 'tab:mapping'),
    section('Project workflow at a glance', 'sec:workflow'),
    p(`${ref('tab:workflow')} summarises what was done in which order, which tool did the work, and what each step produced. Edge Impulse provided the processing block, cloud training, quantisation, on-device estimates and the exported libraries; everything around it (data preparation, automation, evaluation of the deployed library and the monitoring dashboard) was built in this project.`),
    ...table('Project workflow: step, tool and result', ['Step', 'Tool', 'What it produced'], [
      ['1. Dataset download and preparation', 'Python scripts (download_dataset.py, prepare_dataset.py)', '600 balanced 10 s clips from 4 pump units, split 80/20, plus 10 demo clips never uploaded'],
      ['2. Laptop-microphone recording and noise augmentation', 'Python with sounddevice (record_mic_path.py, augment_noise.py)', 'Clips re-recorded through the speaker-to-mic path, room background, and 200 noise-augmented clips'],
      ['3. Upload', ['Edge Impulse ingestion API', cite(5)], '908 labelled clips in the Edge Impulse project'],
      ['4. Impulse design', 'Edge Impulse Studio API (build_impulse.py): MFE block + Keras learning block', '1 s windows → 2,440 MFE features per window'],
      ['5. Training', 'Edge Impulse cloud GPU, expert-mode Keras script (full shuffle, time budget)', 'int8 1D CNN, 5,043 parameters'],
      ['6. On-device estimates', 'Edge Impulse profiling, Arduino Nano 33 BLE Sense', `${fmt(perf('nn_ms'), 0, ' ms')} per window for the int8 classifier`],
      ['7. Export', 'Edge Impulse deployment (EON compiler)', 'C++ library and WebAssembly package'],
      ['8. C++ build and test-set evaluation', ['WSL Ubuntu, example-standalone-inferencing-linux', cite(6), ', eval.sh'], `${pct(testAcc())} of 2,679 held-out test windows correct`],
      ['9. Live dashboard', 'WebAssembly model + JavaScript (dsp.js, charts.js, app.js)', 'Monitor with alarm logic, occlusion analysis, noise stress test, model card'],
      ['10. Automation', 'ei_pipeline.py, GitHub Actions, Edge Impulse project pipeline', 'Scheduled upload → retrain → build'],
      ['11. Documentation', 'docx (Node.js), Word, Slides', 'This report, presentation slides, demo script'],
    ], [26, 37, 37], 'tab:workflow'),
    section('Organisation of the report'),
    p('Chapter 2 describes the dataset and the data pipeline, Chapter 3 the impulse and its training, Chapter 4 the evaluation on held-out recordings, and Chapter 5 the deployment as C++ and WebAssembly together with the automation. Chapter 6 concludes and lists the limitations. Appendix A lists the abbreviations used.'),

    // ===================================================================== 2
    chapter(2, 'Dataset and Data Pipeline', 'ch:data'),
    section('Pump recordings (DCASE 2020 Task 2 / MIMII)'),
    p(['The machine sounds come from the development set of the DCASE 2020 Challenge Task 2, “Unsupervised detection of anomalous sounds for machine condition monitoring”', cite(2), ', which is built from Hitachi’s MIMII dataset', cite(3), '. Each clip is a 10-second, 16 kHz, single-channel recording of a real water pump in a factory, with real factory background noise mixed in. Anomalous clips were recorded after deliberately introducing faults such as leakage, contamination and clogging. Four different pump units are included (machine IDs 00, 02, 04, 06). The pump subset contains 3,749 normal and 456 anomalous clips (1.0 GB) and is licensed CC BY-NC-SA 4.0', cite(4), '.']),
    ...fig('spectrograms.png', 'Log-mel spectrograms of a normal and an abnormal clip from the same pump: the fault adds broadband transients and changes the tonal pattern', 'fig:spec'),
    section('Subset and split'),
    p(`prepare_dataset.py draws a balanced subset of 300 normal and 300 abnormal clips, spread evenly over the four pump units, and splits it 80/20 into training and testing, stratified by class and unit. Ten further clips (five per class) are held back entirely for the live demonstration and are never uploaded. Every file is checked for format (mono, 16 kHz, 16-bit PCM, 10 s) and renamed <label>.<machine>-id<ID>-<origin>-<index>.wav so the label travels with the file. ${ref('tab:sets')} lists the resulting sets.`),
    ...table('Number of 10-second clips per set and class', ['Set', 'normal', 'abnormal', 'background', 'Source'], [
      ['Training (dataset)', dcase('training/normal'), dcase('training/abnormal'), '—', 'DCASE 2020 pump clips'],
      ['Testing (dataset)', dcase('testing/normal'), dcase('testing/abnormal'), '—', 'DCASE 2020 pump clips'],
      ['Training (laptop mic)', mic('training/normal'), mic('training/abnormal'), mic('training/background'), 'Replayed clips + room recordings'],
      ['Testing (laptop mic)', mic('testing/normal'), mic('testing/abnormal'), mic('testing/background'), 'Replayed clips + room recordings'],
      ['Noise-augmented (training)', '', '', '', `${D('dataset.augmented') || 0} clips mixed with room noise`],
      ['Live-demo hold-out', dcase('demo/normal'), dcase('demo/abnormal'), '—', 'never uploaded'],
    ], [24, 11, 11, 13, 41], 'tab:sets'),
    ...shot('02-data-acquisition.png', 'The dataset in Edge Impulse (Data acquisition)', 'fig:acq', 'Data acquisition page with the train/test split'),
    section('Laptop-microphone recordings', 'sec:mic'),
    p('The model must eventually listen through a real microphone. A model trained only on the clean dataset plus a background class recorded with the laptop microphone can learn a shortcut: “anything that sounds like the laptop microphone is background”. To remove that shortcut we re-recorded 60 pump clips through the laptop’s own speaker and microphone (record_mic_path.py), so normal and abnormal examples exist with the same acoustic colouring as the background, and recorded about eight minutes of the room (silence, speech, typing, footsteps) as the background class. Each re-recording is aligned to its source by cross-correlation; the median correlation with the original clip was 0.47 (range 0.26–0.68).'),
    ...fig('mic_path.png', 'Average spectrum of a clip before and after the laptop speaker-to-microphone path: the low frequencies (below about 300 Hz) lose 15–20 dB', 'fig:micpath'),
    p([{ text: 'Practical finding. ', bold: true }, 'Windows applies voice-oriented “audio enhancements” (echo cancellation and noise suppression) to laptop microphone arrays. On this laptop they erased the speaker sound almost completely (the level fell from −94 to −103 dBFS while the speaker played noise), whereas the raw kernel-streaming device heard it clearly (+20 dB). A machine-monitoring product must use a raw microphone path, as a dedicated MEMS microphone on a microcontroller does; for the demonstration the enhancements were switched off.']),
    section('Noise augmentation'),
    p(['augment_noise.py mixes the recorded room noise into dataset training clips at a random signal-to-noise ratio between 5 and 20 dB (the same idea as Edge Impulse’s mix-noise transformation block', cite(6), '), so the classifier keeps working when people talk or a fan runs nearby.']),
    section('Data pipeline'),
    ...numbered([
      'download_dataset.py downloads the pump archive from Zenodo (with resume and size check) and extracts it into its own folder.',
      'prepare_dataset.py selects, validates, splits and renames clips and writes manifest.csv (the transformation step).',
      'upload_to_ei.py sends the files to the Edge Impulse ingestion API in batches, always with an explicit x-label header, then reads the sample counts back from the project to verify them against the manifest. It can resume after a network failure (--skip-existing).',
      'record_mic_path.py and augment_noise.py create the laptop-microphone and noise-augmented sets, uploaded the same way.',
    ]),
    p('All uploads were verified: the project holds exactly the clips listed in the manifest, and no source clip appears in both the training and the testing set.'),

    // ===================================================================== 3
    chapter(3, 'Impulse Design and Training', 'ch:impulse'),
    p(['The impulse was created and configured through the Edge Impulse Studio API by build_impulse.py, following Edge Impulse’s official Python API example', cite(6), ', and can equally be inspected and edited in Studio.']),
    ...shot('04-impulse-design.png', 'The impulse: audio input, MFE processing block and classifier', 'fig:impulse', 'Impulse design → Create impulse'),
    section('Input block'),
    p('Audio at 16 kHz is cut into 1,000 ms windows every 500 ms (zero-padding enabled). One-second windows give the network several rotations of the pump and keep the raw input at 16,000 samples (32 KB as int16), small enough for a microcontroller. A 10-second clip yields 19 overlapping windows.'),
    section('MFE processing block'),
    p(`Mel-filterbank energy (MFE) features suit non-voice audio: they keep the spectral envelope on a perceptual frequency scale without the speech-oriented decorrelation of MFCCs. Each window becomes 61 frames × 40 mel bands = 2,440 features (${ref('tab:mfe')}).`),
    ...table('MFE parameters', ['Parameter', 'Value', 'Why'], [
      ['Frame length / stride', '32 ms / 16 ms', '512-sample FFT frames with 50 % overlap'],
      ['Filters', '40', 'Enough resolution for machine harmonics'],
      ['FFT length', '512', 'Matches the frame length'],
      ['Frequency range', '0–8000 Hz', 'Full band at 16 kHz'],
      ['Noise floor', '−72 dB', 'Keeps quiet machine detail; clips silence'],
    ], [29, 24, 47], 'tab:mfe'),
    ...shot('05-mfe-parameters.png', 'MFE parameters and the processed spectrogram in Edge Impulse', 'fig:mfeparams'),
    ...shot('06-mfe-features.png', 'MFE feature explorer in Edge Impulse: the 14,573 training windows projected to two dimensions', 'fig:mfefeat'),
    p(`The feature explorer (${ref('fig:mfefeat')}) already shows what the classifier will find easy and hard: background windows form a cluster of their own, while normal and abnormal windows share several clusters. The test results in Chapter 4 follow the same pattern: background is never confused with a pump, and the errors are between normal and abnormal.`),
    section('Versions of the impulse', 'sec:versions'),
    p(`Like the guide, the first version (v1) used two processing blocks: MFE for the classifier and a Spectral Analysis block (128-point FFT plus statistics) for a Gaussian-mixture (GMM) anomaly detector. That design trained well but its exported library failed at run time (${ref('sec:ondevice')}). The final version (v2) is a single MFE block feeding the classifier, rebuilt from scratch in a second Edge Impulse project by scripts/run_all.py. Both versions use the same data, MFE parameters and network; they differ in the anomaly block and in how long the classifier could be trained (${ref('sec:training')}).`),
    section('Learning block'),
    ...table('Classifier architecture and training settings', ['Layer', 'Configuration'], [
      ['Input', '2,440 MFE features reshaped to 61 frames × 40 mel bands'],
      ['1D convolution + max-pool', '16 filters, kernel 3'],
      ['Dropout', '0.25'],
      ['1D convolution + max-pool', '32 filters, kernel 3'],
      ['Dropout', '0.25'],
      ['Flatten → Dense (softmax)', 'one output per class'],
      ['Training (final, v2)', 'Adam, learning rate 0.003, batch 256, 240 s training budget (58 epochs), epoch with the lowest validation loss kept; 20 % validation, class weights, full shuffle each epoch'],
      ['Training (v1)', 'Adam, learning rate 0.001, batch 32, 100 epochs; otherwise the same'],
    ], [35, 65], 'tab:arch'),
    section('Training results', 'sec:training'),
    p([{ text: 'Debugging lesson. ', bold: true }, `The first training runs stayed at exactly 50 % accuracy with a loss of 0.693 (= ln 2) for all 100 epochs, and every window was predicted as the same class. The features were not the problem: a simple logistic regression on per-band MFE statistics, computed from the features downloaded through the API, separated the classes with 78 % accuracy on held-out clips. The cause was data order. Edge Impulse delivers the training windows grouped by class, and the default training script shuffles with a buffer of only 4 × batch size (128 windows), so almost every batch contained a single class and the network only chased the label of the current batch. Switching the learning block to expert mode and shuffling the complete training set every epoch, one changed line, fixed it. A second run then showed that Studio’s default learning rate of 0.005 was too high for this data: validation accuracy swung between about 57 % and 80 % and collapsed to one class after 75 epochs. Lowering it to 0.001 gave a stable model (v1 in ${ref('fig:story')}).`]),
    p([{ text: 'Compute lesson. ', bold: true }, `The free plan meters training compute per project per month. Our experiments used up the first project’s allowance, and the second project had 25 minutes for everything (feature generation, training, profiling). The v1 schedule (20 s per epoch × 100 epochs) would have needed about 35 minutes. Most of each 54 ms training step was fixed overhead rather than arithmetic (the network needs only about 170,000 multiply-accumulates per window), so we raised the batch size to 256 (46 steps per epoch, 4 s per epoch) and the learning rate to 0.003, and added a callback to the expert-mode script that stops training after 240 s and restores the epoch with the lowest validation loss. The final run stopped after 58 epochs. Its curve follows v1’s closely up to that point (${ref('fig:story')}), so the remaining gap to v1 is training time, not method.`]),
    ...fig('shuffle_fix.png', 'Validation accuracy per epoch: default shuffle (stuck at chance), full shuffle with learning rate 0.005 (unstable), v1 (learning rate 0.001, 100 epochs) and the final time-budgeted run', 'fig:story'),
    ...table('Validation results of the classifier (validation windows come from the same clips as training windows, so the test results of Chapter 4 are the honest measure)', ['Metric', 'float32', 'int8 (deployed)'], [
      ['Validation accuracy (final, v2)', pct(valAcc32), pct(valAcc8)],
      ['Validation loss (final, v2)', fmt(train('float32', 'loss'), 3), fmt(train('int8', 'loss'), 3)],
      ['Validation accuracy (v1, 100 epochs)', pct(trainV1('float32')), pct(trainV1('int8'))],
    ], [44, 28, 28], 'tab:val'),
    ...fig('training_curves.png', 'Final model: training and validation accuracy and loss per epoch', 'fig:curves'),
    ...shot('08-classifier.png', 'Classifier page in Edge Impulse: accuracy, confusion matrix and on-device performance', 'fig:classifier', 'Classifier page after training'),

    // ===================================================================== 4
    chapter(4, 'Testing and Evaluation', 'ch:test'),
    section('Model testing on held-out recordings', 'sec:testing'),
    p(`Edge Impulse’s Model testing runs as a cloud job, and the final project had no compute left after training. The final model was therefore tested locally with the deployed artefact itself, the compiled int8 C++ library: deployment/cpp-linux/eval.sh builds the standalone Linux application with a small batch front end (batch_eval.cpp) that slides the 1 s window with the 0.5 s stride over every test clip, as the impulse does, and scripts/eval_cpp_results.py scores the windows. The test set is the ${ev('v2', 'overall.clips') || 141} clips (${ev('v2', 'overall.windows') || 2679} windows) uploaded as the Edge Impulse test set; none of them was used in training. The v1 library was scored the same way for comparison (${ref('tab:test')}).`),
    ...table('Test results on held-out clips (int8, C++ library)', ['Test metric', 'Final (v2)', 'v1'], [
      ['Window accuracy (top class)', pct(ev('v2', 'overall.window_accuracy')), pct(ev('v1', 'overall.window_accuracy'))],
      ['Windows with top score ≥ 0.6 / accuracy on them', `${pct(ev('v2', 'overall.confident_share'))} / ${pct(ev('v2', 'overall.confident_accuracy'))}`, `${pct(ev('v1', 'overall.confident_share'))} / ${pct(ev('v1', 'overall.confident_accuracy'))}`],
      ['Clip accuracy (majority vote over 19 windows)', pct(ev('v2', 'overall.clip_accuracy')), pct(ev('v1', 'overall.clip_accuracy'))],
      ['Dataset clips / laptop-microphone clips', `${pct(ev('v2', 'dcase.clip_accuracy'))} / ${pct(ev('v2', 'mic.clip_accuracy'))}`, `${pct(ev('v1', 'dcase.clip_accuracy'))} / ${pct(ev('v1', 'mic.clip_accuracy'))}`],
      ['Recall: abnormal / normal / background', ['abnormal', 'normal', 'background'].map((l) => pct(ev('v2', `per_class.${l}.recall`))).join(' / '), ['abnormal', 'normal', 'background'].map((l) => pct(ev('v1', `per_class.${l}.recall`))).join(' / ')],
    ], [46, 27, 27], 'tab:test'),
    ...(() => {
      const cm = ev('v2', 'confusion');
      if (!cm) return [];
      const labels = Object.keys(cm);
      const rows = labels.map((t) => {
        const total = labels.reduce((s, q) => s + cm[t][q], 0) || 1;
        return [t, ...labels.map((q) => `${cm[t][q]} (${Math.round((cm[t][q] / total) * 100)} %)`)];
      });
      return table('Test confusion matrix of the final model (1 s windows; rows: true class, columns: predicted)', ['actual \\ predicted', ...labels], rows, labels.map(() => 1).concat(1), 'tab:cm');
    })(),
    p(`Background is never confused with a pump (${ref('tab:cm')}). The errors are between normal and abnormal pumps and depend strongly on the pump unit: all abnormal clips of unit 02 are recognised, but only 7 of its 17 normal clips, while unit 04’s normal clips are all correct. Per-unit differences of this size are typical of the DCASE pump data, whose four units sound quite different from each other.`),
    p(`Two cross-checks support the local evaluation. Edge Impulse’s own Model testing of v1 gave ${pct(v1Test)} (it counts windows below the 0.6 confidence threshold as errors) and ${pct(D('test_clip_level_v1.clip_accuracy'))} of clips; the C++ evaluation of the same model gives ${pct(ev('v1', 'overall.clip_accuracy'))} of clips and ${pct(ev('v1', 'overall.confident_accuracy'))} accuracy on its confident windows.`),
    ...shot('10-model-testing.png', 'Model testing of v1 in Edge Impulse', 'fig:testing', 'Model testing → Classify all (first project)'),
    section('Comparison with related work'),
    ...table('Comparison with related work (the public project tests whole clips of two classes; the DCASE baseline solves a harder, unsupervised task)', ['System', 'Task', 'Result'], [
      ['This project, final (int8, MFE + 1D CNN)', 'normal / abnormal / background, 1 s windows', `${pct(testA)} of test windows, ${pct(ev('v2', 'overall.clip_accuracy'))} of clips`],
      ['This project, v1 (100 epochs)', 'same', `${pct(ev('v1', 'overall.window_accuracy'))} of test windows, ${pct(ev('v1', 'overall.clip_accuracy'))} of clips`],
      [['Public Edge Impulse pump project (Manivannan)', cite(7)], 'normal / abnormal, MFE + NN', '93.5 % test accuracy (97.8 % validation)'],
      [['DCASE 2020 baseline (autoencoder)', cite(8)], 'unsupervised, normal-only training', '72.9 % AUC on pump'],
    ], [38, 32, 30], 'tab:related'),
    section('Anomaly detection', 'sec:anomaly'),
    p(`A classifier can only answer with one of the classes it was trained on. The guide’s answer to sounds never seen in training is an anomaly block, and v1 had one: a Gaussian mixture model (5 components) on the 69 Spectral Analysis features, trained and tested in Edge Impulse. A later experiment in the same project tried K-means clustering (32 clusters) on the MFE features instead (${ref('fig:anom')}). The exported library of the GMM version, however, could not run unmodified (${ref('sec:ondevice')}), so its behaviour on unknown sounds in deployment could not be shown reliably.`),
    ...shot('07-anomaly-explorer.png', 'Anomaly detection page of the first Edge Impulse project: K-means settings (32 clusters) and the anomaly explorer on two MFE features', 'fig:anom'),
    p('For the final version we fitted the GMM to a slice of the MFE features instead (all 40 mel bands of two frames, 80 values). The mixture trained, but Edge Impulse’s profiling step then failed with “cannot reshape array of size 35,558,120 into shape (14573, 61)”: it reshaped the anomaly block’s input to the classifier’s 61 × 40 input shape. This looks like a platform limitation for a GMM that shares a processing block with an expert-mode classifier, and the failed job used a quarter of the remaining compute, so the final model is classifier-only.'),
    p('The consequence is visible in the dashboard. Three synthetic sounds that are neither pump nor room (white noise, a two-tone hum and a frequency sweep) are all labelled abnormal. A confidence threshold does not rescue this: white noise is classified with 100 % confidence, and only the hum falls below 0.6. Detecting unknown sounds therefore remains future work: re-adding the GMM or K-means block once compute resets, or an unsupervised model trained on normal sound only, as in DCASE.'),
    section('EON Tuner'),
    p('The EON Tuner searches combinations of processing parameters and network architectures under the target’s latency and memory constraints. It is configured for the Arduino Nano 33 BLE Sense with a latency budget below the 500 ms window stride (build_impulse.py tuner starts it through the API).'),
    ...(tuner().rows ? table('Best EON Tuner configurations', ['Configuration', 'Accuracy', 'Latency', 'RAM', 'Flash'], tuner().rows, [36, 16, 16, 16, 16], 'tab:tuner')
      : [p(`Each EON Tuner candidate is a full training job, so a search costs far more compute than the free plan left this month; the run is prepared but was not executed. The manual design already meets the real-time constraint with a wide margin (${ref('sec:ondevice')}), which is the main thing the tuner would have checked.`)]),

    // ===================================================================== 5
    chapter(5, 'Deployment and Automation', 'ch:deploy'),
    section('On-device estimates and real-time feasibility', 'sec:ondevice'),
    ...table('Estimated on-device performance per block on the Arduino Nano 33 BLE Sense, Cortex-M4F 64 MHz (Edge Impulse profiling, as shown on the public project page)', ['Block', 'Latency', 'RAM', 'Flash'], [
      ['MFE processing block', perf('mfe_ms') ? fmt(perf('mfe_ms'), 0, ' ms') : '—', perf('mfe_ram_kb') ? fmt(perf('mfe_ram_kb'), 1, ' KB') : '—', '—'],
      ['Classifier, int8 + EON compiler (deployed)', fmt(perf('classifier_int8.latency_ms'), 0, ' ms'), fmt(mem('classifier_int8.ram_kb'), 1, ' KB'), fmt(mem('classifier_int8.flash_kb'), 1, ' KB')],
      ['Classifier, float32 (for comparison)', fmt(perf('classifier_float32.latency_ms'), 0, ' ms'), fmt(mem('classifier_float32.ram_kb'), 1, ' KB'), fmt(mem('classifier_float32.flash_kb'), 1, ' KB')],
      ['v1 only: GMM anomaly detector', fmt(D('performance_v1.gmm_ms'), 0, ' ms'), fmt(D('performance_v1.anomaly_float32.ram_kb'), 1, ' KB'), fmt(D('performance_v1.anomaly_float32.flash_kb'), 1, ' KB')],
    ], [49, 17, 17, 17], 'tab:ondevice'),
    p(['For continuous monitoring a new window starts every 500 ms, so feature extraction and the classifier must finish within 500 ms. The int8 classifier takes ', fmt(perf('nn_ms'), 0, ' ms'), ', leaving almost the whole budget to the MFE block. Quantising the classifier to int8 cut its latency from ', fmt(perf('classifier_float32.latency_ms'), 0, ' ms'), ' to ', fmt(perf('nn_ms'), 0, ' ms'), ` with a validation accuracy within two points of float32 (${ref('tab:val')}). For comparison, the public reference pump model`, cite(7), ' needs 555 ms per inference on a Cortex-M4F at 80 MHz and would fall behind a 500 ms stride.']),
    ...shot('01-dashboard.png', 'Edge Impulse project dashboard: target device Arduino Nano 33 BLE Sense (top right) and public sharing', 'fig:target'),
    ...shot('12-deployment.png', 'Deployment page: C++ library with the EON compiler', 'fig:deployment', 'Deployment page'),
    p([{ text: 'Deployment lesson. ', bold: true }, `(1) With two processing blocks (MFE for the classifier, Spectral Analysis for the anomaly detector), the v1 C++ and WebAssembly builds both failed at run time with “Would write outside feature buffer”: the Edge Impulse SDK writes every processing block’s output, one after another, into one buffer sized by the classifier’s input (2,440 values), so the 69 spectral values overflowed it. Enlarging that buffer by hand in the exported model_variables.h (nn_input_frame_size = 2,509) made the v1 C++ library run, which is how v1 is evaluated in ${ref('sec:testing')}. (2) Feeding both blocks to the classifier and cropping back to the MFE part inside the network fixed the buffer without patching, but the int8 model collapsed from 91 % to 6 % validation accuracy: an int8 model quantises its whole input with one scale, and the spectral statistics (values up to about 400) left the MFE values (0–0.8) with almost no resolution. The robust design is one processing block per impulse, as in the final version, whose libraries run unmodified.`]),
    section('C++ library on Linux (as in the reference guide)', 'sec:cpp'),
    p(['The final C++ library (EON compiler, int8) was built with Edge Impulse’s example-standalone-inferencing-linux project', cite(6), ' inside WSL2 Ubuntu on the laptop, exactly as in the guide (deployment/cpp-linux/build.sh), and run on one window of raw features from a held-out clip of each class:']),
    ...code(['$ APP_CUSTOM=1 make -j$(nproc)', '$ for f in features-{abnormal,normal,background}.txt; do ./build/custom $f; done', ...((cpp().output_lines) || ['(output inserted after the build)'])]),
    p(`On the laptop CPU one window takes about ${fmt(ev('v2', 'mean_dsp_ms'), 1, ' ms')} of feature extraction and ${fmt(ev('v2', 'mean_classification_ms'), 1, ' ms')} of inference. The same library, with the batch front end of ${ref('sec:testing')}, scored all ${ev('v2', 'overall.windows') || 2679} test windows.`),
    section('WebAssembly live dashboard', 'sec:dashboard'),
    p('The same impulse, built as WebAssembly, runs inside a browser page served from the laptop (demo/), with no network connection needed. The page is organised like the screen of a condition-monitoring instrument, in four tabs. All signal processing beyond the model itself (FFT, MFE features for display, band removal, noise mixing) is implemented in plain JavaScript in demo/dsp.js.'),
    subsection('Monitor'),
    p(`The input is a held-out clip played through the speakers, the live microphone, or any audio file dropped onto the page. Every 250 ms the latest 1 s window is classified; the current-window verdict averages the last three inferences and reports “uncertain” below 0.6 confidence. On top of the classifier sits the logic a real monitor needs: a machine-state indicator with hysteresis (ALARM when at least 6 of the last 8 windows are abnormal, cleared only when at most 2 are; NO MACHINE when background dominates), so one odd window never raises an alarm. State changes are written to an event log that can be exported as CSV, and a counter shows the edge-computing argument directly: the device hears 32 KB of audio per second but would send only a 16-byte message per state change. The panel “What the network sees” draws the 61 × 40 MFE features of the current window (${ref('fig:dashboard')}).`),
    ...fig('dashboard.png', 'Monitor tab: a held-out abnormal clip drives the machine state to ALARM; probabilities over time, decision strip, MFE features of the current window and event log', 'fig:dashboard', 520),
    subsection('Analyze a clip: which frequencies matter'),
    p(`For a selected clip the page classifies every 0.5 s window, decides the clip by majority vote, and runs an occlusion test: each of eight frequency bands is deleted from every window (FFT, zero the band, inverse FFT) and the model is run again, 152 extra inferences for a 10 s clip. The drop in the predicted class’s probability shows which bands the model relies on. For abnormal clip 1 of pump 00, removing 300–600 Hz lowers p(abnormal) by 0.33 on average and removing 1–1.5 kHz by 0.09, while the bands above 4 kHz have no effect (${ref('fig:analyze')}). The model has therefore learned the low-frequency signature of the fault rather than broadband noise, which also explains why playback through laptop speakers, which lose 15–20 dB below 300 Hz (${ref('fig:micpath')}), makes live classification harder.`),
    ...fig('dashboard_analyze.png', 'Analyze tab: majority vote over 19 windows, probability per window, MFE features of the whole clip and the frequency-band occlusion test', 'fig:analyze', 520),
    subsection('Noise stress test'),
    p(`The training set was augmented with room noise at 5–20 dB SNR. The stress test checks what that bought: the ten held-out demonstration clips are mixed with white noise or with a recording of the room at seven levels from clean to −5 dB SNR and classified in the browser (1,330 inferences per run, ${ref('tab:stress')}).`),
    ...table('Noise stress test in the browser (10 held-out clips, 0.5 s stride; window accuracy / normal recall; abnormal recall is 100 % at every level)', ['SNR', 'White noise', 'Room recording'], [
      ['clean', '77.4 % / 55 %', '77.4 % / 55 %'],
      ['30 dB', '82.1 % / 64 %', '75.8 % / 52 %'],
      ['20 dB', '82.6 % / 65 %', '73.2 % / 46 %'],
      ['10 dB', '75.8 % / 52 %', '72.1 % / 44 %'],
      ['5 dB', '73.2 % / 46 %', '72.1 % / 44 %'],
      ['0 dB', '72.6 % / 45 %', '70.5 % / 41 %'],
      ['−5 dB', '60.0 % / 20 %', '68.9 % / 38 %'],
    ], [20, 40, 40], 'tab:stress'),
    p('Accuracy stays within five points of the clean value down to 0 dB of white noise and degrades slowly with room noise; light white noise even raises it, consistent with the noise-augmented training data. The errors always go one way: noise makes normal pumps sound abnormal, while every abnormal window is still recognised. For a maintenance monitor this is the safe failure mode, more false alarms rather than missed faults, and the hysteresis of the machine-state logic absorbs isolated false alarms.'),
    ...fig('dashboard_stress.png', 'Noise stress test tab: accuracy as white noise increases; the shaded band is the SNR range used for augmentation in training', 'fig:stress', 520),
    subsection('Model card and in-browser benchmark'),
    p(`The last tab summarises the deployed model: the signal path with the real tensor shapes (16,000 samples → 61 × 40 MFE → Conv1D 61 × 16 → 31 × 16 → Conv1D 31 × 32 → 16 × 32 → 3 outputs, 5,043 parameters), the C++ test results and confusion matrix of ${ref('sec:testing')}, the on-device estimates, and a benchmark that times 300 inferences of the WebAssembly model on the viewing machine and re-classifies the ten demonstration clips, which were never uploaded to Edge Impulse: ${pct(D('demo_eval.windowAccuracy'))} of windows and ${pct(D('demo_eval.clipAccuracy'))} of clips are correct, with the same normal-to-abnormal confusion as in ${ref('sec:testing')}.`),


    section('Automation'),
    p('The guide automates two pipelines; both are reproduced here:'),
    ...bullets([
      [{ text: 'Data import and retraining: ', bold: true }, 'ei_pipeline.py uploads any new labelled recordings, starts a retrain job (which regenerates the DSP features and retrains every learning block), builds the C++ library and the WebAssembly package, and downloads both. A GitHub Actions workflow runs it every two days (cron 0 3 */2 * *) and stores the libraries as build artifacts.'],
      [{ text: 'Project pipeline: ', bold: true }, 'Edge Impulse’s native Data sources pipeline (“Don’t import data”) retrains the model, creates a new project version and builds a new C++ deployment on a schedule — the same configuration as the guide’s Figures 32–33.'],
    ]),
    ...shot('13-versioning.png', 'Project versions in the first Edge Impulse project, created automatically by the project pipeline (retrain → create version)', 'fig:versions'),
    ...shot('14-pipeline.png', 'Native Edge Impulse pipeline: retrain, create version, build deployment', 'fig:pipeline', 'Data acquisition → Data sources'),
    section('Public project and reproducibility'),
    p(`The Edge Impulse project is public (${eiLink}): anyone can inspect the 908 samples, the impulse, the trained model and its on-device performance, and clone it (${ref('fig:public')}). The complete code is public on GitHub (${ghLink}), and its README reproduces every step, from the dataset download to the dashboard.`, { alignment: AlignmentType.LEFT }),
    ...shot('16-launch-in-browser.png', 'Public view of Edge Impulse project 1165553: 908 samples, validation accuracy 83.6 % (float32) and on-device performance of the int8 model (7 ms, 7.9 KB RAM, 34.6 KB flash)', 'fig:public'),

    // ===================================================================== 6
    chapter(6, 'Conclusions', 'ch:conclusions'),
    section('Conclusions'),
    p('Following the reference guide end to end on a free plan and without IoT hardware, we built, evaluated and deployed an Edge AI pump-sound monitor: a 1D CNN on MFE features that distinguishes normal, abnormal and background audio, a C++ build running on Linux, a live WebAssembly dashboard, and an automated retrain-and-deploy loop. Every number in this report comes from scripts in the public repository, and the test results were measured with the deployed library itself. The dashboard’s own analyses add evidence beyond accuracy: the occlusion test shows the model relies on the 300–600 Hz fault signature, and the noise stress test shows it degrades gracefully and fails safe, turning noisy normal pumps into false alarms rather than missing faults.'),
    p('Along the way the project produced lessons that the clean dataset and the documentation hide: the model must be trained on the acoustic path it will hear in deployment; consumer audio processing can silently erase the very sounds a monitor needs; a training script that sees class-ordered data must shuffle all of it; and each extra processing block or anomaly block must be checked end to end in the exported library, not only in Studio.'),
    section('Limitations and future work'),
    ...bullets([
      'No physical microcontroller was used; on-device numbers are Edge Impulse estimates and the laptop stands in for the device. The next step is flashing the same library to an Arduino Nano 33 BLE Sense and listening with its on-board microphone.',
      'The free Edge Impulse plan allows little compute per project per month. The final model’s training had to stop after 240 s; training it for as long as v1 when compute resets should recover v1’s accuracy.',
      'The final model has no anomaly block, so a sound it has never heard is forced into a known class. Re-adding a GMM or K-means block, or an unsupervised model trained on normal sound only (as in DCASE), is the most important next step for a real deployment.',
      'Normal clips of some pump units are often called abnormal. Per-unit models or per-unit normalisation of the features, as used by most DCASE systems, would address this; a stricter test holds out a whole unit (prepare_dataset.py --holdout-id 06).',
      'Laptop speakers cannot reproduce the pump’s low frequencies, so the live demonstration hears a band-limited version of the machine.',
      'More machine types (fans, valves, slide rails) and Edge Impulse’s performance calibration for continuous audio would make the system broader and quieter in production.',
    ]),

    // ===================================================================== references
    appendix('', 'Bibliography and References', 'refs'),
    ...numbered([
      'Turasov I, “Industrial Productivity Solutions Guide: Adding a sixth sense to your industrial machines”, Edge Impulse Inc., solutions guide (course reference document).',
      'Koizumi Y, Kawaguchi Y, Imoto K, Nakamura T, Nikaido Y, Tanabe R, Purohit H, Suefusa K, Endo T, Yasuda M and Harada N, “Description and discussion on DCASE2020 Challenge Task 2: Unsupervised anomalous sound detection for machine condition monitoring”, Proc. Detection and Classification of Acoustic Scenes and Events (DCASE) Workshop, 2020.',
      'Purohit H, Tanabe R, Ichige K, Endo T, Nikaido Y, Suefusa K and Kawaguchi Y, “MIMII Dataset: Sound dataset for malfunctioning industrial machine investigation and inspection”, Proc. Detection and Classification of Acoustic Scenes and Events (DCASE) Workshop, 2019.',
      'Koizumi Y et al., “DCASE 2020 Challenge Task 2 development dataset”, Zenodo, 2020, https://zenodo.org/records/3678171',
      'Edge Impulse, “Edge Impulse documentation: ingestion API, processing blocks (MFE, Spectral Analysis), anomaly detection (GMM), EON Tuner, deployment”, https://docs.edgeimpulse.com',
      'Edge Impulse, “example-standalone-inferencing-linux, notebooks (Python API bindings example), example-transform-mix-audio”, https://github.com/edgeimpulse',
      'Manivannan, “Industry 4.0 – Predictive Maintenance”, Hackster.io, and public Edge Impulse project 88799, https://www.hackster.io/manivannan/industry-4-0-predictive-maintenance-3bb415',
      'Kawaguchi Y, “DCASE 2020 Task 2 baseline system”, https://github.com/y-kawagu/dcase2020_task2_baseline',
    ], 'refs'),

    // ===================================================================== appendices
    appendix('A', 'List of Abbreviations', 'app:abbr'),
    ...table('Abbreviations used in this report', ['Abbreviation', 'Meaning'], [
      ['API', 'Application programming interface'],
      ['AUC', 'Area under the ROC curve'],
      ['CNN', 'Convolutional neural network'],
      ['DCASE', 'Detection and Classification of Acoustic Scenes and Events (challenge and workshop)'],
      ['DSP', 'Digital signal processing'],
      ['EON', 'Edge Optimized Neural (Edge Impulse compiler)'],
      ['FFT', 'Fast Fourier transform'],
      ['GMM', 'Gaussian mixture model'],
      ['int8 / float32', '8-bit integer (quantised) / 32-bit floating-point model'],
      ['MFCC', 'Mel-frequency cepstral coefficients'],
      ['MFE', 'Mel-filterbank energy'],
      ['MIMII', 'Malfunctioning Industrial Machine Investigation and Inspection (dataset)'],
      ['RAM', 'Random-access memory'],
      ['SDK', 'Software development kit'],
      ['SNR', 'Signal-to-noise ratio'],
      ['WASM', 'WebAssembly'],
      ['WSL', 'Windows Subsystem for Linux'],
    ], [25, 75], 'tab:abbr'),
  ];
}

function build() {
  const team = TEAM.members || [{ name: '[Member 1 name]', id: '[Enrolment No.]' }, { name: '[Member 2 name]', id: '[Enrolment No.]' },
    { name: '[Member 3 name]', id: '[Enrolment No.]' }];
  const info = {
    degree: TEAM.degree || 'Master of Technology',
    specialisation: TEAM.specialisation || 'in Data Science and Machine Learning',
    specialisationShort: TEAM.specialisation_short || 'Data Science and Machine Learning',
    professor: TEAM.professor || '[Name of supervisor]',
    designation: TEAM.professor_designation || '[Designation]',
    school: TEAM.school || 'School of Information Technology, Artificial Intelligence and Cyber Security',
    schoolShort: TEAM.school_short || 'SITAICS',
    date: TEAM.date || 'November 2026',
  };

  // two passes: the first assigns chapter, section, figure and table numbers, the second resolves ref()
  R.resetPass();
  body(team);
  R.resetPass();
  const content = body(team);

  const footer = () => new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] });
  const page = (margins = R.MARGIN) => ({ size: { width: R.PAGE.width, height: R.PAGE.height }, margin: { ...margins, header: 708, footer: 708 } });

  const doc = new Document({
    creator: 'Edge AI project team',
    title: TITLE,
    styles: {
      default: {
        document: { run: { font: FONT, size: 24 }, paragraph: { spacing: { line: LINE } } },
        heading1: { run: { font: FONT, size: 32, bold: true },
          paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 360 }, keepNext: true, outlineLevel: 0 } },
        heading2: { run: { font: FONT, size: 28, bold: true },
          paragraph: { spacing: { before: 300, after: 120 }, keepNext: true, outlineLevel: 1 } },
        heading3: { run: { font: FONT, size: 24, bold: true },
          paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 2 } },
      },
      paragraphStyles: [
        { id: 'FrontTitle', name: 'FrontTitle', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: FONT, size: 28, bold: true },
          paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 360 }, keepNext: true } },
        { id: 'FigureCaption', name: 'FigureCaption', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: FONT, size: 22 },
          paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240, line: 276 } } },
        { id: 'TableCaption', name: 'TableCaption', basedOn: 'Normal', next: 'Normal', quickFormat: true,
          run: { font: FONT, size: 22 },
          paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100, line: 276 }, keepNext: true } },
      ],
    },
    numbering: {
      config: [
        { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
        { reference: 'steps', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
        { reference: 'refs', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 540, hanging: 540 } } } }] },
      ],
    },
    sections: [
      // front page and cover page (no page numbers), as in the template
      {
        properties: { page: page({ left: 1440, right: 1440, top: 1000, bottom: 800 }) },
        children: [...titlePage(team, info), pageBreak(), ...titlePage(team, info)],
      },
      // preliminary pages, lower-case roman numbers
      {
        properties: { type: SectionType.NEXT_PAGE, page: { ...page(), pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } },
        footers: { default: footer() },
        children: [
          ...declaration(team, info),
          ...certificate(team, info),
          ...acknowledgements(info),
          new Paragraph({ style: 'FrontTitle', pageBreakBefore: true, children: [new TextRun('Table of Contents')] }),
          new TableOfContents('Table of Contents', { hyperlink: true, headingStyleRange: '1-3' }),
          new Paragraph({ style: 'FrontTitle', pageBreakBefore: true, children: [new TextRun('List of Figures')] }),
          new TableOfContents('List of Figures', { hyperlink: true, stylesWithLevels: [new StyleLevel('FigureCaption', 1)] }),
          new Paragraph({ style: 'FrontTitle', pageBreakBefore: true, children: [new TextRun('List of Tables')] }),
          new TableOfContents('List of Tables', { hyperlink: true, stylesWithLevels: [new StyleLevel('TableCaption', 1)] }),
          ...abstractPage(team, info),
        ],
      },
      // body, arabic page numbers from 1
      {
        properties: { type: SectionType.NEXT_PAGE, page: { ...page(), pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
        headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: SHORT_TITLE, size: 18, italics: true, color: '595959' })] })] }) },
        footers: { default: footer() },
        children: content,
      },
    ],
  });
  return Packer.toBuffer(doc).then((buf) => {
    fs.writeFileSync(R.OUT, buf);
    console.log(`wrote ${path.relative(R.ROOT, R.OUT)} (${(buf.length / 1024).toFixed(0)} KB)`);
  });
}

module.exports = { build };
