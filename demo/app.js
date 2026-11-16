// Pump Sound Monitor: runs the Edge Impulse WebAssembly model on replayed clips, the live
// microphone or your own audio files, entirely in the browser.
//   01 Monitor     live decisions, machine-state alarm logic with hysteresis, event log, MFE view
//   02 Analyze     whole-clip analysis: per-window probabilities, majority vote, occlusion test
//   03 Stress test accuracy of all held-out clips as noise is mixed in at decreasing SNR
//   04 Model card  signal path, test results, device estimates, in-browser benchmark
'use strict';

const CLASS_COLOR = { normal: '#1f6feb', abnormal: '#e8452c', background: '#6b7280', uncertain: '#e0a800' };
const VERDICT_TEXT = { normal: 'NORMAL', abnormal: 'ABNORMAL', background: 'BACKGROUND', uncertain: 'UNCERTAIN' };
const HOP_MS = 250;     // live/replay: an inference every 250 ms on the latest 1 s window
const SMOOTH = 3;       // the current-window verdict averages the last 3 inferences
const MIN_CONF = 0.6;   // below this the verdict is "uncertain" (Edge Impulse's default threshold)
const STATE_WIN = 8;    // machine state looks at the last 8 decisions (2 s)
const ANALYZE_HOP = 8000; // offline analysis: 0.5 s stride, as in training (19 windows per 10 s clip)
const BANDS = [[0, 300], [300, 600], [600, 1000], [1000, 1500], [1500, 2500], [2500, 4000], [4000, 6000], [6000, 8000]];
const SNRS = [Infinity, 30, 20, 10, 5, 0, -5];
const TAP_WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
}
registerProcessor('tap', Tap);`;

const $ = (id) => document.getElementById(id);
const fmtPct = (v, d = 0) => (Number.isFinite(v) ? `${(v * 100).toFixed(d)} %` : '–');
const snrName = (s) => (Number.isFinite(s) ? `${s} dB` : 'clean');
const sleep = () => new Promise((r) => setTimeout(r, 0));

const state = {
  clf: null, labels: [], freq: 16000, winSamples: 16000, hasAnomaly: false,
  ctx: null, analyser: null, mute: null, workletReady: false,
  timer: null, replay: null, live: null, playback: null,
  clips: [], files: [], noiseCache: {}, card: null,
  // monitor session
  recent: [], tops: [], history: [], strip: [], events: [], machine: 'idle',
  count: 0, latSum: 0, correct: 0, judged: 0, audioSec: 0, upBytes: 0, sessionStart: 0, source: '', truth: null,
};

// ====================================================================== model
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error('could not load ' + src));
    document.body.appendChild(s);
  });
}

async function loadModel() {
  const status = $('model-status');
  try {
    await loadScript('model/edge-impulse-standalone.js');
    await loadScript('run-impulse.js');
    const clf = new EdgeImpulseClassifier();
    await clf.init();
    let props = {};
    try { props = clf.getProperties() || {}; } catch (e) { /* not exported by every build */ }
    state.freq = Number(props.frequency) || 16000;
    state.winSamples = Number(props.frame_sample_count) || Number(props.input_features_count) || state.freq;
    console.log('model properties', JSON.stringify(props), 'window samples', state.winSamples);
    const probe = clf.classify(new Float32Array(state.winSamples));
    state.labels = probe.results.map((r) => r.label);
    state.hasAnomaly = Boolean(props.has_anomaly);
    state.clf = clf;
    status.textContent = 'Model ready · WebAssembly';
    status.className = 'status status-ok';
    renderBars();
  } catch (err) {
    console.error(err);
    status.textContent = 'Model missing';
    status.className = 'status status-err';
    setVerdict('idle', 'NO MODEL', 'build it: python scripts/ei_pipeline.py --skip-upload --skip-retrain --targets wasm');
  }
}

// one inference on a 1 s window of samples in [-1, 1]
function infer(win) {
  const scaled = new Float32Array(win.length);
  for (let i = 0; i < win.length; i++) scaled[i] = win[i] * 32768; // int16 range, like the training WAVs
  const t0 = performance.now();
  const res = state.clf.classify(scaled);
  const ms = performance.now() - t0;
  const probs = Object.fromEntries(res.results.map((r) => [r.label, r.value]));
  return { probs, ms, top: topOf(probs) };
}
const topOf = (probs) => Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];

function toModelRate(buf, rate) {
  if (rate === state.freq && buf.length === state.winSamples) return buf;
  const out = new Float32Array(state.winSamples);
  const ratio = buf.length / state.winSamples;
  for (let i = 0; i < out.length; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const f = x - i0;
    out[i] = (buf[i0] || 0) * (1 - f) + (buf[Math.min(i0 + 1, buf.length - 1)] || 0) * f;
  }
  return out;
}

// ====================================================================== audio sources
async function fetchSamples(file) {
  const buf = await (await fetch('clips/' + file)).arrayBuffer();
  return DSP.decodeToModelRate(buf);
}

async function noiseFor(type, length, seed) {
  if (type === 'room') {
    if (!state.noiseCache.room) {
      try { state.noiseCache.room = await fetchSamples('noise_room.wav'); } catch (e) { state.noiseCache.room = null; }
    }
    if (state.noiseCache.room) {
      const room = state.noiseCache.room;
      const out = new Float32Array(length);
      const offset = (seed * 7919) % room.length;
      for (let i = 0; i < length; i++) out[i] = room[(offset + i) % room.length];
      return out;
    }
  }
  return DSP.whiteNoise(length, seed);
}

async function ensureContext() {
  if (!state.ctx) {
    let ctx;
    try { ctx = new AudioContext({ sampleRate: state.freq }); } catch (e) { ctx = new AudioContext(); }
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    const mute = ctx.createGain();
    mute.gain.value = 0;
    analyser.connect(mute);
    mute.connect(ctx.destination);
    Object.assign(state, { ctx, analyser, mute });
    levelLoop();
  }
  if (state.ctx.state === 'suspended') await state.ctx.resume();
  return state.ctx;
}

function playSamples(samples, onEnded) {
  const ctx = state.ctx;
  const buffer = ctx.createBuffer(1, samples.length, state.freq);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.connect(state.analyser);
  const start = ctx.currentTime + 0.05;
  source.start(start);
  source.onended = onEnded;
  return { source, start };
}

function stopAll() {
  clearInterval(state.timer);
  state.timer = null;
  if (state.replay) {
    try { state.replay.source.stop(); } catch (e) { /* already stopped */ }
    state.replay.button && state.replay.button.classList.remove('playing');
    state.replay = null;
  }
  if (state.live) {
    state.live.stream.getTracks().forEach((t) => t.stop());
    state.live.src.disconnect();
    state.live.node.disconnect();
    state.live = null;
    $('mic-btn').textContent = 'Start microphone';
    $('mic-btn').classList.remove('stop');
    $('level').style.width = '0';
  }
  stopPlayback();
}

function levelLoop() {
  const time = new Float32Array(state.analyser.fftSize);
  const step = () => {
    if (state.live) {
      state.analyser.getFloatTimeDomainData(time);
      $('level').style.width = `${Math.min(100, DSP.rms(time) * 500)}%`;
    }
    requestAnimationFrame(step);
  };
  step();
}

// ====================================================================== 01 monitor
function renderBars() {
  $('bars').innerHTML = state.labels.map((l) => `
    <div class="bar-row"><span>${l}</span>
      <div class="track"><div class="fill" id="bar-${l}" style="background:${CLASS_COLOR[l] || CLASS_COLOR.uncertain}"></div></div>
      <span class="bar-val" id="val-${l}">–</span></div>`).join('');
}

function setVerdict(kind, text, sub) {
  $('verdict').className = `verdict v-${kind}`;
  $('verdict-text').textContent = text;
  $('verdict-sub').textContent = sub || '';
}

const MACHINE = {
  idle: { cls: 'm-idle', text: 'NO DATA', chip: 'c-nomachine' },
  healthy: { cls: 'm-healthy', text: 'HEALTHY', chip: 'c-healthy' },
  watch: { cls: 'm-watch', text: 'WATCH', chip: 'c-watch' },
  alarm: { cls: 'm-alarm', text: 'ALARM', chip: 'c-alarm' },
  nomachine: { cls: 'm-nomachine', text: 'NO MACHINE', chip: 'c-nomachine' },
};

// Alarm logic with hysteresis, as a real monitor would do it: a single odd window never raises
// an alarm, and an alarm only clears once the machine has sounded normal for a while.
function nextMachineState(prev, tops) {
  const n = tops.length;
  if (n < 4) return prev;
  const abn = tops.filter((t) => t === 'abnormal').length;
  const bg = tops.filter((t) => t === 'background').length;
  const scaled = (k) => (k * STATE_WIN) / n;
  if (scaled(bg) >= 5) return 'nomachine';
  if (prev === 'alarm') return scaled(abn) <= 2 ? (scaled(abn) >= 1 ? 'watch' : 'healthy') : 'alarm';
  if (scaled(abn) >= 6) return 'alarm';
  if (scaled(abn) >= 3) return 'watch';
  return 'healthy';
}

function resetSession(source, truth, truthText) {
  Object.assign(state, {
    recent: [], tops: [], history: [], strip: [], machine: 'idle', correct: 0, judged: 0,
    audioSec: 0, upBytes: 0, sessionStart: performance.now(), source, truth,
  });
  $('strip').innerHTML = '';
  $('truth').textContent = truthText || '';
  setMachine('idle', 'ALARM = 6 of the last 8 windows abnormal');
  drawProbChart();
}

function setMachine(m, sub) {
  $('machine').className = `machine ${MACHINE[m].cls}`;
  $('machine-text').textContent = MACHINE[m].text;
  if (sub !== undefined) $('machine-sub').textContent = sub;
}

function logEvent(m, detail) {
  const ol = $('events');
  if (ol.firstElementChild && ol.firstElementChild.classList.contains('muted')) ol.innerHTML = '';
  const t = new Date();
  const time = t.toLocaleTimeString([], { hour12: false });
  state.events.push({ time: t.toISOString(), state: m, source: state.source, detail });
  state.upBytes += 16; // a state change is all an edge device would need to send upstream
  const li = document.createElement('li');
  li.innerHTML = `<time>${time}</time><span class="chip ${MACHINE[m].chip}">${MACHINE[m].text}</span><span>${detail}</span>`;
  ol.prepend(li);
}

function monitorStep(win) {
  const r = infer(win);
  state.count += 1;
  state.latSum += r.ms;
  state.audioSec += HOP_MS / 1000;
  state.recent.push(r.probs);
  if (state.recent.length > SMOOTH) state.recent.shift();
  const avg = {};
  for (const l of state.labels) avg[l] = state.recent.reduce((s, p) => s + p[l], 0) / state.recent.length;
  const best = topOf(avg);
  const verdict = avg[best] >= MIN_CONF ? best : 'uncertain';
  if (state.truth) {
    state.judged += 1;
    if (verdict === state.truth) state.correct += 1;
  }

  // machine state
  state.tops.push(r.top);
  if (state.tops.length > STATE_WIN) state.tops.shift();
  const m = nextMachineState(state.machine, state.tops);
  const abn = state.tops.filter((t) => t === 'abnormal').length;
  if (m !== state.machine) {
    const conf = state.recent.reduce((s, p) => s + p[r.top], 0) / state.recent.length;
    logEvent(m, `${abn}/${state.tops.length} recent windows abnormal · ${state.source}`);
    state.machine = m;
    setMachine(m, `${abn} of the last ${state.tops.length} windows abnormal · mean confidence ${conf.toFixed(2)}`);
  } else if (m !== 'idle') {
    $('machine-sub').textContent = `${abn} of the last ${state.tops.length} windows abnormal`;
  }

  // history for the chart, strip and CSV
  const t = state.audioSec;  // seconds of audio heard in this session
  state.history.push({ t, probs: r.probs, verdict, machine: state.machine, truth: state.truth || '', source: state.source });
  if (state.history.length > 2400) state.history.shift();

  // UI
  for (const l of state.labels) {
    $(`bar-${l}`).style.width = `${(r.probs[l] * 100).toFixed(1)}%`;
    $(`val-${l}`).textContent = r.probs[l].toFixed(2);
  }
  setVerdict(verdict, VERDICT_TEXT[verdict], `p = ${avg[best].toFixed(2)} · average of last ${state.recent.length} windows`);
  $('t-lat').textContent = r.ms.toFixed(2);
  $('t-count').textContent = state.count;
  $('t-agree').textContent = state.judged ? fmtPct(state.correct / state.judged) : '–';
  $('t-uplink').textContent = `${state.upBytes} B / ${Math.round(state.audioSec * 32)} KB`;
  const block = document.createElement('i');
  block.style.background = CLASS_COLOR[verdict];
  if (state.truth && verdict !== state.truth) block.className = 'miss';
  block.title = `${verdict} (${avg[best].toFixed(2)})${state.truth && verdict !== state.truth ? ' · wrong' : ''}`;
  const strip = $('strip');
  strip.appendChild(block);
  while (strip.children.length > 100) strip.removeChild(strip.firstChild);
  drawProbChart();
  drawLiveMfe(win);
}

function drawProbChart() {
  const pts = state.history.filter((h) => h.source === state.source).slice(-60);
  const series = state.labels.map((l) => ({ name: l, color: CLASS_COLOR[l], values: pts.map((h) => h.probs[l]) }));
  const x = pts.map((h) => h.t);
  Charts.line($('prob-chart'), {
    series, x, yMin: 0, yMax: 1, yTicks: [0, 0.5, 1], yFormat: (v) => v.toFixed(1),
    xTicks: x.length ? [x[0], x[x.length - 1]] : [], xFormat: (v) => `${v.toFixed(1)} s`,
    endLabels: x.length > 0, empty: 'Play a clip or start the microphone',
    tooltip: (i) => `<b>${pts[i].t.toFixed(2)} s</b> · decision ${pts[i].verdict}<br>` +
      state.labels.map((l) => `<i style="background:${CLASS_COLOR[l]}"></i>${l} ${pts[i].probs[l].toFixed(3)}`).join('<br>'),
  });
}

function drawLiveMfe(win) {
  const f = DSP.mfe(win);
  Charts.heatmap($('mfe-live'), {
    ...f,
    tooltip: (fr, b, v) => `<b>${(fr * 16).toFixed(0)} ms</b> · mel band ${b + 1} (~${Math.round(DSP.melCentres[b])} Hz)<br>value ${v.toFixed(3)}`,
  });
}

// ---------- replay ----------
async function loadClips() {
  try {
    state.clips = await (await fetch('clips/clips.json')).json();
  } catch (e) {
    $('clip-list').textContent = 'No demo clips found. Run: python scripts/prepare_dataset.py';
    return;
  }
  renderClipList();
  fillAnalyzeSelect();
}

function clipTitle(clip) {
  if (clip.label === 'unknown') return { tag: 'not a pump', name: clip.machine_id.replace(/_/g, ' ') };
  return { tag: clip.label, name: `Clip ${clip.file.split('_')[1]} · pump ${clip.machine_id}` };
}

function renderClipList() {
  const list = $('clip-list');
  list.innerHTML = '';
  state.clips.forEach((clip, i) => {
    const b = document.createElement('button');
    b.className = 'clip';
    const { tag, name } = clipTitle(clip);
    const colour = CLASS_COLOR[clip.label] || CLASS_COLOR.uncertain;
    b.innerHTML = `<span class="bar-c" style="background:${colour}"></span><span class="name"><b>${tag}</b><span>${name}</span></span>` +
      (i < 10 ? `<kbd>${(i + 1) % 10}</kbd>` : '');
    b.onclick = () => playClip(clip, b);
    list.appendChild(b);
  });
}

async function playClip(clip, button, samplesOverride) {
  if (!state.clf) return;
  stopAll();
  await ensureContext();
  let samples = samplesOverride || await fetchSamples(clip.file);
  const snr = $('snr-sel').value;
  const noiseType = $('noise-sel').value;
  const noisy = !samplesOverride && snr !== 'off';
  if (noisy) samples = DSP.mixAtSnr(samples, await noiseFor(noiseType, samples.length, 7), Number(snr));
  const truth = ['normal', 'abnormal', 'background'].includes(clip.label) ? clip.label : null;
  const { name } = clipTitle(clip);
  const label = clip.label === 'unknown' ? 'not a pump sound (the model has no such class)' : clip.label.toUpperCase();
  resetSession(name + (noisy ? ` + ${noiseType} noise @ ${snr} dB` : ''), truth,
    clip.isFile ? `Your file: ${clip.name}` : `True label: ${label}${noisy ? ` · ${noiseType} noise mixed in at ${snr} dB SNR` : ''}`);
  const { source, start } = playSamples(samples, () => {
    if (!state.replay || state.replay.source !== source) return;
    clearInterval(state.timer);
    state.timer = null;
    button && button.classList.remove('playing');
    if (truth) $('truth').textContent += ` · ${state.correct}/${state.judged} decisions correct (${fmtPct(state.correct / state.judged)})`;
    state.replay = null;
  });
  state.replay = { samples, source, start, button };
  button && button.classList.add('playing');
  state.timer = setInterval(() => {
    const r = state.replay;
    if (!r) return;
    const end = Math.min(Math.floor((state.ctx.currentTime - r.start) * state.freq), r.samples.length);
    if (end >= state.winSamples) monitorStep(r.samples.subarray(end - state.winSamples, end));
  }, HOP_MS);
}

// ---------- live microphone ----------
async function toggleMic() {
  if (state.live) { stopAll(); return; }
  if (!state.clf) return;
  stopAll();
  const ctx = await ensureContext();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
  } catch (e) {
    $('truth').textContent = 'Microphone permission denied or no microphone found.';
    return;
  }
  if (!state.workletReady) {
    await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP_WORKLET], { type: 'application/javascript' })));
    state.workletReady = true;
  }
  const src = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, 'tap');
  const ring = new Float32Array(Math.round((state.winSamples / state.freq) * ctx.sampleRate));
  const live = { stream, src, node, ring, write: 0, filled: 0 };
  node.port.onmessage = (e) => {
    const chunk = e.data;
    for (let i = 0; i < chunk.length; i++) {
      ring[live.write] = chunk[i];
      live.write = (live.write + 1) % ring.length;
    }
    live.filled = Math.min(ring.length, live.filled + chunk.length);
  };
  src.connect(node);
  src.connect(state.analyser);
  node.connect(state.mute);
  state.live = live;
  resetSession('live microphone', null, 'Listening to the microphone…');
  $('mic-btn').textContent = 'Stop microphone';
  $('mic-btn').classList.add('stop');
  state.timer = setInterval(() => {
    if (live.filled < ring.length) return;
    const win = new Float32Array(ring.length);
    win.set(ring.subarray(live.write), 0);
    win.set(ring.subarray(0, live.write), ring.length - live.write);
    monitorStep(toModelRate(win, ctx.sampleRate));
  }, HOP_MS);
}

// ---------- your files ----------
async function addFiles(fileList) {
  for (const f of fileList) {
    try {
      const samples = await DSP.decodeToModelRate(await f.arrayBuffer());
      if (samples.length < state.winSamples) throw new Error('shorter than 1 s');
      const clip = { isFile: true, name: f.name, label: 'file', machine_id: f.name, file: `file:${state.files.length}`, samples };
      state.files.push(clip);
      const b = document.createElement('button');
      b.className = 'clip';
      b.innerHTML = `<span class="bar-c" style="background:#101418"></span><span class="name"><b>your file · ${(samples.length / state.freq).toFixed(1)} s</b><span>${f.name}</span></span>`;
      b.onclick = () => playClip(clip, b, clip.samples);
      $('file-list').appendChild(b);
      fillAnalyzeSelect();
    } catch (e) {
      $('truth').textContent = `Could not read ${f.name}: ${e.message}`;
    }
  }
}

function exportCsv() {
  const rows = [['time_s', 'source', 'true_label', ...state.labels.map((l) => `p_${l}`), 'decision', 'machine_state']];
  for (const h of state.history) rows.push([h.t.toFixed(2), h.source, h.truth, ...state.labels.map((l) => h.probs[l].toFixed(4)), h.verdict, h.machine]);
  const csv = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = `pump-monitor-session-${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.csv`;
  a.click();
}

// ====================================================================== 02 analyze
function fillAnalyzeSelect() {
  const sel = $('an-sel');
  const keep = sel.value;
  sel.innerHTML = state.clips.map((c, i) => {
    const { tag, name } = clipTitle(c);
    return `<option value="c${i}">${tag} · ${name}</option>`;
  }).join('') + state.files.map((c, i) => `<option value="f${i}">your file · ${c.name}</option>`).join('');
  if (keep) sel.value = keep;
}

function selectedClip() {
  const v = $('an-sel').value || 'c0';
  return v[0] === 'f' ? state.files[Number(v.slice(1))] : state.clips[Number(v.slice(1))];
}

function stopPlayback() {
  if (state.playback) { try { state.playback.source.stop(); } catch (e) { /* stopped */ } state.playback = null; }
  $('an-play').textContent = 'Play';
}

async function analyze(clip) {
  if (!state.clf || !clip) return;
  const btn = $('an-run');
  btn.disabled = true;
  stopPlayback();
  const samples = clip.samples || await fetchSamples(clip.file);
  state.analyzed = { clip, samples };
  const truth = ['normal', 'abnormal', 'background'].includes(clip.label) ? clip.label : null;

  // 1) every window, 0.5 s stride
  const wins = [];
  for (let end = state.winSamples; end <= samples.length; end += ANALYZE_HOP) {
    const r = infer(samples.subarray(end - state.winSamples, end));
    wins.push({ end, t: (end - state.winSamples) / state.freq + 0.5, ...r });
  }
  const votes = {};
  for (const w of wins) votes[w.top] = (votes[w.top] || 0) + 1;
  const majority = topOf(votes);
  const verdictKind = VERDICT_TEXT[majority] ? majority : 'uncertain';
  $('an-verdict').className = `verdict v-${verdictKind}`;
  $('an-verdict-text').textContent = VERDICT_TEXT[majority] || majority.toUpperCase();
  const truthNote = truth ? (majority === truth ? ' · matches the true label' : ` · true label is ${truth}`)
    : (clip.label === 'unknown' ? ' · not a pump: no class fits' : '');
  $('an-verdict-sub').textContent = `${votes[majority]} of ${wins.length} windows${truthNote}`;
  $('an-votes').innerHTML = state.labels.map((l) => `<div class="vote"><span>${l}</span>` +
    `<div class="track"><div class="fill" style="width:${((votes[l] || 0) / wins.length) * 100}%;background:${CLASS_COLOR[l]}"></div></div>` +
    `<b class="num">${votes[l] || 0}</b></div>`).join('');

  // 2) features of the whole clip
  const feats = DSP.mfe(samples);
  const dur = samples.length / state.freq;
  $('an-dur').textContent = `${dur.toFixed(1)} s`;
  const drawMfe = (hi) => Charts.heatmap($('an-mfe'), {
    ...feats,
    highlight: hi,
    tooltip: (fr, b, v) => `<b>${((fr * 256) / state.freq).toFixed(2)} s</b> · mel band ${b + 1} (~${Math.round(DSP.melCentres[b])} Hz)<br>value ${v.toFixed(3)}`,
  });
  drawMfe(null);
  $('an-mfe-note').textContent = `${feats.frames} frames × ${feats.bands} bands`;

  Charts.line($('an-chart'), {
    series: state.labels.map((l) => ({ name: l, color: CLASS_COLOR[l], values: wins.map((w) => w.probs[l]) })),
    x: wins.map((w) => w.t), xMin: 0, xMax: dur, yMin: 0, yMax: 1, yTicks: [0, 0.5, 1], yFormat: (v) => v.toFixed(1),
    xTicks: [0, dur / 2, dur], xFormat: (v) => `${v.toFixed(1)} s`, endLabels: true, markers: true,
    tooltip: (i) => {
      const w = wins[i];
      const f0 = (w.end - state.winSamples) / 256;
      drawMfe([f0, Math.min(feats.frames, f0 + 61)]);
      return `<b>window ${i + 1}: ${(w.t - 0.5).toFixed(1)}–${(w.t + 0.5).toFixed(1)} s</b><br>` +
        state.labels.map((l) => `<i style="background:${CLASS_COLOR[l]}"></i>${l} ${w.probs[l].toFixed(3)}`).join('<br>');
    },
  });
  $('an-chart').onmouseleave = () => drawMfe(null);

  // 3) occlusion: remove each band from every window and re-run the model
  $('an-status').textContent = 'Running the occlusion test…';
  $('occ-count').textContent = `${wins.length * BANDS.length}`;
  const importance = [];
  for (const [lo, hi] of BANDS) {
    let drop = 0;
    for (const w of wins) {
      const removed = DSP.removeBand(samples.subarray(w.end - state.winSamples, w.end), lo, hi);
      const r = infer(removed);
      drop += w.probs[majority] - r.probs[majority];
    }
    importance.push(drop / wins.length);
    await sleep();
  }
  const fmtHz = (hz) => (hz >= 1000 ? `${hz / 1000} k` : `${hz}`);
  const items = BANDS.map(([lo, hi], i) => ({
    label: `${fmtHz(lo)}–${fmtHz(hi)} Hz`, value: importance[i],
    color: importance[i] >= 0 ? (CLASS_COLOR[majority] || '#101418') : '#b9b6ad',
  }));
  Charts.bars($('an-occ'), {
    items, format: (v) => (v >= 0 ? '+' : '') + v.toFixed(2),
    tooltip: (i) => `<b>${items[i].label}</b><br>removing it changes p(${majority}) by ${(-importance[i]).toFixed(3)} on average`,
  });
  const k = importance.indexOf(Math.max(...importance));
  $('occ-take').textContent = importance[k] > 0.02
    ? `The model's ${majority} decision leans most on ${items[k].label}: removing that band lowers p(${majority}) by ${importance[k].toFixed(2)} on average.`
    : `No single band changes the ${majority} decision much: the evidence is spread across the spectrum.`;
  $('an-status').textContent = `${wins.length} windows + ${wins.length * BANDS.length} occlusion inferences, all in this browser.`;
  btn.disabled = false;
}

async function togglePlayback() {
  if (state.playback) { stopPlayback(); return; }
  const clip = selectedClip();
  if (!clip) return;
  await ensureContext();
  const samples = clip.samples || await fetchSamples(clip.file);
  state.playback = playSamples(samples, () => stopPlayback());
  $('an-play').textContent = 'Stop';
}

// ====================================================================== 03 stress test
async function stressTest() {
  if (!state.clf) return;
  const btn = $('st-run');
  btn.disabled = true;
  const noiseType = document.querySelector('#st-noise button.active').dataset.noise;
  const clips = state.clips.filter((c) => c.label === 'normal' || c.label === 'abnormal');
  const clean = [];
  for (const c of clips) clean.push(await fetchSamples(c.file));
  const results = [];
  const total = SNRS.length * clips.length;
  let done = 0;
  for (const snr of SNRS) {
    const acc = { snr, win: 0, winOk: 0, clipOk: 0, rec: { normal: [0, 0], abnormal: [0, 0] }, wrong: {} };
    for (let ci = 0; ci < clips.length; ci++) {
      const truth = clips[ci].label;
      const noise = await noiseFor(noiseType, clean[ci].length, ci + 1);
      const mixed = DSP.mixAtSnr(clean[ci], noise, snr);
      const votes = {};
      for (let end = state.winSamples; end <= mixed.length; end += ANALYZE_HOP) {
        const { top } = infer(mixed.subarray(end - state.winSamples, end));
        votes[top] = (votes[top] || 0) + 1;
        acc.win += 1;
        acc.rec[truth][1] += 1;
        if (top === truth) { acc.winOk += 1; acc.rec[truth][0] += 1; } else acc.wrong[top] = (acc.wrong[top] || 0) + 1;
      }
      if (topOf(votes) === truth) acc.clipOk += 1;
      done += 1;
      $('st-prog').style.width = `${(done / total) * 100}%`;
      $('st-status').textContent = `${snrName(snr)} · clip ${ci + 1}/${clips.length}`;
      await sleep();
    }
    acc.winAcc = acc.winOk / acc.win;
    acc.clipAcc = acc.clipOk / clips.length;
    results.push(acc);
    drawStress(results, noiseType);
  }
  const inferences = results.reduce((s, r) => s + r.win, 0);
  $('st-status').textContent = `${inferences} inferences on ${clips.length} clips × ${SNRS.length} noise levels.`;
  const base = results[0].winAcc;
  const holds = results.filter((r) => r.winAcc >= base - 0.05);
  const lowest = holds[holds.length - 1];
  const worst = results[results.length - 1];
  const best = results.reduce((a, b) => (b.winAcc > a.winAcc ? b : a));
  $('st-take').textContent = `With ${noiseType === 'room' ? 'room' : 'white'} noise, window accuracy stays within 5 points of the clean ${fmtPct(base)} down to ${snrName(lowest.snr)} SNR, and is ${fmtPct(worst.winAcc)} at ${snrName(worst.snr)}.` +
    (best.winAcc > base + 0.02 ? ` Light noise (${snrName(best.snr)}) even raises it to ${fmtPct(best.winAcc)}, consistent with the noise-augmented training data.` : '');
  btn.disabled = false;
}

function drawStress(results, noiseType) {
  const idx = SNRS.map((_, i) => i);
  Charts.line($('st-chart'), {
    series: [
      { name: 'windows', color: '#101418', values: results.map((r) => r.winAcc) },
      { name: 'clips', color: '#1f6feb', dash: [6, 4], values: results.map((r) => r.clipAcc) },
    ],
    x: idx.slice(0, results.length), xMin: 0, xMax: SNRS.length - 1, xTicks: idx, xFormat: (i) => snrName(SNRS[i]),
    yMin: 0, yMax: 1, yTicks: [0, 0.25, 0.5, 0.75, 1], yFormat: (v) => `${Math.round(v * 100)}%`,
    endLabels: true, markers: true, shade: [2, 4],
    tooltip: (i) => `<b>${snrName(results[i].snr)} · ${noiseType} noise</b><br>windows correct ${fmtPct(results[i].winAcc, 1)}<br>clips correct ${fmtPct(results[i].clipAcc)}`,
  });
  $('st-table').querySelector('tbody').innerHTML = results.map((r) => {
    const wrong = Object.entries(r.wrong).sort((a, b) => b[1] - a[1])[0];
    return `<tr><td>${snrName(r.snr)}</td><td class="num">${fmtPct(r.winAcc, 1)}</td><td class="num">${fmtPct(r.clipAcc)}</td>` +
      `<td class="num">${fmtPct(r.rec.normal[0] / r.rec.normal[1])}</td><td class="num">${fmtPct(r.rec.abnormal[0] / r.rec.abnormal[1])}</td>` +
      `<td>${wrong ? `${wrong[0]} (${wrong[1]} windows)` : '–'}</td></tr>`;
  }).join('');
}

// ====================================================================== 04 model card
async function loadCard() {
  try { state.card = await (await fetch('model_card.json')).json(); } catch (e) { state.card = null; }
  renderCard();
}

function renderCard() {
  const c = state.card || {};
  const stages = [
    ['Microphone', '16 000', 'samples, 1 s at 16 kHz', ''],
    ['MFE (DSP)', '61 × 40', '32 ms frames, 40 mel bands', 'dsp'],
    ['Conv1D 16, k3', '61 × 16', '1 936 parameters', ''],
    ['Max-pool', '31 × 16', 'dropout 0.25', ''],
    ['Conv1D 32, k3', '31 × 32', '1 568 parameters', ''],
    ['Max-pool', '16 × 32', 'dropout 0.25', ''],
    ['Dense + softmax', '3', '1 539 parameters', ''],
    ['Decision', 'normal · abnormal · background', 'int8, 5 043 parameters', 'out'],
  ];
  $('pipeline').innerHTML = stages.map(([b, code, span, cls], i) =>
    `${i ? '<span class="arrow">→</span>' : ''}<div class="stage ${cls}"><b>${b}</b><code>${code}</code><span>${span}</span></div>`).join('');

  const t = c.test || {};
  const row = (name, k, fmt = fmtPct) => `<tr><td>${name}</td><td class="num">${fmt(get(t, `v2.${k}`))}</td><td class="num">${fmt(get(t, `v1.${k}`))}</td></tr>`;
  $('card-test').innerHTML = c.test ? `<table class="data"><thead><tr><th>Metric (int8)</th><th class="num">Final model</th><th class="num">v1 (longer training)</th></tr></thead><tbody>` +
    row('1 s windows correct', 'window_accuracy') + row('10 s clips correct (majority)', 'clip_accuracy') +
    row('Abnormal recall', 'recall.abnormal') + row('Normal recall', 'recall.normal') + row('Background recall', 'recall.background') +
    row('Laptop-microphone clips correct', 'mic_clip_accuracy') +
    `</tbody></table><p class="muted small">v2 trained for 240 s (free-plan compute limit); v1 for 100 epochs. Same data and network.</p>` : '<p class="muted">model_card.json missing: run python scripts/make_model_card.py</p>';

  const cm = get(c, 'test.v2.confusion');
  if (cm) {
    const labels = Object.keys(cm);
    $('card-cm').innerHTML = `<table class="data cm"><thead><tr><th></th>${labels.map((l) => `<th class="num">${l}</th>`).join('')}</tr></thead><tbody>` +
      labels.map((tr) => {
        const total = labels.reduce((s, p) => s + cm[tr][p], 0) || 1;
        return `<tr><th>${tr}</th>` + labels.map((p) => {
          const share = cm[tr][p] / total;
          const bg = `rgba(16,20,24,${(0.06 + share * 0.94).toFixed(2)})`;
          return `<td class="cell" style="background:${bg};color:${share > 0.45 ? '#fff' : '#101418'}">${cm[tr][p]}<small>${fmtPct(share)}</small></td>`;
        }).join('') + '</tr>';
      }).join('') + '</tbody></table>';
  }

  const d = c.device || {};
  const budget = 500;
  $('card-device').innerHTML = `<div class="facts" style="grid-template-columns:repeat(2,1fr)">` +
    `<div class="fact"><span>${d.int8_ms ?? '–'} ms</span><small>classifier per window, int8 + EON</small></div>` +
    `<div class="fact"><span>${d.float32_ms ?? '–'} ms</span><small>same network in float32</small></div>` +
    `<div class="fact"><span>${d.ram_kb ?? '–'} KB</span><small>classifier RAM (of 256 KB)</small></div>` +
    `<div class="fact"><span>${d.flash_kb ?? '–'} KB</span><small>classifier flash (of 1 MB)</small></div></div>` +
    `<p class="small" style="margin-top:12px">Real-time budget: a new window every ${budget} ms. The classifier uses <b>${d.int8_ms ?? '–'} ms</b>:</p>` +
    `<div class="track" style="height:18px"><div class="fill" style="width:${Math.max(1, ((d.int8_ms || 0) / budget) * 100)}%;background:#101418"></div></div>`;

  const ds = c.data || {};
  $('card-data').innerHTML = [
    [ds.clips ?? '–', 'clips uploaded (10 s each)'], [ds.train_windows ?? '–', 'training windows (1 s, 0.5 s stride)'],
    [ds.pump_units ?? 4, 'pump units in the dataset'], [ds.mic_clips ?? '–', 'clips recorded through the laptop mic'],
    [ds.augmented ?? '–', 'noise-augmented clips (5–20 dB SNR)'], [ds.test_clips ?? 141, 'held-out test clips'],
    [c.train?.val_int8 ? fmtPct(c.train.val_int8, 1) : '–', 'validation accuracy, int8'], ['5 043', 'model parameters'],
  ].map(([v, l]) => `<div class="fact"><span>${v}</span><small>${l}</small></div>`).join('');
}
const get = (o, path) => path.split('.').reduce((x, k) => (x == null ? undefined : x[k]), o);

async function benchmark() {
  if (!state.clf) return;
  const btn = $('bench-btn');
  btn.disabled = true;
  const known = state.clips.filter((c) => c.label !== 'unknown');
  const samples = await fetchSamples(known[0].file);
  const win = samples.subarray(0, state.winSamples);
  const times = [];
  for (let i = 0; i < 300; i++) { times.push(infer(win).ms); if (i % 50 === 0) await sleep(); }
  times.sort((a, b) => a - b);
  const q = (p) => times[Math.min(times.length - 1, Math.floor(p * times.length))];
  const mean = times.reduce((s, x) => s + x, 0) / times.length;
  const ms = (v) => (v >= 0.005 ? `${v.toFixed(2)} ms` : '< 0.01 ms');
  const t0 = performance.now();
  for (let i = 0; i < 20; i++) DSP.mfe(win);
  const jsMfe = (performance.now() - t0) / 20;
  $('card-bench').innerHTML = `<div class="facts" style="grid-template-columns:repeat(2,1fr)">` +
    `<div class="fact"><span>${ms(q(0.5))}</span><small>median inference (DSP + network, WebAssembly)</small></div>` +
    `<div class="fact"><span>${ms(q(0.95))}</span><small>95th percentile over 300 runs</small></div>` +
    `<div class="fact"><span>${mean > 0.005 ? `${Math.round(1000 / mean)}×` : '–'}</span><small>faster than real time (1 s of audio per inference)</small></div>` +
    `<div class="fact"><span>${ms(jsMfe)}</span><small>MFE recomputed in JavaScript for the display</small></div></div>`;
  await evaluateAll();
  btn.disabled = false;
}

// held-out demo clips, 0.25 s stride (used by the benchmark and by scripts/demo_check.py ?evaluate=1)
async function evaluateAll() {
  const rows = [];
  let count = 0;
  let latSum = 0;
  for (const clip of state.clips) {
    const samples = await fetchSamples(clip.file);
    const hop = Math.round((state.freq * HOP_MS) / 1000);
    const votes = {};
    let correct = 0;
    let total = 0;
    for (let end = state.winSamples; end <= samples.length; end += hop) {
      const r = infer(samples.subarray(end - state.winSamples, end));
      count += 1;
      latSum += r.ms;
      votes[r.top] = (votes[r.top] || 0) + 1;
      correct += r.top === clip.label ? 1 : 0;
      total += 1;
    }
    rows.push({ file: clip.file, label: clip.label, machine_id: clip.machine_id, windows: total,
      windowAccuracy: correct / total, majority: topOf(votes), meanAnomaly: 0 });
    await sleep();
  }
  const known = rows.filter((r) => r.label !== 'unknown');
  const winAcc = known.reduce((s, r) => s + r.windowAccuracy * r.windows, 0) / known.reduce((s, r) => s + r.windows, 0);
  const clipAcc = known.filter((r) => r.majority === r.label).length / known.length;
  const avgLatency = latSum / count;
  $('eval').classList.remove('hidden');
  $('eval-summary').innerHTML = `Pump clips: <b>${fmtPct(winAcc, 1)}</b> of windows and <b>${fmtPct(clipAcc)}</b> of clips correct · ` +
    `${count} inferences, ${avgLatency.toFixed(2)} ms each. The “not a pump” sounds are forced into a pump class: the final model has no anomaly block.`;
  $('eval-table').innerHTML = '<thead><tr><th>Clip</th><th>True label</th><th>Majority prediction</th><th class="num">Windows correct</th></tr></thead><tbody>' +
    rows.map((r) => `<tr><td>${r.file}</td><td>${r.label === 'unknown' ? 'not a pump' : r.label}</td>` +
      `<td class="${r.label === 'unknown' ? '' : (r.majority === r.label ? 'ok' : 'bad')}">${r.majority}</td>` +
      `<td class="num">${r.label === 'unknown' ? '–' : fmtPct(r.windowAccuracy)}</td></tr>`).join('') + '</tbody>';
  $('eval-json').textContent = JSON.stringify({ windowAccuracy: winAcc, clipAccuracy: clipAcc, inferences: count,
    meanLatencyMs: avgLatency, meanAnomalyKnown: 0, meanAnomalyUnknown: 0, threshold: null, clips: rows });
  document.title = 'EVAL DONE';
}

// ====================================================================== wiring
function showTab(name) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
  for (const id of ['monitor', 'analyze', 'stress', 'card']) $(`tab-${id}`).classList.toggle('hidden', id !== name);
  Charts.hideTip();
  if (name === 'card') renderCard();
}

function showSource(src) {
  document.querySelectorAll('#source-seg button').forEach((b) => b.classList.toggle('active', b.dataset.src === src));
  for (const s of ['replay', 'live', 'file']) $(`src-${s}`).classList.toggle('hidden', s !== src);
}

document.querySelectorAll('.tab').forEach((t) => { t.onclick = () => { if (t.dataset.tab !== 'monitor') stopAll(); showTab(t.dataset.tab); }; });
document.querySelectorAll('#source-seg button').forEach((b) => { b.onclick = () => { stopAll(); showSource(b.dataset.src); }; });
document.querySelectorAll('#st-noise button').forEach((b) => {
  b.onclick = () => document.querySelectorAll('#st-noise button').forEach((x) => x.classList.toggle('active', x === b));
});
$('mic-btn').onclick = toggleMic;
$('csv-btn').onclick = exportCsv;
$('an-run').onclick = () => analyze(selectedClip());
$('an-sel').onchange = () => analyze(selectedClip());
$('an-play').onclick = togglePlayback;
$('st-run').onclick = stressTest;
$('bench-btn').onclick = benchmark;
$('file-in').onchange = (e) => addFiles(e.target.files);
const drop = $('drop');
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); addFiles(e.dataTransfer.files); };

document.addEventListener('keydown', (e) => {
  if (e.target.closest('input, select, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'Escape') { stopAll(); return; }
  if (e.key === 'm' || e.key === 'M') { showTab('monitor'); showSource('live'); toggleMic(); return; }
  if (/^[0-9]$/.test(e.key)) {
    const i = (Number(e.key) + 9) % 10;
    if (!state.clips[i]) return;
    showTab('monitor');
    showSource('replay');
    playClip(state.clips[i], $('clip-list').children[i]);
  }
});
window.addEventListener('resize', () => { drawProbChart(); });

document.querySelectorAll('canvas[height]').forEach((c) => { c.style.height = `${c.getAttribute('height')}px`; });
drawProbChart();
Charts.heatmap($('mfe-live'), { frames: 0, bands: 40, data: [], empty: 'Waiting for audio' });
loadCard();
Promise.all([loadClips(), loadModel()]).then(() => {
  if (!state.clf) return;
  const params = new URLSearchParams(location.search);
  if (params.get('evaluate')) {
    showTab('card');
    evaluateAll().catch((e) => { console.error(e); document.title = 'EVAL FAILED'; });
  } else if (params.get('offline')) {
    showTab('analyze');
    const i = state.clips.findIndex((c) => c.file === params.get('offline'));
    $('an-sel').value = `c${Math.max(0, i)}`;
    analyze(selectedClip()).then(() => { document.title = 'OFFLINE DONE'; })
      .catch((e) => { console.error(e); document.title = 'OFFLINE FAILED'; });
  } else if (params.get('simulate')) {
    // feeds a clip through the monitor without audio (screenshots, headless checks)
    const i = Number(params.get('simulate')) - 1;
    const clip = state.clips[i];
    fetchSamples(clip.file).then((samples) => {
      resetSession(clipTitle(clip).name, clip.label, `True label: ${clip.label.toUpperCase()} (simulated replay)`);
      $('clip-list').children[i].classList.add('playing');
      for (let end = state.winSamples; end <= samples.length; end += 4000) monitorStep(samples.subarray(end - state.winSamples, end));
      document.title = 'SIM DONE';
    });
  } else if (params.get('play')) {
    const i = Number(params.get('play')) - 1;
    if (params.get('snr')) $('snr-sel').value = params.get('snr');
    playClip(state.clips[i], $('clip-list').children[i]);
  } else if (params.get('tab')) {
    showTab(params.get('tab'));
    if (params.get('noise')) document.querySelectorAll('#st-noise button').forEach((x) => x.classList.toggle('active', x.dataset.noise === params.get('noise')));
    if (params.get('tab') === 'stress' && params.get('run')) stressTest().then(() => { document.title = 'STRESS DONE'; });
    if (params.get('tab') === 'card' && params.get('run')) benchmark().then(() => { document.title = 'BENCH DONE'; });
  }
});
