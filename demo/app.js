// Pump Sound Monitor: runs the Edge Impulse WebAssembly model on replayed test clips or the
// live microphone, entirely in the browser.
'use strict';

const COLORS = { normal: '#2fbf71', abnormal: '#ef4c4c', background: '#7a8796', unknown: '#f2a33a', uncertain: '#a58bd6' };
const VERDICT_TEXT = { normal: 'NORMAL', abnormal: 'ABNORMAL', background: 'BACKGROUND', unknown: 'UNKNOWN SOUND', uncertain: 'UNCERTAIN' };
const HOP_MS = 250;   // run an inference every 250 ms on the latest 1 s window
const SMOOTH = 3;     // the verdict averages the last 3 inferences
const MIN_CONF = 0.6; // below this the verdict is "uncertain"
const GAUGE_MAX = 10; // anomaly gauge full scale
const TAP_WORKLET = `class Tap extends AudioWorkletProcessor {
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) this.port.postMessage(ch.slice(0)); return true; }
}
registerProcessor('tap', Tap);`;

const $ = (id) => document.getElementById(id);
const state = {
  clf: null, labels: [], winSamples: 16000, freq: 16000, hasAnomaly: false,
  ctx: null, analyser: null, mute: null, workletReady: false, mode: 'replay',
  timer: null, recent: [], count: 0, latSum: 0, correct: 0, judged: 0,
  replay: null, live: null, threshold: 1.0,
};

function store(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* storage unavailable */ } }
function recall(key, fallback) {
  try { const v = localStorage.getItem(key); return v === null ? fallback : v; } catch (e) { return fallback; }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error('could not load ' + src));
    document.body.appendChild(s);
  });
}

// ---------- model ----------
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
    const probe = clf.classify(new Float32Array(state.winSamples));
    state.labels = probe.results.map((r) => r.label);
    state.hasAnomaly = props.has_anomaly !== undefined ? Boolean(props.has_anomaly)
      : (typeof probe.anomaly === 'number' && probe.anomaly !== 0);
    state.clf = clf;
    status.textContent = 'Model ready';
    status.className = 'pill pill-ok';
    renderModelInfo();
    renderBars();
  } catch (err) {
    console.error(err);
    status.textContent = 'Model missing';
    status.className = 'pill pill-err';
    $('verdict').textContent = 'NO MODEL';
    $('truth').textContent = 'Build it first: python scripts/ei_pipeline.py --skip-upload --skip-retrain --targets wasm';
  }
}

function renderModelInfo() {
  const rows = [
    ['Labels', state.labels.join(', ')],
    ['Window', `${Math.round(state.winSamples / state.freq * 1000)} ms @ ${state.freq} Hz`],
    ['Anomaly block', state.hasAnomaly ? 'yes (GMM)' : 'no'],
    ['Runs on', 'WebAssembly in this browser'],
  ];
  $('model-info').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

function renderBars() {
  $('bars').innerHTML = state.labels.map((l) => `
    <div class="bar-row"><span>${l}</span>
      <div class="bar"><div class="bar-fill" id="bar-${l}" style="background:${COLORS[l] || COLORS.uncertain}"></div></div>
      <span class="bar-val" id="val-${l}">&ndash;</span></div>`).join('');
}

// ---------- inference ----------
function toModelRate(buf, rate) {
  if (rate === state.freq && buf.length === state.winSamples) return buf;
  const out = new Float32Array(state.winSamples);
  const ratio = buf.length / state.winSamples;
  for (let i = 0; i < out.length; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const f = x - i0;
    const a = buf[i0] || 0;
    const b = buf[Math.min(i0 + 1, buf.length - 1)] || 0;
    out[i] = a * (1 - f) + b * f;
  }
  return out;
}

function runInference(win, truth) {
  const scaled = new Float32Array(win.length);
  for (let i = 0; i < win.length; i++) scaled[i] = win[i] * 32768; // int16 range, like the training WAVs
  const t0 = performance.now();
  const res = state.clf.classify(scaled);
  const ms = performance.now() - t0;
  const probs = Object.fromEntries(res.results.map((r) => [r.label, r.value]));
  state.count += 1;
  state.latSum += ms;
  state.recent.push({ probs, anomaly: res.anomaly || 0 });
  if (state.recent.length > SMOOTH) state.recent.shift();

  const avg = {};
  for (const l of state.labels) avg[l] = state.recent.reduce((s, r) => s + (r.probs[l] || 0), 0) / state.recent.length;
  const avgAnomaly = state.recent.reduce((s, r) => s + r.anomaly, 0) / state.recent.length;
  const verdict = decide(avg, avgAnomaly);
  if (truth) {
    state.judged += 1;
    if (verdict === truth) state.correct += 1;
  }
  updateUI(probs, res.anomaly || 0, verdict, ms, truth);
}

function decide(probs, anomaly) {
  if (state.hasAnomaly && anomaly > state.threshold) return 'unknown';
  let best = null;
  let bestP = -1;
  for (const [label, p] of Object.entries(probs)) if (p > bestP) { best = label; bestP = p; }
  return bestP >= MIN_CONF ? best : 'uncertain';
}

function updateUI(probs, anomaly, verdict, ms, truth) {
  for (const l of state.labels) {
    $(`bar-${l}`).style.width = `${(probs[l] * 100).toFixed(1)}%`;
    $(`val-${l}`).textContent = probs[l].toFixed(2);
  }
  const v = $('verdict');
  v.textContent = VERDICT_TEXT[verdict] || verdict.toUpperCase();
  v.className = `verdict v-${VERDICT_TEXT[verdict] ? verdict : 'uncertain'}`;
  $('anom-val').textContent = state.hasAnomaly ? anomaly.toFixed(2) : 'n/a';
  $('anom-fill').style.width = `${Math.max(0, Math.min(100, anomaly / GAUGE_MAX * 100))}%`;
  $('lat').textContent = ms.toFixed(1);
  $('count').textContent = state.count;
  $('agree').textContent = state.judged ? `${Math.round(state.correct / state.judged * 100)}%` : '–';

  const block = document.createElement('div');
  block.className = 'tl' + (truth && verdict !== truth ? ' miss' : '');
  block.style.background = COLORS[verdict] || COLORS.uncertain;
  block.title = `${verdict} · anomaly ${anomaly.toFixed(2)}`;
  const tl = $('timeline');
  tl.appendChild(block);
  while (tl.children.length > 80) tl.removeChild(tl.firstChild);
}

function resetRun(truthText) {
  state.recent = [];
  state.correct = 0;
  state.judged = 0;
  $('timeline').innerHTML = '';
  $('truth').textContent = truthText || '';
}

// ---------- audio plumbing ----------
async function ensureContext() {
  if (!state.ctx) {
    let ctx;
    try { ctx = new AudioContext({ sampleRate: state.freq }); } catch (e) { ctx = new AudioContext(); }
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0;
    const mute = ctx.createGain();
    mute.gain.value = 0;
    analyser.connect(mute);
    mute.connect(ctx.destination);
    Object.assign(state, { ctx, analyser, mute });
    drawLoop();
  }
  if (state.ctx.state === 'suspended') await state.ctx.resume();
  return state.ctx;
}

function stopAll() {
  clearInterval(state.timer);
  state.timer = null;
  if (state.replay) {
    try { state.replay.source.stop(); } catch (e) { /* already stopped */ }
    state.replay.button.classList.remove('playing');
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
}

// ---------- replay mode ----------
async function loadClips() {
  const list = $('clip-list');
  try {
    const clips = await (await fetch('clips/clips.json')).json();
    list.innerHTML = '';
    clips.forEach((clip) => {
      const b = document.createElement('button');
      b.className = 'clip';
      const k = clip.file.split('_')[1];
      b.innerHTML = `<span class="tag tag-${clip.label}">${clip.label}</span><br>Clip ${k} · pump ID ${clip.machine_id}`;
      b.onclick = () => playClip(clip, b);
      list.appendChild(b);
    });
  } catch (e) {
    list.textContent = 'No demo clips found. Run: python scripts/prepare_dataset.py';
  }
}

async function playClip(clip, button) {
  if (!state.clf) return;
  stopAll();
  const ctx = await ensureContext();
  const audio = await ctx.decodeAudioData(await (await fetch('clips/' + clip.file)).arrayBuffer());
  const source = ctx.createBufferSource();
  source.buffer = audio;
  source.connect(ctx.destination);
  source.connect(state.analyser);
  const start = ctx.currentTime + 0.05;
  source.start(start);
  state.replay = { samples: audio.getChannelData(0), rate: audio.sampleRate, source, start, label: clip.label, button };
  resetRun(`True label: ${clip.label.toUpperCase()} (held-out clip, pump ID ${clip.machine_id})`);
  button.classList.add('playing');

  const winLen = Math.round(state.winSamples / state.freq * audio.sampleRate);
  state.timer = setInterval(() => {
    const r = state.replay;
    if (!r) return;
    const end = Math.min(Math.floor((ctx.currentTime - r.start) * r.rate), r.samples.length);
    if (end >= winLen) runInference(toModelRate(r.samples.subarray(end - winLen, end), r.rate), r.label);
  }, HOP_MS);

  source.onended = () => {
    if (!state.replay || state.replay.source !== source) return;
    clearInterval(state.timer);
    state.timer = null;
    button.classList.remove('playing');
    const pct = state.judged ? Math.round(state.correct / state.judged * 100) : 0;
    $('truth').textContent = `True label: ${clip.label.toUpperCase()} · ${state.correct}/${state.judged} inferences correct (${pct}%)`;
    state.replay = null;
  };
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
  const ring = new Float32Array(Math.round(state.winSamples / state.freq * ctx.sampleRate));
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
  resetRun('Listening… (live microphone)');
  $('mic-btn').textContent = 'Stop microphone';
  $('mic-btn').classList.add('stop');

  state.timer = setInterval(() => {
    if (live.filled < ring.length) return;
    const win = new Float32Array(ring.length);
    win.set(ring.subarray(live.write), 0);
    win.set(ring.subarray(0, live.write), ring.length - live.write);
    runInference(toModelRate(win, ctx.sampleRate), null);
  }, HOP_MS);
}

// ---------- spectrogram + level meter ----------
function heat(v) {
  const r = Math.round(255 * Math.min(1, Math.max(0, (v - 0.45) * 2.2)));
  const g = Math.round(255 * Math.min(1, v * 1.4));
  const b = Math.round(255 * Math.max(0, Math.min(1, 0.3 + v * 0.9 - Math.max(0, v - 0.7) * 2)));
  return `rgb(${r},${g},${b})`;
}

function drawLoop() {
  const canvas = $('spec');
  const g = canvas.getContext('2d');
  const bins = new Uint8Array(state.analyser.frequencyBinCount);
  const time = new Float32Array(state.analyser.fftSize);
  const step = () => {
    state.analyser.getByteFrequencyData(bins);
    const w = canvas.width;
    const h = canvas.height;
    g.drawImage(canvas, -2, 0);
    for (let y = 0; y < h; y++) {
      g.fillStyle = heat(bins[Math.floor((1 - y / h) * (bins.length - 1))] / 255);
      g.fillRect(w - 2, y, 2, 1);
    }
    if (state.live) {
      state.analyser.getFloatTimeDomainData(time);
      let sum = 0;
      for (const x of time) sum += x * x;
      $('level').style.width = `${Math.min(100, Math.sqrt(sum / time.length) * 400)}%`;
    }
    requestAnimationFrame(step);
  };
  step();
}

// ---------- wiring ----------
function setThreshold(value) {
  state.threshold = Number(value);
  $('thr').value = value;
  $('thr-val').textContent = Number(value).toFixed(1);
  $('anom-thr').style.left = `${Math.min(100, state.threshold / GAUGE_MAX * 100)}%`;
  store('pump-thr', String(value));
}

document.querySelectorAll('.tab').forEach((tab) => {
  tab.onclick = () => {
    stopAll();
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
    state.mode = tab.dataset.mode;
    $('pane-replay').classList.toggle('hidden', state.mode !== 'replay');
    $('pane-live').classList.toggle('hidden', state.mode !== 'live');
  };
});
$('mic-btn').onclick = toggleMic;
$('thr').max = GAUGE_MAX;
$('thr').oninput = (e) => setThreshold(e.target.value);
setThreshold(recall('pump-thr', '1.0'));
loadClips();
loadModel();
