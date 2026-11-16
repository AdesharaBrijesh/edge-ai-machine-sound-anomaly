// Signal processing used by the dashboard, independent of the Edge Impulse model:
//  - FFT (radix-2, in place, forward and inverse)
//  - MFE features with the impulse's parameters (32 ms frames, 16 ms stride, 40 mel filters,
//    FFT 512, 0-8000 Hz, noise floor -72 dB), so the page can show what the network receives
//  - band removal (zero a frequency band of a 1 s window) for the occlusion analysis
//  - noise mixing at a chosen signal-to-noise ratio for the stress test
//  - WAV parsing
'use strict';

const DSP = (() => {
  const SR = 16000;
  const MFE = { frame: 512, stride: 256, fft: 512, filters: 40, low: 0, high: 8000, noiseFloor: -72 };

  // ---------- FFT ----------
  function fft(re, im, inverse = false) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = ((inverse ? 2 : -2) * Math.PI) / len;
      const wr0 = Math.cos(ang);
      const wi0 = Math.sin(ang);
      const half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let wr = 1;
        let wi = 0;
        for (let k = 0; k < half; k++) {
          const a = i + k;
          const b = a + half;
          const vr = re[b] * wr - im[b] * wi;
          const vi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - vr; im[b] = im[a] - vi;
          re[a] += vr; im[a] += vi;
          const t = wr * wr0 - wi * wi0;
          wi = wr * wi0 + wi * wr0;
          wr = t;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }

  // ---------- mel filterbank ----------
  const hzToMel = (hz) => 1127 * Math.log(1 + hz / 700);
  const melToHz = (mel) => 700 * (Math.exp(mel / 1127) - 1);
  const filterbank = (() => {
    const bins = MFE.fft / 2 + 1;
    const lo = hzToMel(MFE.low);
    const hi = hzToMel(MFE.high);
    const pts = Array.from({ length: MFE.filters + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (MFE.filters + 1)));
    const bin = pts.map((hz) => Math.floor(((MFE.fft + 1) * hz) / SR));
    const fb = [];
    for (let m = 1; m <= MFE.filters; m++) {
      const w = new Float32Array(bins);
      for (let k = bin[m - 1]; k < bin[m]; k++) w[k] = (k - bin[m - 1]) / Math.max(1, bin[m] - bin[m - 1]);
      for (let k = bin[m]; k <= bin[m + 1] && k < bins; k++) w[k] = (bin[m + 1] - k) / Math.max(1, bin[m + 1] - bin[m]);
      fb.push(w);
    }
    return { weights: fb, centresHz: pts.slice(1, -1) };
  })();

  // MFE of a signal in [-1, 1]: returns { frames, data: Float32Array(frames * 40) } with values 0..1
  function mfe(signal) {
    const frames = signal.length < MFE.frame ? 0 : 1 + Math.floor((signal.length - MFE.frame) / MFE.stride);
    const out = new Float32Array(frames * MFE.filters);
    const re = new Float64Array(MFE.fft);
    const im = new Float64Array(MFE.fft);
    const power = new Float64Array(MFE.fft / 2 + 1);
    for (let f = 0; f < frames; f++) {
      const start = f * MFE.stride;
      for (let i = 0; i < MFE.fft; i++) { re[i] = i < MFE.frame ? signal[start + i] : 0; im[i] = 0; }
      fft(re, im);
      for (let k = 0; k < power.length; k++) power[k] = (re[k] * re[k] + im[k] * im[k]) / MFE.fft;
      for (let m = 0; m < MFE.filters; m++) {
        const w = filterbank.weights[m];
        let e = 0;
        for (let k = 0; k < power.length; k++) e += w[k] * power[k];
        const db = 10 * Math.log10(Math.max(e, 1e-30));
        const v = (db - MFE.noiseFloor) / (-MFE.noiseFloor + 12);
        out[f * MFE.filters + m] = Math.min(1, Math.max(0, v));
      }
    }
    return { frames, bands: MFE.filters, data: out };
  }

  // ---------- band removal (occlusion) ----------
  // Zero all frequency content between loHz and hiHz of a window (FFT -> zero -> inverse FFT).
  function removeBand(win, loHz, hiHz) {
    let n = 1;
    while (n < win.length) n <<= 1;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    re.set(win);
    fft(re, im);
    const k0 = Math.max(0, Math.floor((loHz / SR) * n));
    const k1 = Math.min(n / 2, Math.ceil((hiHz / SR) * n));
    for (let k = k0; k <= k1; k++) {
      re[k] = 0; im[k] = 0;
      if (k > 0 && k < n / 2) { re[n - k] = 0; im[n - k] = 0; }
    }
    fft(re, im, true);
    return Float32Array.from(re.subarray(0, win.length));
  }

  // ---------- noise ----------
  const rms = (x) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; return Math.sqrt(s / Math.max(1, x.length)); };

  function whiteNoise(n, seed = 1) {
    let s = seed >>> 0;
    const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {  // Box-Muller
      const u = Math.max(1e-12, rand());
      const v = rand();
      const r = Math.sqrt(-2 * Math.log(u));
      out[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return out;
  }

  // signal + noise scaled so that 20 log10(rms(signal) / rms(noise)) = snrDb; Infinity = clean
  function mixAtSnr(signal, noise, snrDb) {
    if (!isFinite(snrDb)) return signal;
    const ps = rms(signal);
    const pn = rms(noise) || 1;
    const gain = ps / (pn * Math.pow(10, snrDb / 20));
    const out = new Float32Array(signal.length);
    let peak = 0;
    for (let i = 0; i < signal.length; i++) {
      out[i] = signal[i] + gain * noise[i % noise.length];
      peak = Math.max(peak, Math.abs(out[i]));
    }
    if (peak > 1) for (let i = 0; i < out.length; i++) out[i] /= peak;  // avoid clipping
    return out;
  }

  // ---------- WAV ----------
  function parsePcm16Wav(buf, expectRate = SR) {
    const v = new DataView(buf);
    if (v.byteLength < 44 || v.getUint32(0, false) !== 0x52494646 || v.getUint32(8, false) !== 0x57415645) return null;
    let fmt = null;
    for (let p = 12; p + 8 <= v.byteLength;) {
      const id = v.getUint32(p, false);
      const size = v.getUint32(p + 4, true);
      if (id === 0x666d7420) {
        fmt = { format: v.getUint16(p + 8, true), channels: v.getUint16(p + 10, true),
          rate: v.getUint32(p + 12, true), bits: v.getUint16(p + 22, true) };
      } else if (id === 0x64617461 && fmt) {
        if (fmt.format !== 1 || fmt.channels !== 1 || fmt.bits !== 16 || fmt.rate !== expectRate) return null;
        const n = Math.floor(Math.min(size, v.byteLength - p - 8) / 2);
        const out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = v.getInt16(p + 8 + 2 * i, true) / 32768;
        return out;
      }
      p += 8 + size + (size & 1);
    }
    return null;
  }

  // Any audio file -> mono Float32 at 16 kHz (fast path for 16-bit mono 16 kHz WAV)
  async function decodeToModelRate(buf) {
    const direct = parsePcm16Wav(buf);
    if (direct) return direct;
    const probe = new OfflineAudioContext(1, 1, SR);
    const audio = await probe.decodeAudioData(buf.slice(0));
    const off = new OfflineAudioContext(1, Math.ceil(audio.duration * SR), SR);
    const src = off.createBufferSource();
    src.buffer = audio;
    src.connect(off.destination);
    src.start();
    const rendered = await off.startRendering();
    return rendered.getChannelData(0);
  }

  return { SR, MFE, fft, mfe, removeBand, whiteNoise, mixAtSnr, rms, parsePcm16Wav, decodeToModelRate, melCentres: filterbank.centresHz };
})();
