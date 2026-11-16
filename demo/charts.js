// Small canvas charts with a hover layer (crosshair + tooltip). No libraries, works offline.
'use strict';

const Charts = (() => {
  const INK = '#101418';
  const MUTED = '#5b616b';
  const GRID = '#e7e5df';
  const SURFACE = '#ffffff';
  const FONT = '12px "Cascadia Mono", Consolas, monospace';
  const LABEL_FONT = '600 12px Bahnschrift, "Segoe UI", sans-serif';

  let tip = null;
  function tooltip() {
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'chart-tip';
      document.body.appendChild(tip);
    }
    return tip;
  }
  function showTip(html, clientX, clientY) {
    const t = tooltip();
    t.innerHTML = html;
    t.style.display = 'block';
    const w = t.offsetWidth;
    const h = t.offsetHeight;
    let x = clientX + 14;
    let y = clientY - h - 10;
    if (x + w > window.innerWidth - 8) x = clientX - w - 14;
    if (y < 8) y = clientY + 16;
    t.style.left = `${x}px`;
    t.style.top = `${y}px`;
  }
  function hideTip() { if (tip) tip.style.display = 'none'; }

  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1;
    // remember the intended CSS height once: setting canvas.height below rewrites the height
    // attribute, so reading it again would compound the device-pixel ratio (zoom != 100 %)
    if (!canvas.dataset.cssHeight) canvas.dataset.cssHeight = canvas.getAttribute('height') || canvas.clientHeight || 200;
    const h = Number(canvas.dataset.cssHeight);
    const w = canvas.clientWidth || canvas.width;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.height = `${h}px`;
    }
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return { g, w, h };
  }

  function bindHover(canvas, onMove) {
    if (canvas.__hoverBound) { canvas.__onMove = onMove; return; }
    canvas.__hoverBound = true;
    canvas.__onMove = onMove;
    canvas.addEventListener('mousemove', (e) => {
      const r = canvas.getBoundingClientRect();
      canvas.__onMove && canvas.__onMove(e.clientX - r.left, e.clientY - r.top, e.clientX, e.clientY);
    });
    canvas.addEventListener('mouseleave', () => {
      hideTip();
      canvas.__hover = null;
      canvas.__redraw && canvas.__redraw();
    });
  }

  // ---------- line chart ----------
  // opts: { series: [{ name, color, values, dash }], x: [numbers], yMin, yMax, yTicks, yFormat, xFormat,
  //         xTicks, tooltip(i) -> html, endLabels, markers, padRight }
  function line(canvas, opts) {
    const draw = () => {
      const { g, w, h } = setup(canvas);
      const pad = { l: 44, r: opts.endLabels ? (opts.padRight || 92) : 14, t: 12, b: 26 };
      const pw = w - pad.l - pad.r;
      const ph = h - pad.t - pad.b;
      const n = opts.x.length;
      const yMin = opts.yMin ?? 0;
      const yMax = opts.yMax ?? 1;
      const xMin = opts.xMin ?? (n ? opts.x[0] : 0);
      const xMax = opts.xMax ?? (n ? opts.x[n - 1] : 1);
      const X = (v) => pad.l + (xMax === xMin ? pw : ((v - xMin) / (xMax - xMin)) * pw);
      const Y = (v) => pad.t + ph - ((v - yMin) / (yMax - yMin)) * ph;

      if (opts.shade) {
        g.fillStyle = '#fff4c2';
        g.fillRect(X(opts.shade[0]), pad.t, X(opts.shade[1]) - X(opts.shade[0]), ph);
      }
      g.font = FONT;
      g.textBaseline = 'middle';
      for (const t of opts.yTicks || [yMin, (yMin + yMax) / 2, yMax]) {
        g.strokeStyle = GRID;
        g.lineWidth = 1;
        g.beginPath(); g.moveTo(pad.l, Math.round(Y(t)) + 0.5); g.lineTo(pad.l + pw, Math.round(Y(t)) + 0.5); g.stroke();
        g.fillStyle = MUTED;
        g.textAlign = 'right';
        g.fillText((opts.yFormat || String)(t), pad.l - 8, Y(t));
      }
      g.strokeStyle = INK;
      g.beginPath(); g.moveTo(pad.l, pad.t + ph + 0.5); g.lineTo(pad.l + pw, pad.t + ph + 0.5); g.stroke();
      g.textAlign = 'center';
      g.textBaseline = 'top';
      g.fillStyle = MUTED;
      for (const t of opts.xTicks || (n ? [xMin, xMax] : [])) {
        const x = X(t);
        g.textAlign = x <= pad.l + 1 ? 'left' : (x >= pad.l + pw - 1 ? 'right' : 'center');
        g.fillText((opts.xFormat || String)(t), x, pad.t + ph + 7);
      }

      if (!n) {
        g.fillStyle = MUTED;
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.font = LABEL_FONT;
        g.fillText(opts.empty || 'No data yet', pad.l + pw / 2, pad.t + ph / 2);
        return;
      }
      for (const s of opts.series) {
        g.strokeStyle = s.color;
        g.lineWidth = 2;
        g.lineJoin = 'round';
        g.setLineDash(s.dash || []);
        g.beginPath();
        s.values.forEach((v, i) => (i ? g.lineTo(X(opts.x[i]), Y(v)) : g.moveTo(X(opts.x[i]), Y(v))));
        g.stroke();
        g.setLineDash([]);
        if (opts.markers) {
          s.values.forEach((v, i) => {
            g.fillStyle = SURFACE; g.beginPath(); g.arc(X(opts.x[i]), Y(v), 6, 0, 7); g.fill();
            g.fillStyle = s.color; g.beginPath(); g.arc(X(opts.x[i]), Y(v), 4, 0, 7); g.fill();
          });
        }
      }
      if (opts.endLabels) {
        const ends = opts.series.map((s) => ({ s, y: Y(s.values[s.values.length - 1]) })).sort((a, b) => a.y - b.y);
        for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);
        // keep the stack inside the plot: push colliding labels up from the bottom, never the free ones
        const bottom = pad.t + ph - 4;
        if (ends.length && ends[ends.length - 1].y > bottom) {
          ends[ends.length - 1].y = bottom;
          for (let i = ends.length - 2; i >= 0; i--) ends[i].y = Math.min(ends[i].y, ends[i + 1].y - 15);
        }
        g.font = LABEL_FONT;
        g.textAlign = 'left';
        g.textBaseline = 'middle';
        for (const e of ends) {
          g.fillStyle = e.s.color;
          g.fillRect(pad.l + pw + 8, e.y - 4, 8, 8);
          g.fillStyle = INK;
          g.fillText(e.s.name, pad.l + pw + 20, e.y);
        }
      }
      const hv = canvas.__hover;
      if (hv != null && hv < n) {
        const x = X(opts.x[hv]);
        g.strokeStyle = INK;
        g.lineWidth = 1;
        g.setLineDash([3, 3]);
        g.beginPath(); g.moveTo(x + 0.5, pad.t); g.lineTo(x + 0.5, pad.t + ph); g.stroke();
        g.setLineDash([]);
        for (const s of opts.series) {
          g.fillStyle = SURFACE; g.beginPath(); g.arc(x, Y(s.values[hv]), 6, 0, 7); g.fill();
          g.fillStyle = s.color; g.beginPath(); g.arc(x, Y(s.values[hv]), 4, 0, 7); g.fill();
        }
      }
      canvas.__geom = { X, pad, pw, n };
    };
    canvas.__redraw = draw;
    draw();
    bindHover(canvas, (mx, my, cx, cy) => {
      const geo = canvas.__geom;
      if (!geo || !geo.n || mx < geo.pad.l - 10 || mx > geo.pad.l + geo.pw + 10) { hideTip(); return; }
      let best = 0;
      let bestD = Infinity;
      opts.x.forEach((v, i) => { const d = Math.abs(geo.X(v) - mx); if (d < bestD) { bestD = d; best = i; } });
      canvas.__hover = best;
      draw();
      if (opts.tooltip) showTip(opts.tooltip(best), cx, cy);
    });
  }

  // ---------- horizontal bars (signed) ----------
  // opts: { items: [{ label, value, color }], max, format, tooltip(i) -> html }
  function bars(canvas, opts) {
    const draw = () => {
      const { g, w, h } = setup(canvas);
      const pad = { l: 110, r: 64, t: 6, b: 6 };
      const items = opts.items;
      const pw = w - pad.l - pad.r;
      const rowH = (h - pad.t - pad.b) / Math.max(1, items.length);
      const max = opts.max || Math.max(0.01, ...items.map((it) => Math.abs(it.value)));
      const hasNeg = items.some((it) => it.value < 0);
      const zero = pad.l + (hasNeg ? pw / 2 : 0);
      const scale = (hasNeg ? pw / 2 : pw) / max;
      g.font = FONT;
      g.textBaseline = 'middle';
      items.forEach((it, i) => {
        const y = pad.t + i * rowH;
        const bh = Math.min(22, rowH - 6);
        const by = y + (rowH - bh) / 2;
        if (canvas.__hover === i) { g.fillStyle = '#f4f2ec'; g.fillRect(0, y, w, rowH); }
        g.fillStyle = INK;
        g.textAlign = 'right';
        g.fillText(it.label, pad.l - 10, y + rowH / 2);
        const len = Math.abs(it.value) * scale;
        const x0 = it.value >= 0 ? zero : zero - len;
        g.fillStyle = it.color;
        g.fillRect(x0, by, Math.max(1, len), bh);
        g.fillStyle = INK;
        g.textAlign = it.value >= 0 ? 'left' : 'right';
        g.fillText((opts.format || String)(it.value), it.value >= 0 ? zero + len + 6 : zero - len - 6, y + rowH / 2);
      });
      g.strokeStyle = INK;
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(zero + 0.5, pad.t); g.lineTo(zero + 0.5, h - pad.b); g.stroke();
      canvas.__geom = { rowH, pad };
    };
    canvas.__redraw = draw;
    draw();
    bindHover(canvas, (mx, my, cx, cy) => {
      const geo = canvas.__geom;
      const i = Math.floor((my - geo.pad.t) / geo.rowH);
      if (i < 0 || i >= opts.items.length) { hideTip(); return; }
      canvas.__hover = i;
      draw();
      if (opts.tooltip) showTip(opts.tooltip(i), cx, cy);
    });
  }

  // ---------- heatmap (frames x bands, values 0..1, low frequencies at the bottom) ----------
  // sequential ramp, light -> dark (paper, safety yellow, amber, umber, ink)
  const RAMP = [[255, 251, 234], [255, 214, 64], [204, 138, 28], [92, 66, 34], [16, 20, 24]];
  function ramp(v) {
    const x = Math.max(0, Math.min(1, v)) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(x));
    const f = x - i;
    return RAMP[i].map((c, k) => Math.round(c + (RAMP[i + 1][k] - c) * f));
  }
  // opts: { frames, bands, data, highlight: [f0, f1], frameMs, bandHz: [...], tooltip }
  function heatmap(canvas, opts) {
    const draw = () => {
      const { g, w, h } = setup(canvas);
      const { frames, bands, data } = opts;
      if (!frames) {
        g.fillStyle = '#f3f1ea';
        g.fillRect(0, 0, w, h);
        g.fillStyle = MUTED;
        g.font = LABEL_FONT;
        g.textAlign = 'center';
        g.textBaseline = 'middle';
        g.fillText(opts.empty || 'Waiting for audio', w / 2, h / 2);
        return;
      }
      // contrast stretch between the 2nd and 99th percentile, so quiet clips stay readable
      const sorted = Float32Array.from(data).sort();
      const lo = sorted[Math.floor(sorted.length * 0.02)];
      const hi = Math.max(lo + 1e-3, sorted[Math.floor(sorted.length * 0.99)]);
      const img = g.createImageData(frames, bands);
      for (let f = 0; f < frames; f++) {
        for (let b = 0; b < bands; b++) {
          const [r, gg, bb] = ramp((data[f * bands + b] - lo) / (hi - lo));
          const o = ((bands - 1 - b) * frames + f) * 4;
          img.data[o] = r; img.data[o + 1] = gg; img.data[o + 2] = bb; img.data[o + 3] = 255;
        }
      }
      const off = document.createElement('canvas');
      off.width = frames;
      off.height = bands;
      off.getContext('2d').putImageData(img, 0, 0);
      g.imageSmoothingEnabled = false;
      g.drawImage(off, 0, 0, w, h);
      if (opts.highlight) {
        const [f0, f1] = opts.highlight;
        g.strokeStyle = '#1f6feb';
        g.lineWidth = 3;
        g.strokeRect((f0 / frames) * w + 1.5, 1.5, ((f1 - f0) / frames) * w - 3, h - 3);
      }
      canvas.__geom = { w, h };
    };
    canvas.__redraw = draw;
    draw();
    if (opts.tooltip) {
      bindHover(canvas, (mx, my, cx, cy) => {
        const geo = canvas.__geom;
        const f = Math.floor((mx / geo.w) * opts.frames);
        const b = opts.bands - 1 - Math.floor((my / geo.h) * opts.bands);
        if (f < 0 || f >= opts.frames || b < 0 || b >= opts.bands) { hideTip(); return; }
        showTip(opts.tooltip(f, b, opts.data[f * opts.bands + b]), cx, cy);
      });
    }
  }

  return { line, bars, heatmap, ramp, hideTip };
})();
