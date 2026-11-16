// Builds the project report (DOCX) in the Rashtriya Raksha University dissertation format
// ("Report Template.docx": A4, Times New Roman 12 pt, 1.5 line spacing, left margin 1.5", other
// margins 1"; front page, cover page, declaration, certificate, acknowledgements, contents, lists of
// figures and tables, abstract, numbered chapters, references, appendices).
//   node docs/report/build_report.js        -> docs/report/Pump_Sound_Anomaly_Detection_Report.docx
// Numbers come from report_data.json (make_figures.py). Missing screenshots become visible
// placeholders and missing numbers show as "—", so the report can be rebuilt at any stage.
// Chapter, section, figure and table numbers are assigned automatically; cross-references use
// labels resolved in a second pass (ref('fig:x')).
'use strict';
const fs = require('fs');
const path = require('path');
const docx = require('docx');
const {
  Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak,
} = docx;

const HERE = __dirname;
const ROOT = path.resolve(HERE, '..', '..');
const DATA = fs.existsSync(path.join(HERE, 'report_data.json'))
  ? JSON.parse(fs.readFileSync(path.join(HERE, 'report_data.json'), 'utf8')) : {};
const TEAM = fs.existsSync(path.join(HERE, 'team.json'))
  ? JSON.parse(fs.readFileSync(path.join(HERE, 'team.json'), 'utf8')) : {};
const OUT = process.env.REPORT_OUT || path.join(HERE, 'Pump_Sound_Anomaly_Detection_Report.docx');

// ---------- page format (RRU guidelines) ----------
const PAGE = { width: 11906, height: 16838 };                      // A4
const MARGIN = { left: 2160, right: 1440, top: 1440, bottom: 1440 }; // 1.5" left, 1" elsewhere
const TEXT_W = PAGE.width - MARGIN.left - MARGIN.right;              // 8306 DXA
const MAX_IMG_PX = Math.floor((TEXT_W / 1440) * 96);                 // 553 px at 96 dpi
const FONT = 'Times New Roman';
const FRONT_FONT = 'Garamond';
const MONO = 'Consolas';
const LINE = 360;                                                     // 1.5 line spacing
const HEADER_FILL = 'E7E6E6';

// ---------- data helpers ----------
const get = (obj, keyPath) => keyPath.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
// accepts fractions (0.80) and Edge Impulse's percentages (80.3)
const pct = (v, digits = 1) => (typeof v === 'number' && isFinite(v) ? `${(v > 1 ? v : v * 100).toFixed(digits)} %` : '—');
const fmt = (v, digits = 2, unit = '') => (typeof v === 'number' && isFinite(v) ? `${v.toFixed(digits)}${unit}` : '—');
const first = (...vals) => vals.find((v) => v !== undefined && v !== null);
const D = (keyPath) => get(DATA, keyPath);

// ---------- numbering context (two passes) ----------
const ctx = { chapter: '', sec: 0, sub: 0, fig: 0, tab: 0, labels: {}, prev: {} };
function resetPass() {
  ctx.prev = ctx.labels;
  Object.assign(ctx, { chapter: '', sec: 0, sub: 0, fig: 0, tab: 0, labels: {} });
}
const ref = (label) => ctx.prev[label] || '?';
const setLabel = (label, value) => { if (label) ctx.labels[label] = value; };

// ---------- text ----------
function runs(content, base = {}) {
  if (typeof content === 'string') return [new TextRun({ text: content, ...base })];
  return content.map((c) => (typeof c === 'string' ? new TextRun({ text: c, ...base }) : new TextRun({ ...base, ...c })));
}
// superscript citation, e.g. cite(2) or cite(2, 3)
const cite = (...n) => ({ text: n.join(','), superScript: true });
const p = (content, opts = {}) => new Paragraph({
  children: runs(content), alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: LINE }, ...opts,
});
const bullets = (items) => items.map((it) => new Paragraph({
  children: runs(it), numbering: { reference: 'bullets', level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 60, line: LINE },
}));
const numbered = (items, reference = 'steps') => items.map((it) => new Paragraph({
  children: runs(it), numbering: { reference, level: 0 }, alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 60, line: LINE },
}));
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const spacer = () => new Paragraph({ children: [], spacing: { after: 120 } });

// ---------- headings ----------
function chapter(num, title, label) {
  ctx.chapter = String(num);
  Object.assign(ctx, { sec: 0, sub: 0, fig: 0, tab: 0 });
  setLabel(label, `Chapter ${num}`);
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
    children: [new TextRun(`Chapter ${num}  ${title}`)] });
}
// unnumbered top-level heading in the body (References) or appendix ("Appendix A  Title")
function appendix(letter, title, label) {
  ctx.chapter = letter || '';
  Object.assign(ctx, { sec: 0, sub: 0, fig: 0, tab: 0 });
  setLabel(label, letter ? `Appendix ${letter}` : title);
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
    children: [new TextRun(letter ? `Appendix ${letter}  ${title}` : title)] });
}
function section(title, label) {
  ctx.sec += 1; ctx.sub = 0;
  const num = `${ctx.chapter}.${ctx.sec}`;
  setLabel(label, `Section ${num}`);
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(`${num}  ${title}`)] });
}
function subsection(title, label) {
  ctx.sub += 1;
  const num = `${ctx.chapter}.${ctx.sec}.${ctx.sub}`;
  setLabel(label, `Section ${num}`);
  return new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(`${num}  ${title}`)] });
}
// front-matter page titles (not numbered, not in the contents)
const frontTitle = (text) => new Paragraph({ style: 'FrontTitle', children: [new TextRun(text)] });

// ---------- captions (collected by the List of Figures / List of Tables fields) ----------
function figCaption(text, label) {
  ctx.fig += 1;
  const num = `${ctx.chapter}.${ctx.fig}`;
  setLabel(label, `Figure ${num}`);
  return new Paragraph({ style: 'FigureCaption', children: [new TextRun(`Figure ${num}: ${text}`)] });
}
function tabCaption(text, label) {
  ctx.tab += 1;
  const num = `${ctx.chapter}.${ctx.tab}`;
  setLabel(label, `Table ${num}`);
  return new Paragraph({ style: 'TableCaption', children: [new TextRun(`Table ${num}: ${text}`)] });
}

// ---------- code and tables ----------
function code(lines) {
  return lines.map((line, i) => new Paragraph({
    children: [new TextRun({ text: line.length ? line : ' ', font: MONO, size: 17 })],
    shading: { type: ShadingType.CLEAR, color: 'auto', fill: 'F2F2F2' },
    spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 160 : 0, line: 240 },
    indent: { left: 120, right: 120 },
  }));
}

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: '808080' };
const borders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

// widths are relative; they are scaled to the text width
function tableEl(headers, rows, widths) {
  const sum = widths.reduce((a, b) => a + b, 0);
  const w = widths.map((x) => Math.floor((x / sum) * TEXT_W));
  w[w.length - 1] += TEXT_W - w.reduce((a, b) => a + b, 0);
  const mkCell = (content, width, isHeader) => new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    shading: isHeader ? { type: ShadingType.CLEAR, color: 'auto', fill: HEADER_FILL } : undefined,
    margins: { top: 40, bottom: 40, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(Array.isArray(content) ? content : String(content), { bold: isHeader, size: 20 }), spacing: { after: 0, line: 260 } })],
  });
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => mkCell(h, w[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => mkCell(c, w[i], false)) })),
    ],
  });
}
// caption above the table, as is conventional
const table = (captionText, headers, rows, widths, label) =>
  [tabCaption(captionText, label), tableEl(headers, rows, widths), spacer()];

// ---------- figures ----------
function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}
function figure(file, captionText, label, maxWidthPx = MAX_IMG_PX, placeholderHint = '') {
  let body;
  if (fs.existsSync(file)) {
    const { w, h } = pngSize(file);
    const width = Math.min(maxWidthPx, MAX_IMG_PX);
    const height = Math.round((h / w) * width);
    body = new Paragraph({
      children: [new ImageRun({ type: 'png', data: fs.readFileSync(file), transformation: { width, height },
        altText: { title: captionText, description: captionText, name: path.basename(file) } })],
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    });
  } else {
    body = new Paragraph({
      children: [new TextRun({ text: `[Screenshot to add: docs/screenshots/${path.basename(file)}]`, bold: true, color: '9C5700' }),
        ...(placeholderHint ? [new TextRun({ text: ` — ${placeholderHint}`, color: '9C5700' })] : [])],
      shading: { type: ShadingType.CLEAR, color: 'auto', fill: 'FFF4E5' },
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    });
  }
  return [body, figCaption(captionText, label)];
}
const fig = (name, cap, label, w) => figure(path.join(HERE, 'figures', name), cap, label, w);
// Studio screenshots are optional: a missing file is simply left out of the report
const shot = (name, cap, label, hint) => {
  const file = path.join(ROOT, 'docs', 'screenshots', name);
  return fs.existsSync(file) ? figure(file, cap, label, MAX_IMG_PX, hint) : [];
};

module.exports = {
  D, get, pct, fmt, first, runs, cite, p, bullets, numbered, pageBreak, spacer, chapter, appendix, section, subsection,
  frontTitle, figCaption, tabCaption, code, table, fig, shot, ref, resetPass,
  TEXT_W, PAGE, MARGIN, LINE, OUT, TEAM, ROOT, HERE, FONT, FRONT_FONT, MONO, docx,
};

if (require.main === module) {
  require('./report_content.js').build().catch((e) => { console.error(e); process.exit(1); });
}
