// Builds the project report (DOCX) from report_data.json, figures/ and docs/screenshots/.
//   node docs/report/build_report.js        -> docs/report/Pump_Sound_Anomaly_Detection_Report.docx
// Missing screenshots/figures become visible placeholders and missing numbers show as "—",
// so the report can be rebuilt at any stage of the project.
'use strict';
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, Header, Footer, PageNumber,
  TableOfContents, LevelFormat,
} = require('docx');

const HERE = __dirname;
const ROOT = path.resolve(HERE, '..', '..');
const DATA = fs.existsSync(path.join(HERE, 'report_data.json'))
  ? JSON.parse(fs.readFileSync(path.join(HERE, 'report_data.json'), 'utf8')) : {};
const TEAM = fs.existsSync(path.join(HERE, 'team.json'))
  ? JSON.parse(fs.readFileSync(path.join(HERE, 'team.json'), 'utf8')) : {};
const OUT = path.join(HERE, 'Pump_Sound_Anomaly_Detection_Report.docx');

const FONT = 'Calibri';
const MONO = 'Consolas';
const TEXT_W = 9026; // A4 width 11906 - 2 x 1440 margins (DXA)
const ACCENT = '1F4E79';
const HEADER_FILL = 'DCE6F2';
const NOTE_FILL = 'FFF4E5';

// ---------- data helpers ----------
const get = (obj, keyPath) => keyPath.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
// accepts fractions (0.80) and Edge Impulse's percentages (80.3)
const pct = (v, digits = 1) => (typeof v === 'number' ? `${(v > 1 ? v : v * 100).toFixed(digits)} %` : '—');
const fmt = (v, digits = 2, unit = '') => (typeof v === 'number' ? `${v.toFixed(digits)}${unit}` : '—');
const first = (...vals) => vals.find((v) => v !== undefined && v !== null);
const D = (keyPath) => get(DATA, keyPath);

// ---------- paragraph helpers ----------
function runs(content, base = {}) {
  if (typeof content === 'string') return [new TextRun({ text: content, ...base })];
  return content.map((c) => (typeof c === 'string' ? new TextRun({ text: c, ...base }) : new TextRun({ ...base, ...c })));
}
const p = (content, opts = {}) => new Paragraph({ children: runs(content), spacing: { after: 120, line: 276 }, ...opts });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const h3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
const bullets = (items) => items.map((it) => new Paragraph({
  children: runs(it), numbering: { reference: 'bullets', level: 0 }, spacing: { after: 60, line: 276 },
}));
const numbered = (items, reference = 'steps') => items.map((it) => new Paragraph({
  children: runs(it), numbering: { reference, level: 0 }, spacing: { after: 60, line: 276 },
}));
const caption = (text) => new Paragraph({
  children: [new TextRun({ text, italics: true, size: 18, color: '595959' })],
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 },
});
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function code(lines) {
  return lines.map((line, i) => new Paragraph({
    children: [new TextRun({ text: line.length ? line : ' ', font: MONO, size: 17 })],
    shading: { type: ShadingType.CLEAR, color: 'auto', fill: 'F2F2F2' },
    spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 160 : 0, line: 240 },
    indent: { left: 120, right: 120 },
  }));
}

function note(content) {
  return new Paragraph({
    children: runs(content, { size: 20 }),
    shading: { type: ShadingType.CLEAR, color: 'auto', fill: NOTE_FILL },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: 'E8A33D', space: 6 } },
    spacing: { before: 80, after: 160, line: 264 }, indent: { left: 160, right: 120 },
  });
}

// ---------- tables ----------
const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
const borders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const mkCell = (content, w, isHeader) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders,
    shading: isHeader ? { type: ShadingType.CLEAR, color: 'auto', fill: HEADER_FILL } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(String(content), { bold: isHeader, size: 19 }), spacing: { after: 0 } })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => mkCell(h, widths[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => mkCell(c, widths[i], false)) })),
    ],
  });
}
const spacer = () => new Paragraph({ children: [], spacing: { after: 120 } });

// ---------- figures ----------
function pngSize(file) {
  const buf = fs.readFileSync(file);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function figure(file, captionText, maxWidthPx = 600, placeholderHint = '') {
  if (fs.existsSync(file)) {
    const { w, h } = pngSize(file);
    const width = Math.min(maxWidthPx, 600);
    const height = Math.round((h / w) * width);
    return [
      new Paragraph({
        children: [new ImageRun({ type: 'png', data: fs.readFileSync(file), transformation: { width, height },
          altText: { title: captionText, description: captionText, name: path.basename(file) } })],
        alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      }),
      caption(captionText),
    ];
  }
  return [
    new Paragraph({
      children: [new TextRun({ text: `[Screenshot to add: docs/screenshots/${path.basename(file)}]`, bold: true, color: '9C5700' }),
        ...(placeholderHint ? [new TextRun({ text: ` — ${placeholderHint}`, color: '9C5700' })] : [])],
      shading: { type: ShadingType.CLEAR, color: 'auto', fill: NOTE_FILL },
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
    }),
    caption(captionText),
  ];
}
const fig = (name, cap, w) => figure(path.join(HERE, 'figures', name), cap, w);
const shot = (name, cap, hint) => figure(path.join(ROOT, 'docs', 'screenshots', name), cap, 600, hint);

module.exports = {
  D, get, pct, fmt, first, p, h1, h2, h3, bullets, numbered, caption, pageBreak, code, note, table, spacer,
  fig, shot, runs, TEXT_W, OUT, TEAM, ROOT, HERE, FONT, MONO, ACCENT,
  docx: { Document, Packer, Paragraph, TextRun, AlignmentType, Header, Footer, PageNumber, TableOfContents, LevelFormat, HeadingLevel },
};

if (require.main === module) {
  require('./report_content.js').build().catch((e) => { console.error(e); process.exit(1); });
}
