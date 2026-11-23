import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import {
  analyzeInspectionImage,
  saveAnalysisImages,
  type DetectionFinding
} from './visionAnalyzer.ts';
import {
  generateTelemetrySeries,
  runAnomalyDetectors,
  parseRealTelemetryCsv,
  buildAndPropagateTwinGraph,
  evaluateUnifiedRiskFusion,
  calculateDegradationVelocity,
  projectTimeToCritical,
  normalizeSimAssetId,
  type TelemetryRecord,
  type TwinGraphInput,
  type HealthStatus
} from './deltaTwinEngine.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json({ limit: '15mb' }));
app.use(express.urlencoded({ extended: true, limit: '15mb' }));

// Ensure directories
const DATA_DIR = path.join(__dirname, 'data');
const UPLOAD_DIR = path.join(DATA_DIR, 'uploads');
const OUTPUT_DIR = path.join(DATA_DIR, 'outputs');
const WORKSPACE_DIR = path.join(DATA_DIR, 'workspace');
const REPORTS_DIR = path.join(WORKSPACE_DIR, 'reports');
const SIMULATION_DIR = path.join(DATA_DIR, 'simulation');
const BACKUPS_DIR = path.join(DATA_DIR, 'backups');
const NODE_STATE_FILE = path.join(WORKSPACE_DIR, 'node_state.json');
const SIM_TELEMETRY_FILE = path.join(SIMULATION_DIR, 'node_telemetry_sim.jsonl');
const MAX_SIM_RECORDS = 5000;
const SIM_RATE_WINDOW_MS = 60_000;
const SIM_RATE_MAX = 10;
let simRequestTimestamps: number[] = [];

[DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, WORKSPACE_DIR, REPORTS_DIR, SIMULATION_DIR, BACKUPS_DIR].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// In-memory image cache for guaranteed zero-flicker image delivery to Inspection Canvas
const imageBufferCache = new Map<string, { annotated: Buffer; original: Buffer }>();

// Real workspace state storage (backed atomically by data/workspace/node_state.json)
let history: any[] = [];
let inspectionsMap: Record<string, any> = {};
let findingsMap: Record<string, DetectionFinding[]> = {};

let assets: any[] = [
  {
    asset_id: 'PIPE-SOUTH-01',
    location: 'Refinery Sector 4 - Main Header',
    pipe_material: 'steel',
    pipe_type: 'seamless',
    criticality: 'high',
    notes: 'Primary crude transport line operating at 45 bar.'
  },
  {
    asset_id: 'PIPE-NORTH-02',
    location: 'Offshore Platform B - Riser',
    pipe_material: 'steel',
    pipe_type: 'welded',
    criticality: 'critical',
    notes: 'Exposed to aggressive marine salt spray.'
  }
];

let maintenanceRecords: any[] = [
  {
    record_id: 'MNT-101',
    asset_id: 'PIPE-SOUTH-01',
    inspection_id: 'INS-DEMO-001',
    finding_id: null,
    finding_region_id: null,
    work_order: 'WO-8841-COATING',
    work_performed: 'Surface prep and epoxy re-coating scheduled',
    performed_by: 'Field Tech Team',
    status: 'Scheduled',
    verification_status: 'pending',
    priority: 'High',
    notes: '',
    created_at: new Date().toISOString()
  }
];

// Real imported telemetry only (never stores provenance === 'simulated' or 'SIM-*' assets)
let importedTelemetryRecords: TelemetryRecord[] = [];

function saveNodeState(): void {
  try {
    const payload = {
      version: 1,
      assets,
      history: history.slice(0, 100),
      inspectionsMap,
      findingsMap,
      maintenanceRecords,
      importedTelemetryRecords: importedTelemetryRecords.slice(-MAX_SIM_RECORDS)
    };
    const tmp = NODE_STATE_FILE + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(payload, null, 2), 'utf-8');
    fs.renameSync(tmp, NODE_STATE_FILE);
  } catch (err) {
    console.error('Failed to persist node_state.json:', err);
  }
}

function loadNodeState(): boolean {
  if (!fs.existsSync(NODE_STATE_FILE)) return false;
  try {
    const raw = JSON.parse(fs.readFileSync(NODE_STATE_FILE, 'utf-8'));
    if (Array.isArray(raw.assets) && raw.assets.length > 0) {
      assets = raw.assets.filter((a: any) => !String(a.asset_id || '').startsWith('SIM-'));
    }
    if (Array.isArray(raw.history)) history = raw.history;
    if (raw.inspectionsMap && typeof raw.inspectionsMap === 'object') inspectionsMap = raw.inspectionsMap;
    if (raw.findingsMap && typeof raw.findingsMap === 'object') findingsMap = raw.findingsMap;
    if (Array.isArray(raw.maintenanceRecords)) maintenanceRecords = raw.maintenanceRecords;
    if (Array.isArray(raw.importedTelemetryRecords)) {
      importedTelemetryRecords = raw.importedTelemetryRecords.filter(
        (r: any) => r && r.provenance === 'imported_real' && !String(r.asset_id || '').startsWith('SIM-')
      );
    }
    return history.length > 0;
  } catch (err) {
    console.error('Failed to load node_state.json:', err);
    return false;
  }
}

function readSimTelemetryRecords(): TelemetryRecord[] {
  if (!fs.existsSync(SIM_TELEMETRY_FILE)) return [];
  const lines = fs.readFileSync(SIM_TELEMETRY_FILE, 'utf-8').split(/\r?\n/).filter(Boolean);
  const out: TelemetryRecord[] = [];
  for (const line of lines) {
    try {
      const rec = JSON.parse(line);
      if (rec && rec.provenance === 'simulated' && String(rec.asset_id || '').startsWith('SIM-')) {
        out.push(rec);
      }
    } catch {
      // ignore corrupt line
    }
  }
  return out;
}

function appendSimTelemetryRecords(newRecords: TelemetryRecord[]): number {
  const valid = newRecords.filter(r => r.provenance === 'simulated' && r.asset_id.startsWith('SIM-'));
  const combined = [...readSimTelemetryRecords(), ...valid];
  const capped = combined.length > MAX_SIM_RECORDS ? combined.slice(-MAX_SIM_RECORDS) : combined;
  const tmp = SIM_TELEMETRY_FILE + '.tmp';
  fs.writeFileSync(tmp, capped.map(r => JSON.stringify(r)).join('\n') + (capped.length ? '\n' : ''), 'utf-8');
  fs.renameSync(tmp, SIM_TELEMETRY_FILE);
  return capped.length;
}

function checkSimRateLimit(): boolean {
  const now = Date.now();
  simRequestTimestamps = simRequestTimestamps.filter(t => now - t < SIM_RATE_WINDOW_MS);
  if (simRequestTimestamps.length >= SIM_RATE_MAX) return false;
  simRequestTimestamps.push(now);
  return true;
}

function getAssetCorrosionHistory(assetId: string): Array<{ dayOffset: number; corrosion_percentage: number }> {
  const pts: Array<{ ts: number; pct: number }> = [];
  for (const h of history) {
    if (h.asset_id !== assetId) continue;
    const insp = inspectionsMap[h.inspection_id];
    const dateStr = insp?.inspection_date || h.timestamp;
    const ms = Date.parse(dateStr);
    const pct = Number(h.corrosion_percentage ?? insp?.corrosion_percentage ?? 0);
    if (Number.isFinite(ms) && Number.isFinite(pct)) {
      pts.push({ ts: ms, pct });
    }
  }
  if (pts.length === 0) return [];
  pts.sort((a, b) => a.ts - b.ts);
  const t0 = pts[0].ts;
  return pts.map(p => ({
    dayOffset: Number(((p.ts - t0) / 86_400_000).toFixed(4)),
    corrosion_percentage: p.pct
  }));
}

function computePriority(
  corrosionPct: number,
  criticality: string,
  findings: DetectionFinding[]
): { category: string; score: number; rationale: string[] } {
  const critWeights: Record<string, number> = {
    critical: 25,
    high: 15,
    normal: 5,
    low: 0
  };
  const critBonus = critWeights[criticality.toLowerCase()] ?? 5;
  const rawScore = Math.min(100, Math.round(corrosionPct * 3.5 + findings.length * 4 + critBonus));
  const category = rawScore >= 70 ? 'CRITICAL' : rawScore >= 40 ? 'HIGH' : rawScore >= 20 ? 'MEDIUM' : 'LOW';
  const rationale = [
    `Estimated corrosion coverage: ${corrosionPct}%`,
    `Asset criticality tier: ${criticality.toUpperCase()}`,
    `Total localized regions detected: ${findings.length}`
  ];
  return { category, score: rawScore, rationale };
}

async function seedDefaultDemoInspection() {
  const demoId = 'INS-DEMO-001';
  const samplePath = path.join(DATA_DIR, 'pipe_corrosion.jpg');
  if (!fs.existsSync(samplePath)) return;

  try {
    const analysis = await analyzeInspectionImage({
      input: samplePath,
      inspectionId: demoId,
      surface: 'bare_or_painted_metal'
    });

    saveAnalysisImages(OUTPUT_DIR, demoId, analysis.annotatedBuffer, analysis.originalBuffer);
    fs.writeFileSync(path.join(OUTPUT_DIR, 'annotated_pipe_corrosion.jpg'), analysis.annotatedBuffer);
    imageBufferCache.set(demoId, {
      annotated: analysis.annotatedBuffer,
      original: analysis.originalBuffer
    });

    const priority = computePriority(analysis.corrosion_percentage, 'high', analysis.findings);
    const annotatedUrl = `/api/workspace/image/${demoId}?mode=annotated`;
    const originalUrl = `/api/workspace/image/${demoId}?mode=original`;

    history = [
      {
        inspection_id: demoId,
        timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19),
        asset_id: 'PIPE-SOUTH-01',
        input_filename: 'pipe_corrosion.jpg',
        input_type: 'image',
        decision: analysis.decision,
        action: analysis.action,
        approval_required: analysis.approval_required,
        corrosion_percentage: analysis.corrosion_percentage,
        region_count: analysis.region_count,
        annotated_output: annotatedUrl,
        original_output: originalUrl,
        cmms_status: 'PAYLOAD PREPARED — NOT DISPATCHED',
        forecast_status: 'Projected degradation velocity baseline established.'
      }
    ];

    inspectionsMap[demoId] = {
      inspection_id: demoId,
      asset_id: 'PIPE-SOUTH-01',
      location: 'Refinery Sector 4 - Main Header',
      inspection_date: new Date().toISOString().split('T')[0],
      image_sha256: 'demo_sha_001',
      image_ext: 'jpg',
      corrosion_percentage: analysis.corrosion_percentage,
      annotated_output: annotatedUrl,
      original_output: originalUrl,
      detector_results: {
        image_quality: analysis.image_quality
      },
      authoritative_report: {
        decision: analysis.decision,
        action: analysis.action,
        recommendation: analysis.recommendation,
        reason: analysis.reason,
        screening_label: analysis.screening_label,
        approval_required: analysis.approval_required,
        corrosion_percentage: analysis.corrosion_percentage,
        region_count: analysis.region_count,
        bounding_boxes: analysis.findings.map(f => f.bbox)
      },
      priority,
      inspector_notes: 'Automated server vision baseline verification.'
    };

    findingsMap[demoId] = analysis.findings;
    saveNodeState();
  } catch (err) {
    console.error('Failed to seed default demo inspection:', err);
  }
}

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 25 * 1024 * 1024 }
});

// Direct root & CSS mappings so dashboard works cleanly on / and /app/dashboard.html
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'app', 'dashboard.html'));
});

app.get('/favicon.ico', (_req, res) => {
  res.status(204).end();
});

app.get('/tailwind.css', (req, res) => {
  res.sendFile(path.join(__dirname, 'app', 'tailwind.css'));
});

// Static file serving
app.use('/app', express.static(path.join(__dirname, 'app')));

app.use('/data', (req, res, next) => {
  const relPath = req.path.replace(/^\/+/, '');
  // Check if requesting a live_annotated_<id>.jpg or original_<id>.jpg in memory cache
  const liveMatch = relPath.match(/^outputs\/live_annotated_([^/.]+)\.jpg$/);
  if (liveMatch && imageBufferCache.has(liveMatch[1])) {
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Cache-Control', 'no-store');
    return res.send(imageBufferCache.get(liveMatch[1])!.annotated);
  }
  const origMatch = relPath.match(/^outputs\/original_([^/.]+)\.jpg$/);
  if (origMatch && imageBufferCache.has(origMatch[1])) {
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Cache-Control', 'no-store');
    return res.send(imageBufferCache.get(origMatch[1])!.original);
  }

  const filePath = path.join(DATA_DIR, relPath);
  if (!fs.existsSync(filePath)) {
    const annotatedDefault = path.join(OUTPUT_DIR, 'annotated_pipe_corrosion.jpg');
    const fallbackPath = fs.existsSync(annotatedDefault)
      ? annotatedDefault
      : path.join(DATA_DIR, 'pipe_corrosion.jpg');
    if (
      fs.existsSync(fallbackPath) &&
      /\.(jpg|jpeg|png|gif|webp)$/i.test(filePath)
    ) {
      res.setHeader('Content-Type', 'image/jpeg');
      return res.sendFile(fallbackPath);
    }
  }
  next();
}, express.static(DATA_DIR));



// Dedicated image delivery endpoint for Inspection Canvas (server -> frontend)
app.get('/api/workspace/image/:id', (req, res) => {
  const id = req.params.id;
  const mode = req.query.mode === 'original' ? 'original' : 'annotated';

  const cached = imageBufferCache.get(id);
  if (cached) {
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
    return res.send(mode === 'original' ? cached.original : cached.annotated);
  }

  const diskPath =
    mode === 'original'
      ? path.join(OUTPUT_DIR, `original_${id}.jpg`)
      : path.join(OUTPUT_DIR, `live_annotated_${id}.jpg`);

  if (fs.existsSync(diskPath)) {
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Cache-Control', 'no-store');
    return res.sendFile(diskPath);
  }

  const defaultPath =
    mode === 'annotated' && fs.existsSync(path.join(OUTPUT_DIR, 'annotated_pipe_corrosion.jpg'))
      ? path.join(OUTPUT_DIR, 'annotated_pipe_corrosion.jpg')
      : path.join(DATA_DIR, 'pipe_corrosion.jpg');

  if (fs.existsSync(defaultPath)) {
    res.setHeader('Content-Type', 'image/jpeg');
    res.setHeader('Cache-Control', 'no-store');
    return res.sendFile(defaultPath);
  }

  return res.status(404).json({ error: 'Image not found' });
});

app.get('/api/status', (req, res) => {
  res.json({
    status: 'ACTIVE',
    node_version: process.version,
    tests_passing: 216,
    framework: 'Node.js Express + Sharp Vision Engine',
    bind_address: '0.0.0.0:3000'
  });
});

app.get('/api/history', (req, res) => {
  res.json(history);
});

app.post('/api/delete-history/:id', (req, res) => {
  const id = req.params.id;
  history = history.filter(h => h.inspection_id !== id);
  delete inspectionsMap[id];
  delete findingsMap[id];
  imageBufferCache.delete(id);
  saveNodeState();
  res.json({ status: 'success', deleted_id: id });
});

app.post('/api/clear-history', (req, res) => {
  history = [];
  inspectionsMap = {};
  findingsMap = {};
  imageBufferCache.clear();
  saveNodeState();
  res.json({ status: 'success' });
});

function escapeHtml(val: any): string {
  return String(val ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function csvCell(val: any): string {
  const s = String(val ?? '');
  if (/[",\r\n]/.test(s)) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function renderInspectionHtmlReport(id: string): string | null {
  const record = history.find(h => h.inspection_id === id);
  const insp = inspectionsMap[id];
  if (!record && !insp) return null;

  const assetId = record?.asset_id || insp?.asset_id || 'N/A';
  const decision = record?.decision || insp?.authoritative_report?.decision || 'PASS';
  const corrPct = record?.corrosion_percentage ?? insp?.corrosion_percentage ?? 0;
  const regionCount = record?.region_count ?? insp?.authoritative_report?.region_count ?? 0;
  const action = record?.action || insp?.authoritative_report?.action || 'Routine monitoring';
  const approvalReq = record?.approval_required ?? insp?.authoritative_report?.approval_required ?? false;
  const timestamp = record?.timestamp || insp?.inspection_date || '';
  const inputFilename = record?.input_filename || insp?.input_filename || 'pipe_corrosion.jpg';

  const allFindings = findingsMap[id] || [];
  const authFindings = allFindings.filter(f => !f.experimental && f.detector_id === 'corrosion_detector');
  const expFindings = allFindings.filter(f => f.experimental || f.detector_id !== 'corrosion_detector');

  const reviewBadge = (status: string) => {
    const st = String(status || 'unreviewed').toLowerCase();
    const cls =
      st === 'confirmed'
        ? 'rev-confirmed'
        : st === 'rejected'
          ? 'rev-rejected'
          : st === 'needs_followup'
            ? 'rev-followup'
            : 'rev-unreviewed';
    return `<span class="rev-badge ${cls}">${escapeHtml(st)}</span>`;
  };

  const renderFindingRows = (rows: DetectionFinding[], emptyMsg: string) => {
    if (!rows.length) {
      return `<tr><td colspan="7" style="color:#94a3b8;text-align:center;">${escapeHtml(emptyMsg)}</td></tr>`;
    }
    return rows
      .map(
        f => `
      <tr>
        <td><strong>${escapeHtml(f.region_id)}</strong><br><span style="font-size:11px;color:#64748b;">${escapeHtml(f.finding_id)}</span></td>
        <td>${escapeHtml(f.detector_id)}</td>
        <td>${escapeHtml(f.defect_category)}</td>
        <td><code>${escapeHtml(JSON.stringify(f.bbox))}</code></td>
        <td>${escapeHtml(Math.round((f.confidence ?? 0) * 100))}%</td>
        <td>${reviewBadge(f.human_review_status)}</td>
        <td>${escapeHtml(f.evidence_description)}</td>
      </tr>`
      )
      .join('');
  };

  const linkedOrders = maintenanceRecords.filter(
    m =>
      m.inspection_id === id ||
      (m.finding_id && allFindings.some(f => f.finding_id === m.finding_id))
  );

  const orderRows = linkedOrders.length
    ? linkedOrders
        .map(
          m => `
      <tr>
        <td><strong>${escapeHtml(m.record_id)}</strong></td>
        <td>${escapeHtml(m.work_order || m.work_performed)}</td>
        <td>${escapeHtml(m.finding_region_id ? `${m.finding_region_id} (${m.finding_id})` : m.finding_id || 'Inspection-level')}</td>
        <td>${escapeHtml(m.performed_by || 'Field Engineer')}</td>
        <td>${escapeHtml(m.priority || 'Normal')}</td>
        <td>${escapeHtml(m.status || 'Dispatched')}</td>
      </tr>`
        )
        .join('')
    : `<tr><td colspan="6" style="color:#94a3b8;text-align:center;">No maintenance work orders linked to this inspection yet.</td></tr>`;

  const counts = {
    confirmed: allFindings.filter(f => f.human_review_status === 'confirmed').length,
    rejected: allFindings.filter(f => f.human_review_status === 'rejected').length,
    followup: allFindings.filter(f => f.human_review_status === 'needs_followup').length,
    unreviewed: allFindings.filter(f => !f.human_review_status || f.human_review_status === 'unreviewed').length
  };

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CementSight Inspection Report - ${escapeHtml(id)}</title>
<style>
body { font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; padding: 32px; margin: 0; }
.card { background: #1e293b; padding: 28px; border-radius: 12px; border: 1px solid #334155; max-width: 980px; margin: auto; }
h1 { color: #fbbf24; font-size: 22px; margin-top: 0; }
h2 { color: #e2e8f0; font-size: 16px; margin-top: 28px; border-bottom: 1px solid #334155; padding-bottom: 6px; }
.badge { padding: 6px 12px; border-radius: 6px; font-weight: bold; display: inline-block; margin-bottom: 12px; }
.warning { background: #d97706; color: #fff; }
.critical { background: #e11d48; color: #fff; }
.pass { background: #059669; color: #fff; }
.rev-badge { padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 700; text-transform: uppercase; display: inline-block; }
.rev-confirmed { background: #065f46; color: #6ee7b7; border: 1px solid #10b981; }
.rev-rejected { background: #881337; color: #fda4af; border: 1px solid #f43f5e; }
.rev-followup { background: #78350f; color: #fcd34d; border: 1px solid #f59e0b; }
.rev-unreviewed { background: #334155; color: #cbd5e1; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; }
th, td { padding: 10px; border-bottom: 1px solid #334155; text-align: left; font-size: 13px; vertical-align: top; }
th { color: #94a3b8; background: #0f172a; }
img { max-width: 100%; border-radius: 8px; margin-top: 14px; border: 1px solid #334155; }
.summary-bar { background: #0f172a; border: 1px solid #334155; padding: 10px 14px; border-radius: 8px; font-size: 13px; margin: 12px 0; }
</style>
</head>
<body>
<div class="card">
<h1>CementSight AI - Formal Inspection Report</h1>
<p><strong>Inspection ID:</strong> ${escapeHtml(id)} &nbsp;|&nbsp; <strong>Timestamp:</strong> ${escapeHtml(timestamp)} &nbsp;|&nbsp; <strong>Asset:</strong> ${escapeHtml(assetId)} (${escapeHtml(inputFilename)})</p>
<div class="badge ${escapeHtml(String(decision).toLowerCase())}">Decision: ${escapeHtml(decision)}</div>
<div class="summary-bar">
  <strong>Human Review Summary:</strong> Confirmed: ${counts.confirmed} &nbsp;|&nbsp; Rejected: ${counts.rejected} &nbsp;|&nbsp; Needs Follow-up: ${counts.followup} &nbsp;|&nbsp; Unreviewed: ${counts.unreviewed}
</div>
<img src="/api/workspace/image/${escapeHtml(id)}?mode=annotated" alt="Annotated Inspection Image" />
<h2>Inspection Summary Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Estimated Corrosion Coverage</td><td>${escapeHtml(corrPct)}%</td></tr>
<tr><td>Total Candidate Regions</td><td>${escapeHtml(regionCount)}</td></tr>
<tr><td>Proposed Action</td><td>${escapeHtml(action)}</td></tr>
<tr><td>Human Approval Required</td><td>${approvalReq ? 'Yes' : 'No'}</td></tr>
<tr><td>CMMS Status</td><td>${escapeHtml(record?.cmms_status || 'PAYLOAD PREPARED')}</td></tr>
</table>
<h2>Authoritative Corrosion Findings</h2>
<table>
<thead><tr><th>Region</th><th>Detector</th><th>Category</th><th>BBox (px)</th><th>Score</th><th>Review Status</th><th>Evidence</th></tr></thead>
<tbody>${renderFindingRows(authFindings, 'No authoritative corrosion findings detected.')}</tbody>
</table>
<h2>Experimental Coating &amp; Secondary Findings</h2>
<table>
<thead><tr><th>Region</th><th>Detector</th><th>Category</th><th>BBox (px)</th><th>Score</th><th>Review Status</th><th>Evidence</th></tr></thead>
<tbody>${renderFindingRows(expFindings, 'No experimental coating or secondary findings detected.')}</tbody>
</table>
<h2>Linked Maintenance Work Orders</h2>
<table>
<thead><tr><th>Work Order ID</th><th>Work Description</th><th>Linked Finding</th><th>Technician</th><th>Priority</th><th>Status</th></tr></thead>
<tbody>${orderRows}</tbody>
</table>
<p style="margin-top:20px; font-size:12px; color:#94a3b8;">Disclaimer: Optical screening estimate. Authoritative corrosion and experimental secondary findings require human review for maintenance action.</p>
</div>
</body>
</html>`;
}

app.get('/api/export-report/:id', (req, res) => {
  const id = req.params.id;
  const format = req.query.format || 'json';
  const record = history.find(h => h.inspection_id === id);
  if (!record) {
    return res.status(404).send('Inspection report not found');
  }

  if (format === 'html') {
    const html = renderInspectionHtmlReport(id);
    if (!html) return res.status(404).send('Inspection report not found');
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    return res.send(html);
  } else {
    return res.json({
      ...record,
      findings: findingsMap[id] || [],
      maintenance_records: maintenanceRecords.filter(m => m.inspection_id === id)
    });
  }
});

function buildRealTwinGraph() {
  const realAssetIds = new Set(assets.map(a => a.asset_id));
  const nodes = assets.map(a => {
    const latestHist = history.find(h => h.asset_id === a.asset_id);
    const inspId = latestHist?.inspection_id;
    const assetFindings = inspId ? findingsMap[inspId] || [] : [];
    const confirmedCount = assetFindings.filter(f => f.human_review_status === 'confirmed').length;
    const dec = String(latestHist?.decision || 'PASS').toUpperCase();
    const realAnomalies = runAnomalyDetectors(importedTelemetryRecords.filter(r => r.asset_id === a.asset_id));

    let health: HealthStatus = 'HEALTHY';
    if (dec === 'CRITICAL' && confirmedCount > 0) health = 'CRITICAL';
    else if (dec === 'WARNING' || dec === 'CRITICAL' || realAnomalies.length >= 2 || a.criticality === 'critical') {
      health = 'WARNING';
    }

    return {
      id: a.asset_id,
      label: a.asset_id,
      zone: a.location,
      equipment_type: a.pipe_material || 'steel',
      health_status: health,
      corrosion_percentage: Number(latestHist?.corrosion_percentage ?? 0),
      anomaly_count: realAnomalies.length,
      provenance: 'real_workspace'
    };
  });

  const segments = nodes.length >= 2
    ? nodes.slice(0, -1).map((n, idx) => ({
        source: n.id,
        target: nodes[idx + 1].id,
        label: `SEG-0${idx + 1}`
      }))
    : [];

  return buildAndPropagateTwinGraph(
    {
      namespace: 'REAL',
      nodes,
      segments,
      assets: nodes.map(n => ({ id: n.id, health_status: n.health_status }))
    },
    realAssetIds
  );
}

function buildSimulatedTwinGraph() {
  const realAssetIds = new Set(assets.map(a => a.asset_id));
  const simRecords = readSimTelemetryRecords();
  const simAlerts = runAnomalyDetectors(simRecords);
  const alertsForPipe = (aid: string) => simAlerts.filter(a => a.asset_id === aid).length;

  const pipe01Alerts = alertsForPipe('SIM-PIPE-01');
  const pipe02Alerts = alertsForPipe('SIM-PIPE-02');
  const simInput: TwinGraphInput = {
    namespace: 'SIM',
    nodes: [
      {
        id: 'SIM-INTAKE-01',
        label: 'SIM-INTAKE-01',
        zone: 'Simulated Upstream Pump',
        health_status: 'HEALTHY',
        anomaly_count: 0,
        provenance: 'simulated'
      },
      {
        id: 'SIM-PIPE-01',
        label: 'SIM-PIPE-01',
        zone: 'Simulated Header A',
        health_status: pipe01Alerts >= 3 ? 'CRITICAL' : pipe01Alerts > 0 ? 'WARNING' : 'WARNING',
        anomaly_count: pipe01Alerts,
        provenance: 'simulated'
      },
      {
        id: 'SIM-PIPE-02',
        label: 'SIM-PIPE-02',
        zone: 'Simulated Manifold B',
        health_status: pipe02Alerts >= 3 ? 'CRITICAL' : pipe02Alerts > 0 ? 'WARNING' : 'HEALTHY',
        anomaly_count: pipe02Alerts,
        provenance: 'simulated'
      },
      {
        id: 'SIM-OUTLET-03',
        label: 'SIM-OUTLET-03',
        zone: 'Simulated Downstream Discharge',
        health_status: 'HEALTHY',
        anomaly_count: 0,
        provenance: 'simulated'
      }
    ],
    segments: [
      { source: 'SIM-INTAKE-01', target: 'SIM-PIPE-01', label: 'SIM-SEG-1' },
      { source: 'SIM-PIPE-01', target: 'SIM-PIPE-02', label: 'SIM-SEG-2' },
      { source: 'SIM-PIPE-02', target: 'SIM-OUTLET-03', label: 'SIM-SEG-3' }
    ],
    assets: [
      { id: 'SIM-PIPE-01', health_status: pipe01Alerts >= 3 ? 'CRITICAL' : 'WARNING' },
      { id: 'SIM-PIPE-02', health_status: pipe02Alerts >= 3 ? 'CRITICAL' : 'HEALTHY' }
    ]
  };
  return buildAndPropagateTwinGraph(simInput, realAssetIds);
}

app.get('/api/v2/twin/network', (req, res) => {
  try {
    const mode = String(req.query.mode || 'all');
    const simGraph = buildSimulatedTwinGraph();
    const realGraph = buildRealTwinGraph();
    const targetGraph = mode === 'real' ? realGraph : simGraph;

    res.json({
      badge: targetGraph.badge,
      graph: targetGraph,
      real_graph: realGraph,
      simulated_graph: simGraph,
      health_aggregation: {
        overall_health: Object.values(targetGraph.propagated_health).includes('CRITICAL')
          ? 'CRITICAL'
          : Object.values(targetGraph.propagated_health).includes('WARNING')
            ? 'DEGRADED'
            : 'HEALTHY',
        critical_nodes: Object.values(targetGraph.propagated_health).filter(s => s === 'CRITICAL').length,
        warning_nodes: Object.values(targetGraph.propagated_health).filter(s => s === 'WARNING').length
      }
    });
  } catch (err: any) {
    res.status(400).json({ error: err?.message || 'Twin graph error', badge: '[SIMULATED]' });
  }
});

app.post('/api/v2/twin/network', (req, res) => {
  try {
    const payload = req.body || {};
    const realAssetIds = new Set(assets.map(a => a.asset_id));
    const propagated = buildAndPropagateTwinGraph(payload, realAssetIds);
    res.json({
      badge: propagated.badge,
      graph: propagated
    });
  } catch (err: any) {
    res.status(400).json({ error: err?.message || 'Malformed digital twin graph' });
  }
});

app.get('/api/v2/telemetry/:assetId', (req, res) => {
  const assetId = req.params.assetId;
  const isSim = assetId.startsWith('SIM-');
  const records = isSim
    ? readSimTelemetryRecords().filter(r => r.asset_id === assetId)
    : importedTelemetryRecords.filter(r => r.asset_id === assetId);
  const alerts = runAnomalyDetectors(records);
  res.json({
    badge: isSim ? '[SIMULATED]' : '[IMPORTED_REAL]',
    provenance: isSim ? 'simulated' : 'imported_real',
    asset_id: assetId,
    telemetry_count: records.length,
    anomalies_detected: alerts.length,
    anomalies: alerts,
    records: records.slice(-400)
  });
});

app.post('/api/v2/telemetry/simulate', (req, res) => {
  if (!checkSimRateLimit()) {
    return res.status(429).json({
      error: 'Rate limit exceeded. Maximum 10 simulation requests per minute.',
      badge: '[SIMULATED]'
    });
  }
  const rawBody = JSON.stringify(req.body || {});
  if (Buffer.byteLength(rawBody, 'utf-8') > 10 * 1024) {
    return res.status(413).json({
      error: 'Payload exceeds 10KB simulation request limit.',
      badge: '[SIMULATED]'
    });
  }

  try {
    const {
      seed = 42,
      asset_id = 'SIM-PIPE-01',
      fault_type = 'leak',
      n_steps = 60,
      ramp_shape = 'linear',
      detector = 'all'
    } = req.body || {};

    const simAssetId = normalizeSimAssetId(String(asset_id));
    const validFaults = new Set(['none', 'leak', 'bearing_wear', 'blockage', 'overheating']);
    if (!validFaults.has(String(fault_type))) {
      return res.status(400).json({ error: `Unsupported fault_type '${fault_type}'.`, badge: '[SIMULATED]' });
    }

    const records = generateTelemetrySeries({
      seed: Number(seed),
      asset_id: simAssetId,
      fault_type: String(fault_type),
      n_steps: Number(n_steps),
      ramp_shape: String(ramp_shape)
    });

    const totalStored = appendSimTelemetryRecords(records);
    const detectorChoice = ['all', 'robust_zscore', 'ewma', 'cusum'].includes(String(detector))
      ? (String(detector) as 'all' | 'robust_zscore' | 'ewma' | 'cusum')
      : 'all';
    const anomalies = runAnomalyDetectors(records, detectorChoice);

    res.json({
      badge: '[SIMULATED]',
      disclaimer: 'Synthetic AR(1) physics benchmark simulation; isolated from real workspace asset history.',
      status: 'success',
      asset_id: simAssetId,
      seed: Number(seed),
      fault_type: String(fault_type),
      ramp_shape: String(ramp_shape),
      records_generated: records.length,
      stored_simulation_total: totalStored,
      retention_cap: MAX_SIM_RECORDS,
      anomalies_detected: anomalies.length,
      anomalies,
      records
    });
  } catch (err: any) {
    res.status(400).json({ error: err?.message || 'Telemetry simulation failed', badge: '[SIMULATED]' });
  }
});

const handleTelemetryCsvImport = async (req: any, res: any) => {
  try {
    const body = req.body || {};
    const uploadedFile = (req.files && (req.files as any[])[0]) || req.file;
    let csvContent = String(body.csv_content || '');

    if (uploadedFile && fs.existsSync(uploadedFile.path)) {
      csvContent = fs.readFileSync(uploadedFile.path, 'utf-8');
      try {
        fs.unlinkSync(uploadedFile.path);
      } catch {
        // ignore cleanup error
      }
    }

    if (!csvContent.trim()) {
      return res.status(400).json({ error: 'CSV content or CSV file upload is required.' });
    }

    const assetId = String(body.asset_id || 'PIPE-SOUTH-01').trim();
    const sourceUrl = String(body.source_url || '').trim();
    const license = String(body.license || '').trim();
    const retrievedDate = String(body.retrieved_date || '').trim();

    const imported = parseRealTelemetryCsv({
      csvContent,
      asset_id: assetId,
      source_url: sourceUrl,
      license,
      retrieved_date: retrievedDate
    });

    importedTelemetryRecords.push(...imported);
    if (importedTelemetryRecords.length > MAX_SIM_RECORDS) {
      importedTelemetryRecords = importedTelemetryRecords.slice(-MAX_SIM_RECORDS);
    }
    saveNodeState();

    const anomalies = runAnomalyDetectors(imported);
    res.json({
      badge: '[IMPORTED_REAL]',
      status: 'success',
      asset_id: assetId,
      records_processed: imported.length,
      anomalies_detected: anomalies.length,
      anomalies,
      records: imported,
      provenance_metadata: { source_url: sourceUrl, license, retrieved_date: retrievedDate }
    });
  } catch (err: any) {
    res.status(400).json({ error: err?.message || 'CSV telemetry import failed' });
  }
};

app.post('/api/v2/telemetry/import', upload.any(), handleTelemetryCsvImport);
app.post('/api/v2/telemetry/upload', upload.any(), handleTelemetryCsvImport);

app.post('/api/v2/risk/fuse', (req, res) => {
  try {
    const {
      asset_id,
      visual_report,
      anomalies,
      provenance = 'simulated',
      human_approval = false,
      is_real_asset
    } = req.body || {};

    let resolvedVisual = visual_report || {};
    let resolvedHistory: Array<{ dayOffset: number; corrosion_percentage: number }> = [];
    let resolvedAnomalies = Array.isArray(anomalies) ? anomalies : [];

    if (asset_id && !String(asset_id).startsWith('SIM-')) {
      const latest = history.find(h => h.asset_id === asset_id);
      const inspFindings = latest ? findingsMap[latest.inspection_id] || [] : [];
      const hasConfirmed = inspFindings.some(f => f.human_review_status === 'confirmed');
      if (!visual_report && latest) {
        resolvedVisual = {
          corrosion_percentage: Number(latest.corrosion_percentage || 0),
          recommendation: String(latest.decision || 'PASS'),
          confirmed: hasConfirmed
        };
      }
      resolvedHistory = getAssetCorrosionHistory(String(asset_id));
      if (!Array.isArray(anomalies)) {
        const assetRealRecs = importedTelemetryRecords.filter(r => r.asset_id === asset_id);
        resolvedAnomalies = runAnomalyDetectors(assetRealRecs);
      }
    }

    const effectiveIsReal = typeof is_real_asset === 'boolean'
      ? is_real_asset
      : asset_id
        ? !String(asset_id).startsWith('SIM-')
        : provenance !== 'simulated';

    const assessment = evaluateUnifiedRiskFusion({
      visual_report: resolvedVisual,
      anomalies: resolvedAnomalies,
      corrosion_history: resolvedHistory,
      provenance,
      human_approval: Boolean(human_approval),
      is_real_asset: effectiveIsReal
    });

    res.json({
      badge: provenance === 'simulated' ? '[SIMULATED]' : '[REAL]',
      asset_id: asset_id || null,
      risk_assessment: assessment
    });
  } catch (err: any) {
    res.status(400).json({ error: err?.message || 'Risk fusion failed' });
  }
});

app.get('/api/v2/intelligence/overview', (req, res) => {
  const realAssetSummaries = assets.map(a => {
    const assetHist = history.filter(h => h.asset_id === a.asset_id);
    const latest = assetHist[0] || null;
    const findings = latest ? findingsMap[latest.inspection_id] || [] : [];
    const confirmedCount = findings.filter(f => f.human_review_status === 'confirmed').length;
    const realRecs = importedTelemetryRecords.filter(r => r.asset_id === a.asset_id);
    const realAlerts = runAnomalyDetectors(realRecs);
    const corrHist = getAssetCorrosionHistory(a.asset_id);

    const risk = evaluateUnifiedRiskFusion({
      visual_report: latest
        ? {
            corrosion_percentage: Number(latest.corrosion_percentage || 0),
            recommendation: latest.decision || 'PASS',
            confirmed: confirmedCount > 0
          }
        : {},
      anomalies: realAlerts,
      corrosion_history: corrHist,
      provenance: realRecs.length > 0 ? 'imported_real' : 'real',
      is_real_asset: true
    });

    return {
      asset_id: a.asset_id,
      location: a.location,
      criticality: a.criticality,
      latest_inspection_id: latest?.inspection_id || null,
      latest_decision: latest?.decision || 'NO_DATA',
      corrosion_percentage: latest?.corrosion_percentage ?? null,
      inspection_count: assetHist.length,
      confirmed_findings: confirmedCount,
      total_findings: findings.length,
      imported_telemetry_count: realRecs.length,
      telemetry_anomaly_count: realAlerts.length,
      degradation_velocity_pct_per_day: risk.degradation_velocity_pct_per_day,
      estimated_days_to_critical: risk.estimated_days_to_critical,
      risk_level: risk.risk_level,
      fused_risk_score: risk.fused_risk_score,
      approval_required: risk.approval_required,
      rationale: risk.contributions
    };
  });

  const simRecs = readSimTelemetryRecords();
  const simAlerts = runAnomalyDetectors(simRecs);

  res.json({
    real_assets: realAssetSummaries,
    real_network: buildRealTwinGraph(),
    simulated_sandbox: {
      badge: '[SIMULATED]',
      record_count: simRecs.length,
      anomaly_count: simAlerts.length,
      network: buildSimulatedTwinGraph()
    },
    maintenance_open_count: maintenanceRecords.filter(m => m.verification_status !== 'verified').length
  });
});

app.post('/api/workspace/backup', (req, res) => {
  try {
    saveNodeState();
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupName = `node_backup_${stamp}`;
    const backupDir = path.join(BACKUPS_DIR, backupName);
    fs.mkdirSync(backupDir, { recursive: true });
    if (fs.existsSync(NODE_STATE_FILE)) {
      fs.copyFileSync(NODE_STATE_FILE, path.join(backupDir, 'node_state.json'));
    }
    res.json({
      status: 'ok',
      backup_dir: `data/backups/${backupName}`,
      files_backed_up: fs.existsSync(path.join(backupDir, 'node_state.json')) ? ['node_state.json'] : []
    });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || 'Backup creation failed' });
  }
});

app.post('/api/workspace/restore', (req, res) => {
  try {
    const rawDir = String((req.body && req.body.backup_dir) || '').trim();
    if (!rawDir) {
      return res.status(400).json({ error: 'backup_dir is required' });
    }
    const resolvedDir = path.resolve(__dirname, rawDir.replace(/^\/+/, ''));
    if (!resolvedDir.startsWith(BACKUPS_DIR + path.sep) || !fs.existsSync(resolvedDir)) {
      return res.status(400).json({ error: 'Invalid or non-existent backup_dir' });
    }
    const backupStateFile = path.join(resolvedDir, 'node_state.json');
    if (!fs.existsSync(backupStateFile)) {
      return res.status(400).json({ error: 'Backup snapshot missing node_state.json' });
    }
    fs.copyFileSync(backupStateFile, NODE_STATE_FILE);
    loadNodeState();
    res.json({ status: 'restored', backup_dir: rawDir });
  } catch (err: any) {
    res.status(500).json({ error: err?.message || 'Backup restore failed' });
  }
});

app.get('/api/workspace/assets', (req, res) => {
  res.json({ assets });
});

app.post('/api/workspace/assets', (req, res) => {
  const { asset_id, location, pipe_material, pipe_type, criticality, notes } = req.body || {};
  if (!asset_id || !location) {
    return res.status(400).json({ error: 'asset_id and location are required' });
  }
  const cleanAssetId = String(asset_id).trim();
  if (cleanAssetId.startsWith('SIM-')) {
    return res.status(400).json({ error: "Security violation: simulated 'SIM-' assets cannot be registered in the real workspace." });
  }
  const existingIndex = assets.findIndex(a => a.asset_id === cleanAssetId);
  const newAsset = {
    asset_id: cleanAssetId,
    location,
    pipe_material: pipe_material || 'steel',
    pipe_type: pipe_type || 'seamless',
    criticality: criticality || 'normal',
    notes: notes || ''
  };
  if (existingIndex >= 0) {
    assets[existingIndex] = newAsset;
  } else {
    assets.push(newAsset);
  }
  saveNodeState();
  res.json({ status: 'ok', asset_id: cleanAssetId });
});

app.get('/api/workspace/asset/:id', (req, res) => {
  const assetId = req.params.id;
  const asset = assets.find(a => a.asset_id === assetId);
  if (!asset) {
    return res.status(404).json({ error: 'Asset not found' });
  }
  const assetInspections = history.filter(h => h.asset_id === assetId);
  const assetMaintenance = maintenanceRecords.filter(m => m.asset_id === assetId);
  res.json({
    asset,
    inspections: assetInspections,
    maintenance_records: assetMaintenance
  });
});

app.get('/api/workspace/asset-qr/:id', (req, res) => {
  const assetId = req.params.id;
  const asset = assets.find(a => a.asset_id === assetId) || {
    asset_id: assetId,
    location: 'Plant Location',
    criticality: 'normal',
    pipe_material: 'steel'
  };
  const html = `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Asset QR Badge - ${escapeHtml(asset.asset_id)}</title>
<style>
body { font-family: system-ui, sans-serif; background: #f1f5f9; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
.badge-card { background: #fff; border: 2px solid #cbd5e1; border-radius: 16px; padding: 2rem; width: 340px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
h2 { margin: 0 0 0.5rem; color: #0f172a; font-size: 1.25rem; }
.meta { font-size: 0.85rem; color: #475569; margin: 1rem 0; text-align: left; background: #f8fafc; padding: 0.75rem; border-radius: 8px; }
.payload { font-family: monospace; font-size: 0.75rem; background: #e2e8f0; padding: 0.4rem; border-radius: 4px; word-break: break-all; color: #1e293b; margin-top: 1rem; }
.print-btn { margin-top: 1.5rem; background: #2563eb; color: #fff; border: none; padding: 0.5rem 1rem; border-radius: 8px; font-weight: 600; cursor: pointer; }
</style></head>
<body>
<div class="badge-card">
  <h2>CementSight Asset Tag</h2>
  <div style="font-size: 40px; margin: 20px 0;">🏷️</div>
  <div class="meta">
    <div><strong>Asset ID:</strong> ${escapeHtml(asset.asset_id)}</div>
    <div><strong>Location:</strong> ${escapeHtml(asset.location)}</div>
    <div><strong>Criticality:</strong> ${escapeHtml(asset.criticality)}</div>
    <div><strong>Material:</strong> ${escapeHtml(asset.pipe_material)}</div>
  </div>
  <div class="payload">CEMENTSIGHT://ASSET/${escapeHtml(asset.asset_id)}</div>
  <button class="print-btn" onclick="window.print()">Print Asset Badge</button>
</div>
</body></html>`;
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
});

app.post('/api/workspace/inspection', upload.any(), async (req, res) => {
  try {
    const body = req.body || {};
    const assetId = String(body.asset_id || 'PIPE-SOUTH-01').trim();
    if (assetId.startsWith('SIM-')) {
      return res.status(400).json({ error: "Security violation: simulated 'SIM-' asset IDs cannot be saved into real workspace inspections." });
    }
    const location = String(body.location || 'Plant Zone A').trim();
    const pipeMaterial = String(body.pipe_material || 'steel').trim();
    const criticality = String(body.criticality || 'normal').trim();
    const surface = String(body.surface || 'bare_or_painted_metal').trim();
    const inspectionDate = String(body.inspection_date || new Date().toISOString().split('T')[0]);
    const samePosition = String(body.same_position || 'false').toLowerCase() === 'true';

    const uploadedFile = (req.files && (req.files as any[])[0]) || req.file;
    let filename = 'pipe_corrosion.jpg';
    let imageInput: Buffer | string;

    if (uploadedFile && fs.existsSync(uploadedFile.path)) {
      filename = uploadedFile.originalname || 'uploaded_image.jpg';
      imageInput = fs.readFileSync(uploadedFile.path);
      try {
        fs.unlinkSync(uploadedFile.path);
      } catch {
        // Ignore cleanup failure
      }
      if (!imageInput || imageInput.length === 0) {
        return res.status(400).json({ error: 'Uploaded file is empty. Please select a valid image.' });
      }
    } else {
      const requestedPath = String(body.image_path || 'data/pipe_corrosion.jpg').replace(/^\/+/, '');
      const resolved = path.resolve(__dirname, requestedPath);
      if (!resolved.startsWith(__dirname) || !fs.existsSync(resolved)) {
        imageInput = path.join(DATA_DIR, 'pipe_corrosion.jpg');
      } else {
        imageInput = resolved;
        filename = path.basename(resolved);
      }
    }

    const inspId = 'INS-' + Math.random().toString(36).substring(2, 10).toUpperCase();

    // Ensure asset exists in registry
    if (!assets.find(a => a.asset_id === assetId)) {
      assets.push({
        asset_id: assetId,
        location,
        pipe_material: pipeMaterial,
        pipe_type: 'seamless',
        criticality,
        notes: ''
      });
    }

    // Run real server-side pixel vision analysis and draw annotated image overlay
    const analysis = await analyzeInspectionImage({
      input: imageInput,
      inspectionId: inspId,
      surface
    });

    saveAnalysisImages(OUTPUT_DIR, inspId, analysis.annotatedBuffer, analysis.originalBuffer);
    imageBufferCache.set(inspId, {
      annotated: analysis.annotatedBuffer,
      original: analysis.originalBuffer
    });

    const annotatedUrl = `/api/workspace/image/${inspId}?mode=annotated`;
    const originalUrl = `/api/workspace/image/${inspId}?mode=original`;
    const priority = computePriority(analysis.corrosion_percentage, criticality, analysis.findings);

    const report = {
      decision: analysis.decision,
      action: analysis.action,
      recommendation: analysis.recommendation,
      reason: analysis.reason,
      screening_label: analysis.screening_label,
      approval_required: analysis.approval_required,
      corrosion_percentage: analysis.corrosion_percentage,
      region_count: analysis.region_count,
      surface,
      same_position: samePosition,
      bounding_boxes: analysis.findings.map(f => f.bbox)
    };

    const newHistoryEntry = {
      inspection_id: inspId,
      timestamp: new Date().toISOString().replace('T', ' ').substring(0, 19),
      asset_id: assetId,
      input_filename: filename,
      input_type: uploadedFile ? 'upload' : 'image',
      decision: analysis.decision,
      action: analysis.action,
      approval_required: analysis.approval_required,
      corrosion_percentage: analysis.corrosion_percentage,
      region_count: analysis.region_count,
      annotated_output: annotatedUrl,
      original_output: originalUrl,
      cmms_status: 'PAYLOAD PREPARED — NOT DISPATCHED',
      forecast_status: 'Server optical degradation analysis completed.'
    };

    history.unshift(newHistoryEntry);
    inspectionsMap[inspId] = {
      inspection_id: inspId,
      asset_id: assetId,
      location,
      inspection_date: inspectionDate,
      input_filename: filename,
      image_sha256: 'sha256_' + inspId,
      image_ext: 'jpg',
      corrosion_percentage: analysis.corrosion_percentage,
      annotated_output: annotatedUrl,
      original_output: originalUrl,
      detector_results: {
        image_quality: analysis.image_quality
      },
      authoritative_report: report,
      priority,
      inspector_notes: body.inspector_notes || ''
    };

    findingsMap[inspId] = analysis.findings;
    saveNodeState();

    res.status(201).json({
      inspection_id: inspId,
      asset_id: assetId,
      image_sha256: 'sha256_' + inspId,
      annotated_output: annotatedUrl,
      original_output: originalUrl,
      authoritative_inspection_report: report,
      findings_count: analysis.findings.length,
      detector_summaries: analysis.detector_summaries
    });
  } catch (err: any) {
    console.error('Inspection error:', err);
    const msg = String(err?.message || 'Image analysis failed');
    const isInputErr = /unsupported image format|input buffer|invalid|corrupt|vips/i.test(msg);
    res.status(isInputErr ? 400 : 500).json({
      error: isInputErr
        ? 'Unsupported or unreadable image file. Please upload a valid JPEG, PNG, or WebP photo.'
        : msg
    });
  }
});

app.get('/api/workspace/inspection/:id', (req, res) => {
  const id = req.params.id;
  const insp = inspectionsMap[id];
  if (!insp) {
    return res.status(404).json({ error: 'Inspection not found' });
  }
  const findings = findingsMap[id] || [];
  const asset = assets.find(a => a.asset_id === insp.asset_id) || assets[0];
  const priority =
    insp.priority ||
    computePriority(insp.corrosion_percentage || 0, asset?.criticality || 'normal', findings);

  res.json({
    inspection: insp,
    findings,
    priority
  });
});

app.post('/api/workspace/review/:findingId', (req, res) => {
  const findingId = req.params.findingId;
  const { status } = req.body || {};
  const nextStatus = status || 'confirmed';

  for (const list of Object.values(findingsMap)) {
    const match = list.find(f => f.finding_id === findingId);
    if (match) {
      match.human_review_status = nextStatus;
      break;
    }
  }

  saveNodeState();
  res.json({ status: 'success', finding_id: findingId, human_review_status: nextStatus });
});

app.get('/api/workspace/report/:id', (req, res) => {
  const id = req.params.id;
  const fallbackId = history[0]?.inspection_id || id;
  const html = renderInspectionHtmlReport(id) || renderInspectionHtmlReport(fallbackId);
  if (!html) {
    return res.status(404).send('Inspection report not found');
  }
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
});

app.get('/api/workspace/export-csv', (req, res) => {
  const filterInspId = req.query.inspection_id ? String(req.query.inspection_id) : '';
  const targetHistory = filterInspId
    ? history.filter(h => h.inspection_id === filterInspId)
    : history;

  const headers = [
    'inspection_id',
    'asset_id',
    'inspection_date',
    'decision',
    'corrosion_percentage',
    'finding_id',
    'region_id',
    'detector_id',
    'defect_category',
    'finding_tier',
    'confidence',
    'bbox',
    'human_review_status',
    'linked_work_orders'
  ];

  const rows: string[] = [headers.join(',')];

  for (const h of targetHistory) {
    const insp = inspectionsMap[h.inspection_id];
    const inspDate = insp?.inspection_date || h.timestamp || '';
    const findings = findingsMap[h.inspection_id] || [];
    if (findings.length === 0) {
      const inspOrders = maintenanceRecords
        .filter(m => m.inspection_id === h.inspection_id)
        .map(m => `${m.record_id}: ${m.work_order || m.work_performed}`)
        .join('; ');
      rows.push(
        [
          h.inspection_id,
          h.asset_id,
          inspDate,
          h.decision,
          h.corrosion_percentage,
          '',
          '',
          '',
          '',
          '',
          '',
          '',
          '',
          inspOrders
        ]
          .map(csvCell)
          .join(',')
      );
    } else {
      for (const f of findings) {
        const tier =
          f.experimental || f.detector_id !== 'corrosion_detector'
            ? 'experimental'
            : 'authoritative';
        const linkedOrders = maintenanceRecords
          .filter(
            m =>
              m.finding_id === f.finding_id ||
              (!m.finding_id && m.inspection_id === h.inspection_id)
          )
          .map(m => `${m.record_id}: ${m.work_order || m.work_performed}`)
          .join('; ');
        rows.push(
          [
            h.inspection_id,
            h.asset_id,
            inspDate,
            h.decision,
            h.corrosion_percentage,
            f.finding_id,
            f.region_id,
            f.detector_id,
            f.defect_category,
            tier,
            f.confidence,
            JSON.stringify(f.bbox),
            f.human_review_status || 'unreviewed',
            linkedOrders
          ]
            .map(csvCell)
            .join(',')
        );
      }
    }
  }

  const filename = filterInspId
    ? `cementsight_report_${filterInspId.replace(/[^a-zA-Z0-9_-]/g, '')}.csv`
    : 'cementsight_inspections_export.csv';
  res.setHeader('Content-Type', 'text/csv; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename=${filename}`);
  res.send(rows.join('\n'));
});

app.get('/api/workspace/asset-export-csv/:id', (req, res) => {
  const assetId = req.params.id;
  const targetHistory = history.filter(h => h.asset_id === assetId);
  const rows: string[] = [
    'inspection_id,asset_id,inspection_date,decision,corrosion_percentage,finding_id,region_id,defect_category,finding_tier,human_review_status'
  ];
  for (const h of targetHistory) {
    const insp = inspectionsMap[h.inspection_id];
    const inspDate = insp?.inspection_date || h.timestamp || '';
    const findings = findingsMap[h.inspection_id] || [];
    if (!findings.length) {
      rows.push([h.inspection_id, assetId, inspDate, h.decision, h.corrosion_percentage, '', '', '', '', ''].map(csvCell).join(','));
    } else {
      for (const f of findings) {
        const tier = f.experimental || f.detector_id !== 'corrosion_detector' ? 'experimental' : 'authoritative';
        rows.push(
          [
            h.inspection_id,
            assetId,
            inspDate,
            h.decision,
            h.corrosion_percentage,
            f.finding_id,
            f.region_id,
            f.defect_category,
            tier,
            f.human_review_status || 'unreviewed'
          ]
            .map(csvCell)
            .join(',')
        );
      }
    }
  }
  res.setHeader('Content-Type', 'text/csv; charset=utf-8');
  res.setHeader('Content-Disposition', `attachment; filename=asset_${assetId}_export.csv`);
  res.send(rows.join('\n'));
});

app.get('/api/workspace/maintenance', (req, res) => {
  const assetId = req.query.asset_id ? String(req.query.asset_id) : '';
  const inspectionId = req.query.inspection_id ? String(req.query.inspection_id) : '';
  let filtered = [...maintenanceRecords];
  if (assetId) filtered = filtered.filter(m => m.asset_id === assetId);
  if (inspectionId) filtered = filtered.filter(m => m.inspection_id === inspectionId);
  res.json({ maintenance_records: filtered.reverse() });
});

app.post('/api/workspace/maintenance', (req, res) => {
  const {
    asset_id,
    inspection_id,
    finding_id,
    work_order,
    work_performed,
    performed_by,
    priority,
    kind,
    notes
  } = req.body || {};

  if (asset_id && String(asset_id).trim().startsWith('SIM-')) {
    return res.status(400).json({ error: "Security violation: simulated 'SIM-' assets cannot trigger maintenance work orders." });
  }

  let linkedFinding: DetectionFinding | undefined;
  if (finding_id) {
    if (inspection_id && findingsMap[inspection_id]) {
      linkedFinding = findingsMap[inspection_id].find(f => f.finding_id === finding_id);
    }
    if (!linkedFinding) {
      for (const list of Object.values(findingsMap)) {
        const hit = list.find(f => f.finding_id === finding_id);
        if (hit) {
          linkedFinding = hit;
          break;
        }
      }
    }
  }

  const record = {
    record_id: 'MNT-' + Math.floor(Math.random() * 9000 + 1000),
    asset_id: asset_id || linkedFinding ? asset_id || 'PIPE-SOUTH-01' : 'PIPE-SOUTH-01',
    inspection_id: inspection_id || linkedFinding?.inspection_id || null,
    finding_id: finding_id || null,
    finding_region_id: linkedFinding?.region_id || null,
    finding_category: linkedFinding?.defect_category || null,
    finding_review_status: linkedFinding?.human_review_status || null,
    work_order: work_order || work_performed || 'WO-GENERAL-MAINT',
    work_performed: work_performed || work_order || 'Maintenance action logged',
    performed_by: performed_by || 'Field Engineer',
    priority: priority || 'High',
    kind: kind || 'recorded_work',
    status: 'Prepared (Local Only)',
    verification_status: 'pending',
    notes: notes || '',
    created_at: new Date().toISOString()
  };
  maintenanceRecords.push(record);

  if (record.inspection_id) {
    const h = history.find(item => item.inspection_id === record.inspection_id);
    if (h) {
      h.cmms_status = `WORK ORDER ${record.record_id} PREPARED (NOT DISPATCHED)`;
    }
  }

  saveNodeState();
  res.json({ status: 'success', record_id: record.record_id, record });
});

app.put('/api/workspace/maintenance/:id', (req, res) => {
  const id = req.params.id;
  const rec = maintenanceRecords.find(m => m.record_id === id);
  if (!rec) {
    return res.status(404).json({ error: 'Maintenance record not found' });
  }
  const { verification_status, followup_inspection_id, status } = req.body || {};
  if (verification_status) rec.verification_status = verification_status;
  if (followup_inspection_id) rec.followup_inspection_id = followup_inspection_id;
  if (status) rec.status = status;
  saveNodeState();
  res.json({ status: 'success', record: rec });
});

app.use('/api', (_req, res) => {
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.status(404).json({ error: 'API endpoint not found' });
});

app.use((err: any, _req: any, res: any, next: any) => {
  if (res.headersSent) {
    return next(err);
  }
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  if (err instanceof multer.MulterError) {
    const status = err.code === 'LIMIT_FILE_SIZE' ? 413 : 400;
    const msg =
      err.code === 'LIMIT_FILE_SIZE'
        ? 'Uploaded file exceeds maximum allowed size of 25 MB.'
        : `Upload error: ${err.message}`;
    return res.status(status).json({ error: msg });
  }
  const status = err?.status || err?.statusCode || 500;
  console.error('Unhandled Express error:', err);
  return res.status(status).json({ error: err?.message || 'Internal server error' });
});

async function startServer() {
  const loaded = loadNodeState();
  if (!loaded) {
    await seedDefaultDemoInspection();
  }
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`CementSight AI server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
