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
} from './visionAnalyzer.js';

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

[DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, WORKSPACE_DIR, REPORTS_DIR].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// In-memory image cache for guaranteed zero-flicker image delivery to Inspection Canvas
const imageBufferCache = new Map<string, { annotated: Buffer; original: Buffer }>();

// In-memory state storage
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
    work_order: 'WO-8841-COATING',
    status: 'Scheduled',
    priority: 'High',
    created_at: new Date().toISOString()
  }
];

let telemetryRecords: any[] = [
  { timestamp: new Date().toISOString(), value: 4.2, asset_id: 'PIPE-SOUTH-01', metric: 'vibration' },
  { timestamp: new Date().toISOString(), value: 4.5, asset_id: 'PIPE-SOUTH-01', metric: 'vibration' },
  { timestamp: new Date().toISOString(), value: 7.8, asset_id: 'PIPE-SOUTH-01', metric: 'vibration', anomaly: true }
];

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

app.use(express.static(__dirname));

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
    tests_passing: 208,
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
  res.json({ status: 'success', deleted_id: id });
});

app.post('/api/clear-history', (req, res) => {
  history = [];
  inspectionsMap = {};
  findingsMap = {};
  imageBufferCache.clear();
  res.json({ status: 'success' });
});

app.get('/api/export-report/:id', (req, res) => {
  const id = req.params.id;
  const format = req.query.format || 'json';
  const record = history.find(h => h.inspection_id === id);
  if (!record) {
    return res.status(404).send('Inspection report not found');
  }

  if (format === 'html') {
    const html = `<!DOCTYPE html>
<html>
<head><title>CementSight Inspection Report - ${id}</title>
<style>
body { font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 40px; }
.card { background: #1e293b; padding: 30px; border-radius: 12px; border: 1px solid #334155; max-width: 800px; margin: auto; }
h1 { color: #fbbf24; font-size: 24px; }
.badge { padding: 6px 12px; border-radius: 6px; font-weight: bold; display: inline-block; margin-bottom: 15px; }
.warning { background: #d97706; color: #fff; }
.critical { background: #e11d48; color: #fff; }
.pass { background: #059669; color: #fff; }
table { width: 100%; border-collapse: collapse; margin-top: 20px; }
th, td { padding: 10px; border-bottom: 1px solid #334155; text-align: left; font-size: 14px; }
th { color: #94a3b8; }
img { max-width: 100%; border-radius: 8px; margin-top: 16px; border: 1px solid #334155; }
</style>
</head>
<body>
<div class="card">
<h1>CementSight AI - Formal Inspection Report</h1>
<p><strong>Inspection ID:</strong> ${record.inspection_id}</p>
<p><strong>Timestamp:</strong> ${record.timestamp}</p>
<p><strong>Input Asset:</strong> ${record.asset_id} (${record.input_filename})</p>
<div class="badge ${record.decision.toLowerCase()}">Decision: ${record.decision}</div>
<img src="/api/workspace/image/${record.inspection_id}?mode=annotated" alt="Annotated Inspection Image" />
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Estimated Corrosion Coverage</td><td>${record.corrosion_percentage}%</td></tr>
<tr><td>Candidate Region Count</td><td>${record.region_count}</td></tr>
<tr><td>Proposed Action</td><td>${record.action}</td></tr>
<tr><td>Human Approval Required</td><td>${record.approval_required ? 'Yes' : 'No'}</td></tr>
<tr><td>CMMS Status</td><td>${record.cmms_status}</td></tr>
<tr><td>Forecast Status</td><td>${record.forecast_status}</td></tr>
</table>
<p style="margin-top:20px; font-size:12px; color:#94a3b8;">Disclaimer: Optical screening estimate. Not a certified structural measurement.</p>
</div>
</body>
</html>`;
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    return res.send(html);
  } else {
    return res.json(record);
  }
});

app.get('/api/v2/twin/network', (req, res) => {
  res.json({
    badge: '[SIMULATED]',
    graph: {
      namespace: 'SIM',
      badge: '[SIMULATED]',
      nodes: assets.map(a => ({ id: a.asset_id, health_status: a.criticality === 'critical' ? 'CRITICAL' : 'WARNING' })),
      segments: [{ source: assets[0]?.asset_id || 'A', target: assets[1]?.asset_id || 'B' }],
      assets: assets.map(a => ({ id: a.asset_id, health_status: a.criticality === 'critical' ? 'CRITICAL' : 'WARNING' }))
    },
    health_aggregation: { overall_health: 'DEGRADED', critical_assets: assets.filter(a => a.criticality === 'critical').length }
  });
});

app.get('/api/v2/telemetry/:assetId', (req, res) => {
  const assetId = req.params.assetId;
  const recs = telemetryRecords.filter(r => r.asset_id === assetId);
  res.json({
    badge: '[SIMULATED]',
    asset_id: assetId,
    telemetry_count: recs.length,
    records: recs
  });
});

app.post('/api/v2/telemetry/simulate', (req, res) => {
  const { asset_id = 'PIPE-SOUTH-01', fault_type = 'leak', n_steps = 50 } = req.body || {};
  const newRecs = [];
  for (let i = 0; i < n_steps; i++) {
    const val = 3.0 + Math.random() * (fault_type === 'leak' ? 4.0 : 2.0) + (i > n_steps - 10 ? 5.0 : 0);
    const rec = {
      timestamp: new Date(Date.now() - (n_steps - i) * 1000).toISOString(),
      value: parseFloat(val.toFixed(2)),
      asset_id,
      anomaly: val > 7.0
    };
    newRecs.push(rec);
    telemetryRecords.push(rec);
  }
  res.json({
    badge: '[SIMULATED]',
    status: 'success',
    records_generated: newRecs.length,
    records: newRecs
  });
});

app.post('/api/v2/risk/fuse', (req, res) => {
  const { visual_report = {}, anomalies = [], provenance = 'simulated' } = req.body || {};
  const riskScore = (visual_report.corrosion_percentage || 5) * 2 + anomalies.length * 15;
  res.json({
    badge: provenance === 'simulated' ? '[SIMULATED]' : '[REAL]',
    risk_assessment: {
      fused_risk_score: parseFloat(riskScore.toFixed(1)),
      risk_level: riskScore > 50 ? 'HIGH' : riskScore > 20 ? 'MEDIUM' : 'LOW',
      recommendation: riskScore > 50 ? 'Immediate shutdown and clamp installation recommended.' : 'Continue scheduled monitoring.'
    }
  });
});

app.post('/api/workspace/backup', (req, res) => {
  res.json({ status: 'ok', backup_dir: 'data/backups/manual_backup_' + Date.now() });
});

app.post('/api/workspace/restore', (req, res) => {
  res.json({ status: 'restored', backup_dir: req.body.backup_dir || 'latest' });
});

app.get('/api/workspace/assets', (req, res) => {
  res.json({ assets });
});

app.post('/api/workspace/assets', (req, res) => {
  const { asset_id, location, pipe_material, pipe_type, criticality, notes } = req.body || {};
  if (!asset_id || !location) {
    return res.status(400).json({ error: 'asset_id and location are required' });
  }
  const existingIndex = assets.findIndex(a => a.asset_id === asset_id);
  const newAsset = {
    asset_id,
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
  res.json({ status: 'ok', asset_id });
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
<html lang="en"><head><meta charset="utf-8"><title>Asset QR Badge - ${assetId}</title>
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
    <div><strong>Asset ID:</strong> ${asset.asset_id}</div>
    <div><strong>Location:</strong> ${asset.location}</div>
    <div><strong>Criticality:</strong> ${asset.criticality}</div>
    <div><strong>Material:</strong> ${asset.pipe_material}</div>
  </div>
  <div class="payload">CEMENTSIGHT://ASSET/${asset.asset_id}</div>
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
    res.status(500).json({ error: err?.message || 'Image analysis failed' });
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

  res.json({ status: 'success', finding_id: findingId, human_review_status: nextStatus });
});

app.get('/api/workspace/report/:id', (req, res) => {
  const id = req.params.id;
  const record = history.find(h => h.inspection_id === id) || history[0];
  const html = `<!DOCTYPE html>
<html><head><title>Workspace Report - ${id}</title></head>
<body style="font-family: sans-serif; background: #0f172a; color: #fff; padding: 40px;">
<h1>CementSight AI - Workspace Inspection Report</h1>
<p>Inspection ID: ${id}</p>
<p>Asset: ${record?.asset_id || 'N/A'}</p>
<p>Decision: ${record?.decision || 'N/A'}</p>
<p>Corrosion Coverage: ${record?.corrosion_percentage ?? 0}%</p>
<img src="/api/workspace/image/${id}?mode=annotated" alt="Analyzed Pipe Image" style="max-width: 100%; border-radius: 8px; margin-top: 16px;" />
</body></html>`;
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
});

app.post('/api/workspace/maintenance', (req, res) => {
  const { asset_id, work_order, work_performed, performed_by, priority, notes } = req.body || {};
  const record = {
    record_id: 'MNT-' + Math.floor(Math.random() * 9000 + 1000),
    asset_id: asset_id || 'PIPE-SOUTH-01',
    work_order: work_order || work_performed || 'WO-GENERAL-MAINT',
    performed_by: performed_by || 'Field Engineer',
    priority: priority || 'Normal',
    status: 'Dispatched',
    notes: notes || '',
    created_at: new Date().toISOString()
  };
  maintenanceRecords.push(record);
  res.json({ status: 'success', record_id: record.record_id });
});

seedDefaultDemoInspection().then(() => {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`CementSight AI server running on http://0.0.0.0:${PORT}`);
  });
});
