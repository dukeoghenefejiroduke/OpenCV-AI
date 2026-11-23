/**
 * DeltaTwin Engineering Intelligence Engine (TypeScript port for Node.js runtime)
 *
 * Implements deterministic, dependency-free equivalents of:
 * - telemetry/schema.py, telemetry/simulator.py
 * - anomaly/detectors.py (Robust Z-Score Median/MAD, EWMA, CUSUM)
 * - twin/graph.py (DAG cycle validation, health propagation, namespace isolation)
 * - app/forecasting.py (degradation velocity & time-to-critical with >= 3 observation guard)
 * - risk/engine.py (rule-based visual + telemetry fusion with simulation firewall)
 * - tools/import_telemetry.py (imported_real CSV parser requiring source_url, license, retrieved_date)
 */

export type Provenance = 'simulated' | 'imported_real';
export type MetricName = 'pressure_kpa' | 'flow_m3h' | 'temperature_c' | 'vibration_mm_s';
export type FaultType = 'none' | 'leak' | 'bearing_wear' | 'blockage' | 'overheating';
export type RampShape = 'linear' | 'exponential' | 'weibull';
export type HealthStatus = 'HEALTHY' | 'WARNING' | 'CRITICAL';

export interface TelemetryRecord {
  sensor_id: string;
  asset_id: string;
  timestamp: string;
  metric: MetricName | string;
  value: number;
  units: string;
  provenance: Provenance;
  scenario_id: string;
  simulator_version: string;
  seed: number;
  quality_flag: string;
  ground_truth_fault?: boolean;
  source_url?: string;
  license?: string;
  retrieved_date?: string;
}

export interface AnomalyAlert {
  metric: string;
  asset_id: string;
  detector: 'robust_zscore' | 'ewma' | 'cusum';
  score: number;
  threshold: number;
  window: number;
  timestamp: string;
  value: number;
  plain_language_reason: string;
  provenance: Provenance;
}

export interface TwinNode {
  id: string;
  label?: string;
  zone?: string;
  equipment_type?: string;
  health_status: HealthStatus;
  corrosion_percentage?: number;
  anomaly_count?: number;
  provenance?: string;
}

export interface TwinSegment {
  source: string;
  target: string;
  label?: string;
}

export interface TwinGraphInput {
  namespace: string;
  nodes: TwinNode[];
  segments: TwinSegment[];
  assets?: Array<{ id: string; health_status?: HealthStatus }>;
}

const VALID_PROVENANCE = new Set<string>(['simulated', 'imported_real']);

function createMulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createGaussianRng(seed: number): (mean?: number, std?: number) => number {
  const rand = createMulberry32(seed);
  return (mean = 0, std = 1) => {
    const u1 = Math.max(1e-12, rand());
    const u2 = rand();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z0 * std;
  };
}

export function validateTelemetryRecord(record: any): TelemetryRecord {
  if (!record || typeof record !== 'object') {
    throw new Error('Telemetry record must be a dictionary.');
  }
  if (!record.sensor_id || typeof record.sensor_id !== 'string') {
    throw new Error("Field 'sensor_id' is mandatory and must be a non-empty string.");
  }
  if (!record.asset_id || typeof record.asset_id !== 'string') {
    throw new Error("Field 'asset_id' is mandatory and must be a non-empty string.");
  }
  if (!record.timestamp || typeof record.timestamp !== 'string' || Number.isNaN(Date.parse(record.timestamp))) {
    throw new Error("Field 'timestamp' is mandatory and must be a valid ISO datetime string.");
  }
  if (!record.metric || typeof record.metric !== 'string') {
    throw new Error("Field 'metric' is mandatory and must be a string.");
  }
  if (typeof record.value !== 'number' || !Number.isFinite(record.value)) {
    throw new Error("Field 'value' is mandatory and must be a finite numeric value.");
  }
  if (!record.units || typeof record.units !== 'string') {
    throw new Error("Field 'units' is mandatory and must be a string.");
  }
  if (!VALID_PROVENANCE.has(record.provenance)) {
    throw new Error(`Field 'provenance' must be 'simulated' or 'imported_real'.`);
  }
  if (!record.scenario_id || typeof record.scenario_id !== 'string') {
    throw new Error("Field 'scenario_id' is mandatory and must be a string.");
  }
  if (!record.simulator_version || typeof record.simulator_version !== 'string') {
    throw new Error("Field 'simulator_version' is mandatory and must be a string.");
  }
  if (typeof record.seed !== 'number' || !Number.isInteger(record.seed)) {
    throw new Error("Field 'seed' is mandatory and must be an integer.");
  }
  if (record.provenance === 'simulated' && typeof record.ground_truth_fault !== 'boolean') {
    throw new Error("Field 'ground_truth_fault' is mandatory boolean for simulated records.");
  }
  if (record.provenance === 'imported_real') {
    if (!record.source_url || !record.license || !record.retrieved_date) {
      throw new Error("Imported real telemetry requires 'source_url', 'license', and 'retrieved_date'.");
    }
  }
  return record as TelemetryRecord;
}

export function normalizeSimAssetId(rawAssetId: string): string {
  const clean = String(rawAssetId || 'SIM-PIPE-01').trim() || 'SIM-PIPE-01';
  return clean.startsWith('SIM-') ? clean : `SIM-${clean}`;
}

export function generateTelemetrySeries(opts: {
  seed?: number;
  asset_id?: string;
  scenario_id?: string;
  fault_type?: FaultType | string;
  fault_start_step?: number;
  n_steps?: number;
  ramp_shape?: RampShape | string;
  simulator_version?: string;
}): TelemetryRecord[] {
  const seed = Number.isInteger(opts.seed) ? Number(opts.seed) : 42;
  const assetId = normalizeSimAssetId(opts.asset_id || 'SIM-PIPE-01');
  const faultType = String(opts.fault_type || 'none');
  const scenarioId = String(opts.scenario_id || `scen_${faultType}_${seed}`);
  const nSteps = Math.max(10, Math.min(250, Number(opts.n_steps) || 60));
  const faultStartStep = Number.isFinite(opts.fault_start_step)
    ? Math.max(1, Math.min(nSteps - 1, Number(opts.fault_start_step)))
    : Math.floor(nSteps / 2);
  const rampShape = String(opts.ramp_shape || 'linear');
  const simulatorVersion = String(opts.simulator_version || 'v1.1');

  const randn = createGaussianRng(seed);
  const startMs = Date.UTC(2026, 10, 1, 0, 0, 0);

  const basePressure = 100.0;
  const baseFlow = 50.0;
  const baseTemp = 45.0;
  const baseVibration = 2.0;

  let prevP = 0;
  let prevF = 0;
  let prevT = 0;
  let prevV = 0;
  const phi = 0.4;

  const records: TelemetryRecord[] = [];

  for (let step = 0; step < nSteps; step++) {
    const tIso = new Date(startMs + step * 15 * 60 * 1000).toISOString().replace('.000Z', 'Z');
    const isFault = step >= faultStartStep && faultType !== 'none';
    const deltaSteps = Math.max(0, step - faultStartStep + 1);

    let factor = 0.0;
    if (isFault) {
      if (rampShape === 'exponential') {
        factor = 1.0 - Math.exp(-0.3 * deltaSteps);
      } else if (rampShape === 'weibull') {
        factor = 1.0 - Math.exp(-Math.pow(deltaSteps / 6.0, 1.8));
      } else {
        factor = Math.min(1.0, deltaSteps / 8.0);
      }
    }

    const rawP = randn(0, 0.8);
    const rawF = randn(0, 0.4) + 0.3 * rawP;
    const rawT = randn(0, 0.15);
    const rawV = randn(0, 0.08) + 0.2 * rawT;

    const pNoise = phi * prevP + rawP;
    const fNoise = phi * prevF + rawF;
    const tNoise = phi * prevT + rawT;
    const vNoise = phi * prevV + rawV;
    prevP = pNoise;
    prevF = fNoise;
    prevT = tNoise;
    prevV = vNoise;

    let pVal = basePressure + pNoise;
    let fVal = baseFlow + fNoise;
    let tVal = baseTemp + tNoise;
    let vVal = baseVibration + vNoise;

    if (isFault) {
      if (faultType === 'leak') {
        pVal -= 18.0 * factor;
        fVal -= 10.0 * factor;
      } else if (faultType === 'bearing_wear') {
        vVal += 5.5 * factor;
        tVal += 3.0 * factor;
      } else if (faultType === 'blockage') {
        pVal += 22.0 * factor;
        fVal -= 14.0 * factor;
      } else if (faultType === 'overheating') {
        tVal += 20.0 * factor;
        pVal += 5.0 * factor;
      }
    }

    const stepRecs: TelemetryRecord[] = [
      {
        sensor_id: `${assetId}-PT-01`,
        asset_id: assetId,
        timestamp: tIso,
        metric: 'pressure_kpa',
        value: Number(pVal.toFixed(2)),
        units: 'kPa',
        provenance: 'simulated',
        scenario_id: scenarioId,
        simulator_version: simulatorVersion,
        seed,
        quality_flag: 'ok',
        ground_truth_fault: isFault && ['leak', 'blockage', 'overheating'].includes(faultType)
      },
      {
        sensor_id: `${assetId}-FT-01`,
        asset_id: assetId,
        timestamp: tIso,
        metric: 'flow_m3h',
        value: Number(fVal.toFixed(2)),
        units: 'm3/h',
        provenance: 'simulated',
        scenario_id: scenarioId,
        simulator_version: simulatorVersion,
        seed,
        quality_flag: 'ok',
        ground_truth_fault: isFault && ['leak', 'blockage'].includes(faultType)
      },
      {
        sensor_id: `${assetId}-TT-01`,
        asset_id: assetId,
        timestamp: tIso,
        metric: 'temperature_c',
        value: Number(tVal.toFixed(2)),
        units: 'C',
        provenance: 'simulated',
        scenario_id: scenarioId,
        simulator_version: simulatorVersion,
        seed,
        quality_flag: 'ok',
        ground_truth_fault: isFault && ['overheating', 'bearing_wear'].includes(faultType)
      },
      {
        sensor_id: `${assetId}-VT-01`,
        asset_id: assetId,
        timestamp: tIso,
        metric: 'vibration_mm_s',
        value: Number(vVal.toFixed(2)),
        units: 'mm/s',
        provenance: 'simulated',
        scenario_id: scenarioId,
        simulator_version: simulatorVersion,
        seed,
        quality_flag: 'ok',
        ground_truth_fault: isFault && faultType === 'bearing_wear'
      }
    ];

    for (const r of stepRecs) {
      validateTelemetryRecord(r);
      records.push(r);
    }
  }

  return records;
}

function medianOf(sortedVals: number[]): number {
  const n = sortedVals.length;
  if (n === 0) return 0;
  const mid = Math.floor(n / 2);
  return n % 2 === 0 ? (sortedVals[mid - 1] + sortedVals[mid]) / 2 : sortedVals[mid];
}

export function detectRobustZScore(
  values: Array<number | null>,
  timestamps: string[],
  metric: string,
  assetId: string,
  provenance: Provenance,
  window = 20,
  threshold = 3.0
): AnomalyAlert[] {
  const alerts: AnomalyAlert[] = [];
  if (values.length < 3) return alerts;

  for (let i = 0; i < values.length; i++) {
    const cur = values[i];
    if (cur === null || !Number.isFinite(cur)) continue;

    const startIdx = Math.max(0, i - window + 1);
    const validSlice: number[] = [];
    for (let j = startIdx; j <= i; j++) {
      const v = values[j];
      if (v !== null && Number.isFinite(v)) validSlice.push(v);
    }
    if (validSlice.length < 2) continue;

    // Flatline / stuck-sensor guard matching anomaly/detectors.py
    if (validSlice.every(v => v === validSlice[0]) && validSlice.length >= Math.floor(window / 2)) {
      continue;
    }

    const sorted = [...validSlice].sort((a, b) => a - b);
    const med = medianOf(sorted);
    const devs = validSlice.map(v => Math.abs(v - med)).sort((a, b) => a - b);
    const mad = medianOf(devs);
    const scaleMad = mad > 0 ? 1.4826 * mad : 1e-6;
    const score = Math.abs(cur - med) / scaleMad;

    if (score > threshold) {
      alerts.push({
        metric,
        asset_id: assetId,
        detector: 'robust_zscore',
        score: Number(score.toFixed(2)),
        threshold,
        window,
        timestamp: timestamps[i] || '',
        value: cur,
        plain_language_reason: `Robust Z-score ${score.toFixed(2)} exceeds ${threshold} (median=${med.toFixed(2)}, MAD=${mad.toFixed(2)})`,
        provenance
      });
    }
  }
  return alerts;
}

export function detectEWMA(
  values: Array<number | null>,
  timestamps: string[],
  metric: string,
  assetId: string,
  provenance: Provenance,
  span = 15,
  threshold = 2.5,
  alpha = 0.3
): AnomalyAlert[] {
  const alerts: AnomalyAlert[] = [];
  let ewma: number | null = null;
  let ewmVar = 0.0;

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null || !Number.isFinite(v)) continue;
    if (ewma === null) {
      ewma = v;
      ewmVar = 1.0;
      continue;
    }
    const diff = v - ewma;
    ewma = alpha * v + (1 - alpha) * ewma;
    ewmVar = (1 - alpha) * (ewmVar + alpha * diff * diff);
    const std = Math.max(Math.sqrt(ewmVar), 1e-5);
    const score = Math.abs(v - ewma) / std;
    if (score > threshold) {
      alerts.push({
        metric,
        asset_id: assetId,
        detector: 'ewma',
        score: Number(score.toFixed(2)),
        threshold,
        window: span,
        timestamp: timestamps[i] || '',
        value: v,
        plain_language_reason: `EWMA deviation score ${score.toFixed(2)} exceeds threshold ${threshold}`,
        provenance
      });
    }
  }
  return alerts;
}

export function detectCUSUM(
  values: Array<number | null>,
  timestamps: string[],
  metric: string,
  assetId: string,
  provenance: Provenance,
  threshold = 4.0,
  drift = 0.5
): AnomalyAlert[] {
  const alerts: AnomalyAlert[] = [];
  const valid = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (valid.length < 5) return alerts;

  const mean = valid.reduce((a, b) => a + b, 0) / valid.length;
  const variance = valid.reduce((a, b) => a + (b - mean) * (b - mean), 0) / valid.length;
  const std = Math.max(Math.sqrt(variance), 1e-5);

  let sHigh = 0;
  let sLow = 0;

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v === null || !Number.isFinite(v)) continue;
    const z = (v - mean) / std;
    sHigh = Math.max(0, sHigh + z - drift);
    sLow = Math.max(0, sLow - z - drift);
    const maxS = Math.max(sHigh, sLow);
    if (maxS > threshold) {
      alerts.push({
        metric,
        asset_id: assetId,
        detector: 'cusum',
        score: Number(maxS.toFixed(2)),
        threshold,
        window: 20,
        timestamp: timestamps[i] || '',
        value: v,
        plain_language_reason: `CUSUM cumulative shift ${maxS.toFixed(2)} exceeds threshold ${threshold}`,
        provenance
      });
      sHigh = 0;
      sLow = 0;
    }
  }
  return alerts;
}

export function runAnomalyDetectors(
  records: TelemetryRecord[],
  detectorFilter: 'all' | 'robust_zscore' | 'ewma' | 'cusum' = 'all'
): AnomalyAlert[] {
  const byAssetMetric = new Map<string, TelemetryRecord[]>();
  for (const r of records) {
    const key = `${r.asset_id}::${r.metric}`;
    if (!byAssetMetric.has(key)) byAssetMetric.set(key, []);
    byAssetMetric.get(key)!.push(r);
  }

  const allAlerts: AnomalyAlert[] = [];
  for (const [, list] of byAssetMetric.entries()) {
    const sorted = [...list].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    const vals = sorted.map(r => r.value);
    const ts = sorted.map(r => r.timestamp);
    const metric = sorted[0].metric;
    const assetId = sorted[0].asset_id;
    const prov = sorted[0].provenance;

    if (detectorFilter === 'all' || detectorFilter === 'robust_zscore') {
      allAlerts.push(...detectRobustZScore(vals, ts, metric, assetId, prov));
    }
    if (detectorFilter === 'all' || detectorFilter === 'ewma') {
      allAlerts.push(...detectEWMA(vals, ts, metric, assetId, prov));
    }
    if (detectorFilter === 'all' || detectorFilter === 'cusum') {
      allAlerts.push(...detectCUSUM(vals, ts, metric, assetId, prov));
    }
  }
  return allAlerts.sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

export function parseRealTelemetryCsv(opts: {
  csvContent: string;
  asset_id: string;
  source_url: string;
  license: string;
  retrieved_date: string;
}): TelemetryRecord[] {
  const { csvContent, asset_id, source_url, license, retrieved_date } = opts;
  if (!source_url || !source_url.trim()) {
    throw new Error("Import refused: mandatory field 'source_url' is missing.");
  }
  if (!license || !license.trim()) {
    throw new Error("Import refused: mandatory field 'license' is missing.");
  }
  if (!retrieved_date || !retrieved_date.trim()) {
    throw new Error("Import refused: mandatory field 'retrieved_date' is missing.");
  }
  if (asset_id.startsWith('SIM-')) {
    throw new Error("Import refused: real imported telemetry cannot use simulated 'SIM-' asset prefix.");
  }

  const lines = String(csvContent || '')
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(Boolean);
  if (lines.length < 2) {
    throw new Error('CSV must include a header row and at least one data row.');
  }

  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const records: TelemetryRecord[] = [];
  const baseMs = Date.now();

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(c => c.trim());
    const row: Record<string, string> = {};
    headers.forEach((h, idx) => {
      row[h] = cols[idx] ?? '';
    });

    const rawVal = row.value ?? row.pressure ?? row.flow ?? row.temperature ?? row.vibration ?? cols[cols.length - 1];
    const numVal = Number(rawVal);
    if (!Number.isFinite(numVal)) continue;

    const rawTs = row.timestamp && !Number.isNaN(Date.parse(row.timestamp))
      ? new Date(row.timestamp).toISOString().replace('.000Z', 'Z')
      : new Date(baseMs + i * 60000).toISOString().replace('.000Z', 'Z');

    const metric = (row.metric || 'pressure_kpa').toLowerCase();
    const defaultUnits =
      metric.includes('flow') ? 'm3/h' :
      metric.includes('temp') ? 'C' :
      metric.includes('vib') ? 'mm/s' : 'kPa';

    const rec: TelemetryRecord = {
      sensor_id: row.sensor_id || `${asset_id}-EXT-01`,
      asset_id,
      timestamp: rawTs,
      metric,
      value: Number(numVal.toFixed(3)),
      units: row.units || defaultUnits,
      provenance: 'imported_real',
      scenario_id: row.scenario_id || 'field_observation',
      simulator_version: 'import_v1.0',
      seed: 0,
      quality_flag: 'ok',
      source_url: source_url.trim(),
      license: license.trim(),
      retrieved_date: retrieved_date.trim()
    };
    validateTelemetryRecord(rec);
    records.push(rec);
  }

  if (records.length === 0) {
    throw new Error('No valid numeric telemetry rows parsed from CSV.');
  }
  return records;
}

export function buildAndPropagateTwinGraph(
  graph: TwinGraphInput,
  realWorkspaceAssetIds?: Set<string>
): {
  namespace: string;
  badge: string;
  node_count: number;
  segment_count: number;
  initial_health: Record<string, HealthStatus>;
  propagated_health: Record<string, HealthStatus>;
  nodes: TwinNode[];
  segments: TwinSegment[];
} {
  if (!graph || !Array.isArray(graph.nodes)) {
    throw new Error("Malformed graph: missing 'nodes' array.");
  }
  const namespace = graph.namespace || 'SIM';
  const nodeIds = new Set(graph.nodes.map(n => n.id));
  for (const n of graph.nodes) {
    if (!n.id) throw new Error('Malformed graph: every node requires an id.');
  }
  for (const seg of graph.segments || []) {
    if (!nodeIds.has(seg.source) || !nodeIds.has(seg.target)) {
      throw new Error(`Malformed segment referencing non-existent node: ${seg.source} -> ${seg.target}`);
    }
  }

  if (namespace === 'SIM' && realWorkspaceAssetIds) {
    for (const n of graph.nodes) {
      if (realWorkspaceAssetIds.has(n.id)) {
        throw new Error(`Namespace isolation violation: simulated node ${n.id} overlaps with real workspace.`);
      }
    }
    for (const a of graph.assets || []) {
      if (realWorkspaceAssetIds.has(a.id)) {
        throw new Error(`Namespace isolation violation: simulated asset ${a.id} overlaps with real workspace.`);
      }
    }
  }

  // DFS cycle detection matching twin/graph.py
  const adj = new Map<string, string[]>();
  for (const id of nodeIds) adj.set(id, []);
  for (const seg of graph.segments || []) {
    adj.get(seg.source)!.push(seg.target);
  }

  const visited = new Set<string>();
  const recStack = new Set<string>();
  const dfs = (id: string): boolean => {
    visited.add(id);
    recStack.add(id);
    for (const nb of adj.get(id) || []) {
      if (!visited.has(nb)) {
        if (dfs(nb)) return true;
      } else if (recStack.has(nb)) {
        return true;
      }
    }
    recStack.delete(id);
    return false;
  };
  for (const id of nodeIds) {
    if (!visited.has(id) && dfs(id)) {
      throw new Error('Malformed graph: cyclic dependency detected.');
    }
  }

  const severity: Record<HealthStatus, number> = { CRITICAL: 3, WARNING: 2, HEALTHY: 1 };
  const initialHealth: Record<string, HealthStatus> = {};
  for (const n of graph.nodes) {
    initialHealth[n.id] = (n.health_status || 'HEALTHY') as HealthStatus;
  }

  const propagated: Record<string, HealthStatus> = { ...initialHealth };
  let changed = true;
  while (changed) {
    changed = false;
    for (const [src, targets] of adj.entries()) {
      const srcStatus = propagated[src] || 'HEALTHY';
      for (const tgt of targets) {
        const tgtStatus = propagated[tgt] || 'HEALTHY';
        if ((severity[srcStatus] || 1) > (severity[tgtStatus] || 1)) {
          propagated[tgt] = srcStatus;
          changed = true;
        }
      }
    }
  }

  const enrichedNodes = graph.nodes.map(n => ({
    ...n,
    health_status: propagated[n.id] || n.health_status
  }));

  return {
    namespace,
    badge: namespace === 'SIM' ? '[SIMULATED]' : '[REAL]',
    node_count: graph.nodes.length,
    segment_count: (graph.segments || []).length,
    initial_health: initialHealth,
    propagated_health: propagated,
    nodes: enrichedNodes,
    segments: graph.segments || []
  };
}

export function calculateDegradationVelocity(
  observations: Array<{ dayOffset: number; corrosion_percentage: number }>
): number {
  if (!Array.isArray(observations) || observations.length < 3) return 0.0;
  const seen = new Set<number>();
  const valid: Array<{ t: number; c: number }> = [];
  for (const obs of observations) {
    if (!Number.isFinite(obs.dayOffset) || !Number.isFinite(obs.corrosion_percentage)) continue;
    if (seen.has(obs.dayOffset)) continue;
    seen.add(obs.dayOffset);
    valid.push({ t: obs.dayOffset, c: obs.corrosion_percentage });
  }
  if (valid.length < 3) return 0.0;
  valid.sort((a, b) => a.t - b.t);
  const dt = valid[valid.length - 1].t - valid[0].t;
  if (dt <= 0) return 0.0;
  const rate = (valid[valid.length - 1].c - valid[0].c) / dt;
  return Number.isFinite(rate) ? Number(rate.toFixed(4)) : 0.0;
}

export function projectTimeToCritical(
  currentCorrosion: number,
  dailyRate: number,
  criticalThreshold = 10.0
): number | null {
  if (!Number.isFinite(currentCorrosion) || !Number.isFinite(dailyRate)) return null;
  if (currentCorrosion >= criticalThreshold) return 0.0;
  if (dailyRate <= 0) return null;
  const days = (criticalThreshold - currentCorrosion) / dailyRate;
  return Number.isFinite(days) ? Number(days.toFixed(1)) : null;
}

export function evaluateUnifiedRiskFusion(params: {
  visual_report?: { corrosion_percentage?: number; recommendation?: string; confirmed?: boolean };
  anomalies?: Array<{ detector?: string; provenance?: string; metric?: string }>;
  corrosion_history?: Array<{ dayOffset: number; corrosion_percentage: number }>;
  provenance?: Provenance | 'real';
  human_approval?: boolean;
  is_real_asset?: boolean;
}): {
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  fused_risk_score: number;
  approval_required: boolean;
  attention_flag: boolean;
  dispatched: false;
  degradation_velocity_pct_per_day: number;
  estimated_days_to_critical: number | null;
  contributions: string[];
  recommendation: string;
  screening_disclaimer: string;
} {
  const visual = params.visual_report || {};
  const isRealAsset = params.is_real_asset !== false;
  const provenance = params.provenance || 'simulated';
  const humanApproval = Boolean(params.human_approval);
  const contributions: string[] = [];

  // Simulation Firewall: simulated telemetry cannot escalate a real workspace asset
  let effectiveAnomalies = params.anomalies || [];
  if (isRealAsset && provenance === 'simulated') {
    if (effectiveAnomalies.length > 0) {
      contributions.push(
        `Rule [SIMULATION_FIREWALL]: ${effectiveAnomalies.length} simulated anomaly alert(s) isolated from real asset risk scoring.`
      );
    }
    effectiveAnomalies = [];
  } else if (isRealAsset) {
    const blockedSim = effectiveAnomalies.filter(a => a.provenance === 'simulated').length;
    if (blockedSim > 0) {
      contributions.push(
        `Rule [SIMULATION_FIREWALL]: Excluded ${blockedSim} simulated anomaly record(s) from real asset assessment.`
      );
      effectiveAnomalies = effectiveAnomalies.filter(a => a.provenance !== 'simulated');
    }
  }

  const corrosion = Number(visual.corrosion_percentage || 0.0);
  const rec = String(visual.recommendation || 'PASS').toUpperCase();
  const confirmedVisual = Boolean(visual.confirmed) && ['CRITICAL', 'WARNING', 'HIGH', 'REJECTED'].includes(rec);
  const anomalyCount = effectiveAnomalies.length;
  const hasImportedReal = effectiveAnomalies.some(a => a.provenance === 'imported_real') || provenance === 'imported_real';

  let riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' = 'LOW';
  let approvalRequired = false;
  let attentionFlag = false;

  if (confirmedVisual && (rec === 'CRITICAL' || corrosion > 15.0)) {
    riskLevel = 'CRITICAL';
    approvalRequired = true;
    contributions.push('Rule [CRITICAL_VISUAL]: Confirmed visual finding with CRITICAL severity or >15% corrosion.');
  } else if (confirmedVisual && (rec === 'WARNING' || rec === 'HIGH')) {
    riskLevel = 'HIGH';
    approvalRequired = true;
    contributions.push('Rule [HIGH_VISUAL]: Confirmed visual finding with WARNING/HIGH severity.');
  } else if (hasImportedReal && !confirmedVisual) {
    attentionFlag = true;
    if (anomalyCount >= 2) {
      riskLevel = 'MEDIUM';
      contributions.push('Rule [IMPORTED_REAL_ATTENTION]: Persistent imported_real anomalies set attention flag; capped at MEDIUM pending visual confirmation.');
    } else {
      riskLevel = 'LOW';
      contributions.push('Rule [IMPORTED_REAL_LOW]: Minor imported_real anomaly flagged for attention.');
    }
  } else if (anomalyCount > 0 && provenance === 'simulated') {
    riskLevel = 'MEDIUM';
    contributions.push(`Rule [SIMULATED_ANOMALY]: ${anomalyCount} simulated telemetry anomalies detected in condition alerting sandbox.`);
  } else {
    riskLevel = 'LOW';
    contributions.push('Rule [BASELINE_SAFE]: No confirmed visual defects or active operational anomalies.');
  }

  // Guard: imported_real never reaches CRITICAL without confirmed visual or human approval
  if (hasImportedReal && riskLevel === 'CRITICAL' && !(confirmedVisual || humanApproval)) {
    riskLevel = 'HIGH';
    approvalRequired = true;
    contributions.push('Rule [IMPORTED_CRITICAL_GUARD]: Imported telemetry cannot reach CRITICAL without confirmed visual finding or human approval.');
  }

  // Predictive degradation velocity correlation
  const dailyRate = calculateDegradationVelocity(params.corrosion_history || []);
  const daysToCritical = projectTimeToCritical(corrosion, dailyRate, 10.0);

  if (confirmedVisual && dailyRate > 0 && daysToCritical !== null && daysToCritical <= 14.0) {
    if (riskLevel === 'LOW' || riskLevel === 'MEDIUM') {
      riskLevel = 'HIGH';
      approvalRequired = true;
    }
    contributions.push(
      `Rule [VELOCITY_ADVISORY]: Corrosion growth (${dailyRate}%/day) projects ${daysToCritical} days to 10% threshold.`
    );
  } else if ((params.corrosion_history || []).length < 3) {
    contributions.push('Rule [FORECAST_INSUFFICIENT_DATA]: Fewer than 3 dated inspections; degradation projection withheld.');
  }

  if (confirmedVisual && anomalyCount >= 2 && riskLevel === 'MEDIUM') {
    riskLevel = 'HIGH';
    approvalRequired = true;
    contributions.push('Rule [CORRELATED_MULTIMODAL]: Multi-metric telemetry anomalies corroborate confirmed visual defect.');
  }

  if (riskLevel === 'CRITICAL') {
    approvalRequired = true;
  }

  const baseScoreMap = { LOW: 15, MEDIUM: 45, HIGH: 72, CRITICAL: 92 };
  const fusedScore = Math.min(100, Number((baseScoreMap[riskLevel] + Math.min(18, corrosion * 0.6 + anomalyCount * 3)).toFixed(1)));

  const recommendation =
    riskLevel === 'CRITICAL'
      ? 'Human supervisor sign-off required before preparing emergency remediation work order (not auto-dispatched).'
      : riskLevel === 'HIGH'
        ? 'Schedule priority field inspection and prepare maintenance work order draft pending engineer review.'
        : riskLevel === 'MEDIUM'
          ? 'Condition attention advised: monitor telemetry trend and verify visually during next routine round.'
          : 'Normal baseline operation; continue scheduled periodic monitoring.';

  return {
    risk_level: riskLevel,
    fused_risk_score: fusedScore,
    approval_required: approvalRequired,
    attention_flag: attentionFlag,
    dispatched: false,
    degradation_velocity_pct_per_day: dailyRate,
    estimated_days_to_critical: daysToCritical,
    contributions,
    recommendation,
    screening_disclaimer: 'Computer-vision and statistical screening estimate only; human confirmation required before maintenance dispatch.'
  };
}
