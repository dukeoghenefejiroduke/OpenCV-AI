import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

export interface DetectionFinding {
  finding_id: string;
  region_id: string;
  inspection_id: string;
  detector_id: string;
  defect_category: string;
  bbox: [number, number, number, number]; // [x, y, w, h]
  confidence: number;
  position: string;
  percentage_of_image: number;
  experimental: boolean;
  requires_human_review: boolean;
  evidence_description: string;
  limitations: string[];
  human_review_status: 'unreviewed' | 'confirmed' | 'rejected' | 'needs_followup';
}

export interface AnalysisResult {
  width: number;
  height: number;
  corrosion_percentage: number;
  region_count: number;
  decision: 'PASS' | 'WARNING' | 'CRITICAL';
  action: string;
  recommendation: string;
  reason: string;
  screening_label: string;
  approval_required: boolean;
  image_quality: {
    verdict: 'PASS' | 'LOW_QUALITY';
    mean_brightness: number;
    sharpness_score: number;
  };
  findings: DetectionFinding[];
  detector_summaries: Array<{
    detector_id: string;
    status: string;
    findings: number;
  }>;
  annotatedBuffer: Buffer;
  originalBuffer: Buffer;
}

interface RawBox {
  x: number;
  y: number;
  w: number;
  h: number;
  pixels: number;
}

function rgbToHsvOpenCV(r: number, g: number, b: number): [number, number, number] {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;

  let h = 0;
  if (d !== 0) {
    if (max === rf) {
      h = 60 * (((gf - bf) / d) % 6);
    } else if (max === gf) {
      h = 60 * ((bf - rf) / d + 2);
    } else {
      h = 60 * ((rf - gf) / d + 4);
    }
    if (h < 0) h += 360;
  }

  const s = max === 0 ? 0 : d / max;
  const v = max;

  // OpenCV uint8 scaling: H in [0..180], S in [0..255], V in [0..255]
  return [Math.round(h / 2), Math.round(s * 255), Math.round(v * 255)];
}

function rgbToLabOpenCV(r: number, g: number, b: number): [number, number, number] {
  // Fast sRGB -> XYZ -> CIELAB (D65) -> OpenCV uint8 [L*255/100, a+128, b+128]
  let sr = r / 255;
  let sg = g / 255;
  let sb = b / 255;

  sr = sr > 0.04045 ? Math.pow((sr + 0.055) / 1.055, 2.4) : sr / 12.92;
  sg = sg > 0.04045 ? Math.pow((sg + 0.055) / 1.055, 2.4) : sg / 12.92;
  sb = sb > 0.04045 ? Math.pow((sb + 0.055) / 1.055, 2.4) : sb / 12.92;

  const x = (sr * 0.4124564 + sg * 0.3575761 + sb * 0.1804375) / 0.95047;
  const y = sr * 0.2126729 + sg * 0.7151522 + sb * 0.0721750;
  const z = (sr * 0.0193339 + sg * 0.1191920 + sb * 0.9503041) / 1.08883;

  const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x);
  const fy = f(y);
  const fz = f(z);

  const l = 116 * fy - 16;
  const a = 500 * (fx - fy);
  const bVal = 200 * (fy - fz);

  return [
    Math.max(0, Math.min(255, Math.round((l * 255) / 100))),
    Math.max(0, Math.min(255, Math.round(a + 128))),
    Math.max(0, Math.min(255, Math.round(bVal + 128)))
  ];
}

function getRegionPosition(bbox: [number, number, number, number], w: number, h: number): string {
  const cx = bbox[0] + bbox[2] / 2;
  const cy = bbox[1] + bbox[3] / 2;
  const vert = cy < h / 3 ? 'top' : cy > (2 * h) / 3 ? 'bottom' : 'middle';
  const horiz = cx < w / 3 ? 'left' : cx > (2 * w) / 3 ? 'right' : 'center';
  if (vert === 'middle' && horiz === 'center') return 'center';
  return `${vert}-${horiz}`;
}

function extractConnectedBoxes(
  mask: Uint8Array,
  w: number,
  h: number,
  minPixels: number,
  minDim: number
): RawBox[] {
  const visited = new Uint8Array(w * h);
  const boxes: RawBox[] = [];
  const stack = new Int32Array(w * h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const startIdx = y * w + x;
      if (mask[startIdx] === 0 || visited[startIdx] === 1) continue;

      let top = 0;
      stack[top++] = startIdx;
      visited[startIdx] = 1;

      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let count = 0;

      while (top > 0) {
        const idx = stack[--top];
        const cy = Math.floor(idx / w);
        const cx = idx - cy * w;
        count++;

        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        // 8-connected neighbors (stride 1)
        const nyStart = Math.max(0, cy - 1);
        const nyEnd = Math.min(h - 1, cy + 1);
        const nxStart = Math.max(0, cx - 1);
        const nxEnd = Math.min(w - 1, cx + 1);

        for (let ny = nyStart; ny <= nyEnd; ny++) {
          const rowOffset = ny * w;
          for (let nx = nxStart; nx <= nxEnd; nx++) {
            const nIdx = rowOffset + nx;
            if (mask[nIdx] === 1 && visited[nIdx] === 0) {
              visited[nIdx] = 1;
              stack[top++] = nIdx;
            }
          }
        }
      }

      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      if (count >= minPixels && bw >= minDim && bh >= minDim) {
        // Reject full-frame spanning artifacts unless genuinely huge
        if (!(bw >= w * 0.92 && bh >= h * 0.92)) {
          boxes.push({ x: minX, y: minY, w: bw, h: bh, pixels: count });
        }
      }
    }
  }
  return boxes;
}

function mergeOverlappingBoxes(boxes: RawBox[], iouThresh = 0.05, proximity = 14): RawBox[] {
  if (boxes.length === 0) return [];
  let current = boxes.map(b => ({ ...b }));
  let changed = true;

  while (changed) {
    changed = false;
    const next: RawBox[] = [];
    const merged = new Array<boolean>(current.length).fill(false);

    for (let i = 0; i < current.length; i++) {
      if (merged[i]) continue;
      let a = { ...current[i] };

      for (let j = i + 1; j < current.length; j++) {
        if (merged[j]) continue;
        const b = current[j];

        const ix1 = Math.max(a.x, b.x);
        const iy1 = Math.max(a.y, b.y);
        const ix2 = Math.min(a.x + a.w, b.x + b.w);
        const iy2 = Math.min(a.y + a.h, b.y + b.h);
        const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
        const union = a.w * a.h + b.w * b.h - inter;
        const iou = union > 0 ? inter / union : 0;

        const distX = Math.abs(a.x + a.w / 2 - (b.x + b.w / 2));
        const distY = Math.abs(a.y + a.h / 2 - (b.y + b.h / 2));
        const near =
          distX < (a.w + b.w) / 2 + proximity && distY < (a.h + b.h) / 2 + proximity;

        if (iou > iouThresh || near) {
          const nx = Math.min(a.x, b.x);
          const ny = Math.min(a.y, b.y);
          const nw = Math.max(a.x + a.w, b.x + b.w) - nx;
          const nh = Math.max(a.y + a.h, b.y + b.h) - ny;
          a = { x: nx, y: ny, w: nw, h: nh, pixels: a.pixels + b.pixels };
          merged[j] = true;
          changed = true;
        }
      }
      next.push(a);
    }
    current = next;
  }

  return current;
}

function dilateBinaryMask(mask: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] === 0) continue;
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(h - 1, y + radius);
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(w - 1, x + radius);
      for (let ny = y0; ny <= y1; ny++) {
        const row = ny * w;
        for (let nx = x0; nx <= x1; nx++) {
          out[row + nx] = 1;
        }
      }
    }
  }
  return out;
}

function erodeBinaryMask(mask: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = radius; y < h - radius; y++) {
    for (let x = radius; x < w - radius; x++) {
      let keep = 1;
      for (let ny = y - radius; ny <= y + radius && keep; ny++) {
        const row = ny * w;
        for (let nx = x - radius; nx <= x + radius; nx++) {
          if (mask[row + nx] === 0) {
            keep = 0;
            break;
          }
        }
      }
      out[y * w + x] = keep;
    }
  }
  return out;
}

function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export async function analyzeInspectionImage(params: {
  input: Buffer | string;
  inspectionId: string;
  surface?: string;
}): Promise<AnalysisResult> {
  const { input, inspectionId, surface = 'bare_or_painted_metal' } = params;

  // Normalize EXIF orientation and convert to high-quality JPEG for consistent frontend delivery
  const baseSharp = sharp(input).rotate();
  const originalBuffer = await baseSharp.clone().jpeg({ quality: 92 }).toBuffer();
  const origMeta = await sharp(originalBuffer).metadata();
  const origW = origMeta.width || 1280;
  const origH = origMeta.height || 720;

  // Resize for fast pixel-level CV processing (max 480px on longest side)
  const maxProcDim = 480;
  const scale = Math.min(1, maxProcDim / Math.max(origW, origH));
  const procW = Math.max(64, Math.round(origW * scale));
  const procH = Math.max(64, Math.round(origH * scale));

  const { data: rgb } = await sharp(originalBuffer)
    .resize(procW, procH, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const totalProcPixels = procW * procH;
  const gray = new Float32Array(totalProcPixels);
  let brightnessSum = 0;

  for (let i = 0; i < totalProcPixels; i++) {
    const idx = i * 3;
    const y = 0.299 * rgb[idx] + 0.587 * rgb[idx + 1] + 0.114 * rgb[idx + 2];
    gray[i] = y;
    brightnessSum += y;
  }

  const meanBrightness = brightnessSum / totalProcPixels;

  // Compute Laplacian gradient magnitude per pixel
  const grad = new Float32Array(totalProcPixels);
  let gradSum = 0;
  for (let y = 1; y < procH - 1; y++) {
    for (let x = 1; x < procW - 1; x++) {
      const i = y * procW + x;
      const lap =
        Math.abs(
          4 * gray[i] -
            gray[i - 1] -
            gray[i + 1] -
            gray[i - procW] -
            gray[i + procW]
        );
      grad[i] = lap;
      gradSum += lap;
    }
  }
  const sharpnessScore = gradSum / totalProcPixels;
  const isLowQuality = meanBrightness < 25 || sharpnessScore < 4.5;

  const isExcludedSurface =
    surface === 'brick_concrete_wood' || surface === 'other_metal_non_steel';

  const rustMaskRaw = new Uint8Array(totalProcPixels);
  const greenPaintMaskRaw = new Uint8Array(totalProcPixels);
  const leakMaskRaw = new Uint8Array(totalProcPixels);
  const darkCrackMaskRaw = new Uint8Array(totalProcPixels);

  let rustPixelCount = 0;

  for (let y = 1; y < procH - 1; y++) {
    for (let x = 1; x < procW - 1; x++) {
      const i = y * procW + x;
      const idx = i * 3;
      const r = rgb[idx];
      const g = rgb[idx + 1];
      const b = rgb[idx + 2];

      const [h, s, v] = rgbToHsvOpenCV(r, g, b);
      const [, aStar, bStar] = rgbToLabOpenCV(r, g, b);
      const gMag = grad[i];

      if (!isExcludedSurface) {
        // 1. Standard orange/brown rust + dark brown/ochre rough rust + specular glare rust
        const isStandardRust = h >= 4 && h <= 25 && s >= 55 && v >= 45;
        const isDarkBrownRust =
          gMag > 8 && s >= 35 && aStar > 128 && bStar > 132 && v >= 30 && v <= 130;
        const isSpecularRust =
          gMag > 10 && (aStar > 132 || bStar > 136) && v > 180;

        if (isStandardRust || isDarkBrownRust || isSpecularRust) {
          rustMaskRaw[i] = 1;
          rustPixelCount++;
        }
      }

      // 2. Peeling green paint / coating delamination boundary
      if (h >= 35 && h <= 85 && s >= 30 && v >= 30 && gMag > 14) {
        greenPaintMaskRaw[i] = 1;
      }

      // 3. Leakage / dark vertical streak
      if (v < 75 && s < 60 && gMag > 4) {
        leakMaskRaw[i] = 1;
      }

      // 4. Dark linear crack candidate (significantly darker than local neighbors)
      const localAvg =
        (gray[i - 1] + gray[i + 1] + gray[i - procW] + gray[i + procW]) / 4;
      if (localAvg - gray[i] > 22 && s < 75) {
        darkCrackMaskRaw[i] = 1;
      }
    }
  }

  const findings: DetectionFinding[] = [];
  const scaleX = origW / procW;
  const scaleY = origH / procH;

  let corrosionPercentage = 0;

  if (!isExcludedSurface) {
    // Morphological open to clean isolated pixel noise while keeping distinct pipe heads separated
    const rustOpened = dilateBinaryMask(erodeBinaryMask(rustMaskRaw, procW, procH, 1), procW, procH, 1);

    const minRustPixels = Math.max(18, Math.round(totalProcPixels * 0.0006));
    const rawRustBoxes = extractConnectedBoxes(rustOpened, procW, procH, minRustPixels, 6);
    const mergedRustBoxes = mergeOverlappingBoxes(rawRustBoxes, 0.05, 8)
      .sort((a, b) => b.w * b.h - a.w * a.h)
      .slice(0, 8);

    const validRustPixels = mergedRustBoxes.reduce((acc, b) => acc + b.pixels, 0);
    corrosionPercentage = parseFloat(
      Math.min(100, (validRustPixels / totalProcPixels) * 100).toFixed(2)
    );

    mergedRustBoxes.forEach((box, idx) => {
      const origBox: [number, number, number, number] = [
        Math.round(box.x * scaleX),
        Math.round(box.y * scaleY),
        Math.max(12, Math.round(box.w * scaleX)),
        Math.max(12, Math.round(box.h * scaleY))
      ];
      const pctOfImg = parseFloat((((box.w * box.h) / totalProcPixels) * 100).toFixed(2));
      const conf = parseFloat(Math.min(0.96, 0.72 + Math.min(0.24, pctOfImg * 0.025)).toFixed(2));
      const pos = getRegionPosition(origBox, origW, origH);

      findings.push({
        finding_id: `find_cor_${inspectionId}_${idx + 1}`,
        region_id: `COR-${String(idx + 1).padStart(3, '0')}`,
        inspection_id: inspectionId,
        detector_id: 'corrosion_detector',
        defect_category: 'corrosion',
        bbox: origBox,
        confidence: isLowQuality ? 0.35 : conf,
        position: pos,
        percentage_of_image: pctOfImg,
        experimental: false,
        requires_human_review: corrosionPercentage >= 10.0 || isLowQuality,
        evidence_description: `Detected ${pctOfImg}% localized oxidation/corrosion patch at ${pos}.`,
        limitations: isLowQuality
          ? ['Optical screening estimate only', 'Low image quality warning']
          : ['Optical screening estimate only'],
        human_review_status: 'unreviewed'
      });
    });
  }

  // Experimental Coating Delamination Findings (COT-*)
  const coatClosed = erodeBinaryMask(dilateBinaryMask(greenPaintMaskRaw, procW, procH, 2), procW, procH, 1);
  const rawCoatBoxes = extractConnectedBoxes(coatClosed, procW, procH, Math.max(15, Math.round(totalProcPixels * 0.0005)), 6);
  const mergedCoatBoxes = mergeOverlappingBoxes(rawCoatBoxes, 0.08, 12)
    .sort((a, b) => b.w * b.h - a.w * a.h)
    .slice(0, 3);

  mergedCoatBoxes.forEach((box, idx) => {
    const origBox: [number, number, number, number] = [
      Math.round(box.x * scaleX),
      Math.round(box.y * scaleY),
      Math.max(12, Math.round(box.w * scaleX)),
      Math.max(12, Math.round(box.h * scaleY))
    ];
    const pctOfImg = parseFloat((((box.w * box.h) / totalProcPixels) * 100).toFixed(2));
    const pos = getRegionPosition(origBox, origW, origH);
    const conf = parseFloat(Math.min(0.91, 0.68 + pctOfImg * 0.03).toFixed(2));

    findings.push({
      finding_id: `find_cot_${inspectionId}_${idx + 1}`,
      region_id: `COT-${String(idx + 1).padStart(3, '0')}`,
      inspection_id: inspectionId,
      detector_id: 'experimental_coating_detector',
      defect_category: 'coating_damage',
      bbox: origBox,
      confidence: conf,
      position: pos,
      percentage_of_image: pctOfImg,
      experimental: true,
      requires_human_review: true,
      evidence_description: `Coating delamination / peeling paint boundary detected at ${pos}.`,
      limitations: ['Experimental detector — human verification required'],
      human_review_status: 'unreviewed'
    });
  });

  // Experimental Crack Findings (CRK-*)
  const rawCrackBoxes = extractConnectedBoxes(darkCrackMaskRaw, procW, procH, 14, 4)
    .filter(b => Math.max(b.w, b.h) / Math.max(1, Math.min(b.w, b.h)) >= 3.2 && Math.max(b.w, b.h) >= 24)
    .sort((a, b) => b.w * b.h - a.w * a.h)
    .slice(0, 2);

  rawCrackBoxes.forEach((box, idx) => {
    const origBox: [number, number, number, number] = [
      Math.round(box.x * scaleX),
      Math.round(box.y * scaleY),
      Math.max(8, Math.round(box.w * scaleX)),
      Math.max(8, Math.round(box.h * scaleY))
    ];
    const pctOfImg = parseFloat((((box.w * box.h) / totalProcPixels) * 100).toFixed(2));
    const pos = getRegionPosition(origBox, origW, origH);
    findings.push({
      finding_id: `find_crk_${inspectionId}_${idx + 1}`,
      region_id: `CRK-${String(idx + 1).padStart(3, '0')}`,
      inspection_id: inspectionId,
      detector_id: 'experimental_crack_detector',
      defect_category: 'crack',
      bbox: origBox,
      confidence: 0.72,
      position: pos,
      percentage_of_image: pctOfImg,
      experimental: true,
      requires_human_review: true,
      evidence_description: `Elongated dark linear discontinuity (crack candidate) at ${pos}.`,
      limitations: ['Experimental crack detector — confirm physically'],
      human_review_status: 'unreviewed'
    });
  });

  // Experimental Leakage Findings (LEK-*)
  const rawLeakBoxes = extractConnectedBoxes(leakMaskRaw, procW, procH, 30, 5)
    .filter(b => b.h / Math.max(1, b.w) >= 2.2 && b.h >= 24)
    .sort((a, b) => b.w * b.h - a.w * a.h)
    .slice(0, 2);

  rawLeakBoxes.forEach((box, idx) => {
    const origBox: [number, number, number, number] = [
      Math.round(box.x * scaleX),
      Math.round(box.y * scaleY),
      Math.max(8, Math.round(box.w * scaleX)),
      Math.max(16, Math.round(box.h * scaleY))
    ];
    const pctOfImg = parseFloat((((box.w * box.h) / totalProcPixels) * 100).toFixed(2));
    const pos = getRegionPosition(origBox, origW, origH);
    findings.push({
      finding_id: `find_lek_${inspectionId}_${idx + 1}`,
      region_id: `LEK-${String(idx + 1).padStart(3, '0')}`,
      inspection_id: inspectionId,
      detector_id: 'experimental_leakage_detector',
      defect_category: 'leakage',
      bbox: origBox,
      confidence: 0.71,
      position: pos,
      percentage_of_image: pctOfImg,
      experimental: true,
      requires_human_review: true,
      evidence_description: `Vertical dark low-saturation drip candidate at ${pos}.`,
      limitations: ['Experimental leakage screening — verify moisture'],
      human_review_status: 'unreviewed'
    });
  });

  // Decision logic matching CementSight thresholds
  let decision: 'PASS' | 'WARNING' | 'CRITICAL' = 'PASS';
  let action = 'Routine monitoring; no immediate intervention required';
  let recommendation = 'Continue scheduled visual inspection cycle.';
  let reason = `Corrosion coverage (${corrosionPercentage}%) is within acceptable baseline (< 2.0%).`;
  let approvalRequired = false;

  if (isExcludedSurface) {
    const surfaceDesc =
      surface === 'brick_concrete_wood'
        ? 'brick, concrete, or wood'
        : 'non-steel metal (copper, brass, aluminium)';
    reason = `Corrosion screening excluded for ${surfaceDesc}.`;
    recommendation = `Manual material-specific inspection recommended for ${surfaceDesc}.`;
  } else if (corrosionPercentage >= 10.0) {
    decision = 'CRITICAL';
    action = 'Immediate Supervisor Sign-Off & Ultrasonic Thickness Gauging';
    recommendation = 'CRITICAL corrosion extent (>= 10.0%). Dispatch maintenance crew and verify wall thickness.';
    reason = `Critical-level corrosion detected: ${corrosionPercentage}% estimated surface coverage across ${findings.filter(f => f.detector_id === 'corrosion_detector').length} regions.`;
    approvalRequired = true;
  } else if (corrosionPercentage >= 2.0) {
    decision = 'WARNING';
    action = 'Routine maintenance inspection & protective coating review';
    recommendation = 'WARNING condition (2.0% - 10.0%). Schedule surface preparation and re-coating.';
    reason = `Warning-level condition detected: ${corrosionPercentage}% corrosion coverage within warning threshold (2.0% - 10.0%).`;
    approvalRequired = false;
  }

  const screeningLabel =
    surface === 'not_sure'
      ? 'WARNING: surface uncertain — rust-coloured brick/wood can trigger false alarms'
      : 'screening aid for steel; optical estimate, not certified physical measurement';

  // Render crisp SVG bounding-box & HUD overlay directly on top of the analyzed image
  const strokeWidth = Math.max(2, Math.round(Math.min(origW, origH) * 0.004));
  const fontSize = Math.max(13, Math.round(Math.min(origW, origH) * 0.022));
  const padX = Math.round(fontSize * 0.45);
  const tagH = Math.round(fontSize * 1.45);

  const colorMap: Record<string, string> = {
    corrosion: '#f59e0b',       // Amber
    coating_damage: '#10b981',  // Emerald
    crack: '#3b82f6',           // Blue
    leakage: '#06b6d4'          // Cyan
  };

  const boxesSvg = findings
    .map(f => {
      const [bx, by, bw, bh] = f.bbox;
      const color = colorMap[f.defect_category] || '#f59e0b';
      const label = `${f.region_id} ${Math.round(f.confidence * 100)}%`;
      const tagW = Math.round(label.length * fontSize * 0.62 + padX * 2);
      const tagY = by - tagH - 2 >= 0 ? by - tagH - 2 : by + 2;
      const tagX = Math.max(2, Math.min(origW - tagW - 2, bx));

      return `
        <rect x="${bx}" y="${by}" width="${bw}" height="${bh}" fill="${color}" fill-opacity="0.12" stroke="${color}" stroke-width="${strokeWidth}" rx="3" />
        <rect x="${tagX}" y="${tagY}" width="${tagW}" height="${tagH}" fill="${color}" rx="3" />
        <text x="${tagX + padX}" y="${tagY + tagH - Math.round(fontSize * 0.35)}" fill="#050811" font-family="monospace" font-weight="bold" font-size="${fontSize}px">${escapeXml(label)}</text>
      `;
    })
    .join('\n');

  const bannerFont = Math.max(12, Math.round(Math.min(origW, origH) * 0.02));
  const bannerH = Math.round(bannerFont * 2.1);
  const decColor = decision === 'CRITICAL' ? '#f43f5e' : decision === 'WARNING' ? '#f59e0b' : '#10b981';
  const bannerText = `CEMENTSIGHT AI  |  ${inspectionId}  |  ${decision}  |  CORROSION: ${corrosionPercentage}%  |  REGIONS: ${findings.length}`;

  const svgOverlay = `<svg xmlns="http://www.w3.org/2000/svg" width="${origW}" height="${origH}" viewBox="0 0 ${origW} ${origH}">
    ${boxesSvg}
    <rect x="0" y="${origH - bannerH}" width="${origW}" height="${bannerH}" fill="#050811" fill-opacity="0.82" />
    <rect x="0" y="${origH - bannerH}" width="6" height="${bannerH}" fill="${decColor}" />
    <text x="${Math.round(bannerFont * 1.1)}" y="${origH - Math.round(bannerFont * 0.65)}" fill="#f8fafc" font-family="monospace" font-weight="bold" font-size="${bannerFont}px">${escapeXml(bannerText)}</text>
  </svg>`;

  const annotatedBuffer = await sharp(originalBuffer)
    .composite([{ input: Buffer.from(svgOverlay), top: 0, left: 0 }])
    .jpeg({ quality: 90 })
    .toBuffer();

  // Summarize counts per detector
  const detectorCounts: Record<string, number> = {};
  for (const f of findings) {
    detectorCounts[f.detector_id] = (detectorCounts[f.detector_id] || 0) + 1;
  }

  const detectorSummaries = Object.entries(detectorCounts).map(([detector_id, count]) => ({
    detector_id,
    status: 'completed',
    findings: count
  }));

  return {
    width: origW,
    height: origH,
    corrosion_percentage: corrosionPercentage,
    region_count: findings.length,
    decision,
    action,
    recommendation,
    reason,
    screening_label: screeningLabel,
    approval_required: approvalRequired,
    image_quality: {
      verdict: isLowQuality ? 'LOW_QUALITY' : 'PASS',
      mean_brightness: parseFloat(meanBrightness.toFixed(1)),
      sharpness_score: parseFloat(sharpnessScore.toFixed(1))
    },
    findings,
    detector_summaries: detectorSummaries,
    annotatedBuffer,
    originalBuffer
  };
}

export function saveAnalysisImages(
  outputDir: string,
  inspectionId: string,
  annotatedBuffer: Buffer,
  originalBuffer: Buffer
): { annotatedPath: string; originalPath: string } {
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  const annotatedPath = path.join(outputDir, `live_annotated_${inspectionId}.jpg`);
  const originalPath = path.join(outputDir, `original_${inspectionId}.jpg`);
  fs.writeFileSync(annotatedPath, annotatedBuffer);
  fs.writeFileSync(originalPath, originalBuffer);
  return { annotatedPath, originalPath };
}
