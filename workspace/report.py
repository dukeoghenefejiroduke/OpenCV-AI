"""Self-contained HTML inspection report generator.

Reads only from the workspace store. Uses stdlib only (no Jinja2).
Embeds original image as data-URI. Marks machine suggestions distinctly
from human-confirmed findings. Never claims certification or engineering approval.
"""

import base64
import html
import os
from typing import List

from workspace.models import Asset, Finding, Inspection, PriorityScore
from workspace.store import WorkspaceStore


def _data_uri(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _mime_for(ext: str) -> str:
    return {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "application/octet-stream")


def generate_html_report(
    store: WorkspaceStore,
    inspection: Inspection,
    findings: List[Finding],
    priority: PriorityScore,
    out_path: str,
) -> str:
    asset = store.get_asset(inspection.asset_id)
    img_path = store.image_path(inspection.image_sha256, inspection.image_ext)
    img_uri = _data_uri(img_path, _mime_for(inspection.image_ext))

    annotated_path = os.path.join("data", "outputs", f"live_annotated_{inspection.inspection_id}.jpg")
    annotated_uri = _data_uri(annotated_path, "image/jpeg") if os.path.exists(annotated_path) else img_uri

    auth_rep = inspection.authoritative_report or {}
    corrosion_pct = auth_rep.get("corrosion_percentage", 0.0)
    region_count = auth_rep.get("region_count", 0)
    surface = auth_rep.get("surface", "bare_or_painted_metal")
    if surface in ("brick_concrete_wood", "other_metal_non_steel") or (corrosion_pct == 0.0 and region_count == 0):
        verdict = "No corrosion candidates found. This does not certify the asset."
    else:
        verdict = "WARNING" if corrosion_pct > 2.0 else "PASS"
        if corrosion_pct > 10.0:
            verdict = "CRITICAL"

    approval_required = "Yes" if verdict in ("WARNING", "CRITICAL") else "No"
    suggested_action = auth_rep.get("recommendation", "Routine maintenance inspection & monitoring.")

    total_findings = len(findings)
    confirmed_count = sum(1 for f in findings if f.human_review_status == "confirmed")
    rejected_count = sum(1 for f in findings if f.human_review_status == "rejected")
    followup_count = sum(1 for f in findings if f.human_review_status == "needs_followup")
    unreviewed_count = sum(1 for f in findings if f.human_review_status == "unreviewed")

    rows = []
    for f in findings:
        human = f.human_review_status
        cls = "confirmed" if human == "confirmed" else ("rejected" if human == "rejected" else "machine")
        status_label = "Confirmed" if human == "confirmed" else ("Rejected" if human == "rejected" else ("Follow-up" if human == "needs_followup" else "Unreviewed"))
        conf = "n/a" if f.confidence is None else f"{f.confidence:.2f} (heuristic score, not a probability)"
        bbox = f.bbox if f.bbox else "n/a"
        rows.append(f"""
        <tr class="{cls}">
          <td><span class="status-badge {cls}">{html.escape(status_label)}</span></td>
          <td>{html.escape(f.detector_id)}</td>
          <td>{html.escape(f.defect_category)}</td>
          <td>{html.escape(str(bbox))}</td>
          <td>{html.escape(conf)}</td>
          <td>{html.escape(f.review_notes or '')}</td>
        </tr>""")

    rationale_html = "".join(f"<li>{html.escape(r)}</li>" for r in (priority.rationale if priority else []))

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supervisor Inspection Report - {html.escape(inspection.inspection_id)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 1rem; }}
  .wrap {{ max-width: 900px; margin: 0 auto; background: #fff; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  h1 {{ font-size: 1.25rem; margin: 0 0 0.25rem; }}
  .banner {{ background: #fef3c7; border: 1px solid #f59e0b; color: #78350f; padding: 0.75rem 1rem; border-radius: 8px; margin: 1rem 0; font-size: 0.9rem; }}
  .not-cert {{ background: #fee2e2; border-color: #ef4444; color: #7f1d1d; }}
  .scope-box {{ background: #f8fafc; border: 1px solid #cbd5e1; color: #334155; padding: 0.75rem 1rem; border-radius: 8px; margin: 1rem 0; font-size: 0.85rem; line-height: 1.4; }}
  .summary-counts {{ font-size: 0.85rem; font-family: monospace; background: #f1f5f9; padding: 0.5rem 0.75rem; border-radius: 6px; margin: 0.75rem 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 1rem; }}
  th, td {{ padding: 0.5rem; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: top; }}
  th {{ background: #f1f5f9; }}
  tr.machine td {{ background: #fefce8; }}
  tr.confirmed td {{ background: #ecfdf5; }}
  tr.rejected td {{ background: #f1f5f9; color: #64748b; text-decoration: line-through; }}
  .status-badge {{ display: inline-block; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.75rem; font-weight: bold; }}
  .status-badge.machine {{ background: #fef08a; color: #713f12; }}
  .status-badge.confirmed {{ background: #bbf7d0; color: #166534; }}
  .status-badge.rejected {{ background: #e2e8f0; color: #475569; }}
  .pill {{ display:inline-block; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
  .img-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }}
  img {{ max-width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; }}
  ul {{ padding-left: 1.25rem; }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .wrap {{ box-shadow: none; border: none; padding: 0; max-width: 100%; }}
  }}
</style></head>
<body><div class="wrap">
<h1>Supervisor Inspection Report — {html.escape(inspection.inspection_id)}</h1>
<div class="banner not-cert"><strong>Not a certification and not engineering approval.</strong> This report documents provisional machine suggestions and human review notes. Final maintenance decisions must be made by a qualified human.</div>

<div class="scope-box"><strong>Scope & Limitations:</strong> This report documents provisional machine suggestions and human review records. It does not constitute certification or engineering approval. Optical screening ({corrosion_pct}% computer-vision screening estimate for steel; rust-coloured materials such as brick and wood can trigger false alarms) cannot establish internal structural integrity, corrosion depth, or remaining pipe lifespan.</div>

<div class="summary-counts">
  Quality Verdict: <strong>{html.escape(verdict)}</strong> | Corrosion Screening Estimate: {corrosion_pct}% | Surface Type: <strong>{html.escape(auth_rep.get('surface', 'bare_or_painted_metal'))}</strong> | Regions: {region_count} | Approval Required: <strong>{html.escape(approval_required)}</strong><br>
  Review State: Confirmed ({confirmed_count}) / Unreviewed ({unreviewed_count}) / Rejected ({rejected_count}) | Software Version: CementSight AI v1.3.0
</div>

<h2 style="font-size:1rem;">Asset Context</h2>
<p>
<strong>Asset ID:</strong> {html.escape(asset.asset_id) if asset else '(unknown)'}<br>
<strong>Location:</strong> {html.escape(asset.location) if asset else '(unknown)'}<br>
<strong>Pipe material:</strong> {html.escape(asset.pipe_material) if asset else '(unknown)'}<br>
<strong>Criticality:</strong> {html.escape(asset.criticality) if asset else '(unknown)'}
</p>

<h2 style="font-size:1rem;">Inspection Evidence & Overlays</h2>
<p><strong>Date:</strong> {html.escape(inspection.inspection_date)} | <strong>Original Image SHA-256:</strong> <code>{html.escape(inspection.image_sha256)}</code></p>
<div class="img-grid">
  <div>
    <div style="font-size:0.8rem; font-weight:bold; margin-bottom:0.25rem;">Original Photo</div>
    <img src="{img_uri}" alt="Original Inspection Image">
  </div>
  <div>
    <div style="font-size:0.8rem; font-weight:bold; margin-bottom:0.25rem;">Annotated Overlay</div>
    <img src="{annotated_uri}" alt="Annotated Overlay">
  </div>
</div>

<h2 style="font-size:1rem;">Suggestions</h2>
<p><strong>Suggested Action:</strong> {html.escape(suggested_action)}</p>
<ul>{rationale_html}</ul>

<h2 style="font-size:1rem;">Machine Findings vs. Human Review</h2>
<table>
<thead><tr><th>Status</th><th>Detector</th><th>Category</th><th>BBox (px)</th><th>Confidence / score</th><th>Notes</th></tr></thead>
<tbody>
{''.join(rows) if rows else '<tr><td colspan="6">No findings.</td></tr>'}
</tbody></table>

<h2 style="font-size:1rem;">Inspector Notes</h2>
<p>{html.escape(inspection.inspector_notes or '(none)')}</p>

<h2 style="font-size:1rem;">Limitations & Disclaimers</h2>
<ul>
<li>Detector scores are heuristic and are not calibrated probabilities.</li>
<li>RGB imagery cannot prove active water leakage, internal structural integrity, corrosion depth, or remaining pipe lifespan.</li>
<li>Pixel measurements are not physical dimensions unless calibrated with known references.</li>
<li>This report does not constitute certification, engineering approval, or authorization to operate.</li>
</ul>

<div style="margin-top: 3rem; display: flex; justify-content: space-between; border-top: 1px solid #000; padding-top: 1rem; page-break-inside: avoid;">
  <div><strong>Supervisor Signature:</strong> ___________________________</div>
  <div><strong>Date:</strong> ___________________</div>
</div>

<p style="margin-top:2rem; font-size:0.75rem; color:#64748b;">
Generated locally by CementSight AI v1.3.0. No image data transmitted to external services.
</p>
</div></body></html>
"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path
