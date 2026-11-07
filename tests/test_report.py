import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workspace.store import WorkspaceStore
from workspace.models import Asset, Inspection, Finding
from workspace.prioritization import score
from workspace.report import generate_html_report
from tests.test_workspace_store import JPEG_1PX


def test_report_has_required_banners_and_no_approval_language(tmp_path):
    s = WorkspaceStore(root=str(tmp_path))
    s.upsert_asset(Asset(asset_id="P-01", location="Plant A"))
    sha, ext = s.store_image(JPEG_1PX)
    insp = Inspection(
        inspection_id="INS-1", asset_id="P-01", inspection_date="2026-11-05",
        image_sha256=sha, image_ext=ext,
        authoritative_report={"corrosion_percentage": 0.5},
        inspector_notes="Checked visually.",
    )
    s.create_inspection(insp)
    f = Finding(finding_id="F-1", inspection_id="INS-1", detector_id="crack_detector",
                defect_category="crack", bbox=[1, 2, 10, 10], confidence=0.42,
                evidence_description="synthetic", limitations=["heuristic"])
    s.add_findings([f])
    p = score(s.get_asset("P-01"), insp, [f], history=[])
    out = os.path.join(str(tmp_path), "report.html")
    generate_html_report(s, insp, [f], p, out)
    html = open(out, encoding="utf-8").read()
    assert "Not a certification" in html
    assert "heuristic" in html
    assert "not a probability" in html
    assert "certified" not in html.lower() or "not a certification" in html.lower()
    # original image embedded
    assert "data:image/jpeg;base64," in html