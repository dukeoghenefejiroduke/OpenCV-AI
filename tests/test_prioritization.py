import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workspace.models import Asset, Inspection, Finding
from workspace.prioritization import score, CATEGORIES


def _asset(crit="normal"):
    return Asset(asset_id="A", location="L", criticality=crit)


def _insp(corr=0.0, iid="INS-1"):
    return Inspection(
        inspection_id=iid, asset_id="A", inspection_date="2026-11-05",
        image_sha256="0" * 64, image_ext="jpg",
        detector_results={}, authoritative_report={"corrosion_percentage": corr},
    )


def _f(cat, status):
    return Finding(
        finding_id=f"F-{cat}-{status}", inspection_id="INS-1",
        detector_id="d", defect_category=cat, bbox=None, confidence=0.5,
        evidence_description="", limitations=[], human_review_status=status,
    )


def test_rejected_only_is_routine():
    s = score(_asset(), _insp(), [_f("crack", "rejected")], history=[])
    assert s.category == "routine_review"


def test_unreviewed_does_not_reach_urgent():
    s = score(_asset("critical"), _insp(corr=15.0), [_f("leakage", "unreviewed")], history=[])
    assert s.category in ("routine_review", "follow_up")


def test_confirmed_leakage_plus_critical_asset_reaches_urgent():
    s = score(_asset("critical"), _insp(corr=12.0), [_f("leakage", "confirmed")], history=[])
    assert s.category == "urgent_human_assessment"


def test_confirmed_crack_with_flag_reaches_urgent():
    s = score(_asset("critical"), _insp(corr=12.0), [_f("crack", "confirmed")], history=[], crack_enabled_for_priority=True)
    assert s.category == "urgent_human_assessment"


def test_rationale_always_nonempty():
    s = score(_asset(), _insp(), [], history=[])
    assert s.rationale
    assert any("recommendation only" in r for r in s.rationale)


def test_all_categories_reachable():
    seen = set()
    for crit, corr, cat_status in [
        ("normal", 0.0, "rejected"),
        ("normal", 2.0, "unreviewed"),
        ("high", 5.0, "confirmed"),
        ("critical", 12.0, "confirmed"),
    ]:
        s = score(_asset(crit), _insp(corr), [_f("crack", cat_status)], history=[])
        seen.add(s.category)
    assert len(seen) >= 3


def test_rejected_findings_never_raise_priority_across_categories():
    for crit in ["low", "normal", "high", "critical"]:
        s_nofind = score(_asset(crit), _insp(0.0), [], history=[])
        s_reject = score(_asset(crit), _insp(0.0), [_f("crack", "rejected"), _f("leakage", "rejected")], history=[])
        assert s_reject.score == s_nofind.score
        assert s_reject.category == s_nofind.category


def test_repeated_confirmations_on_same_category_accumulate():
    a = _asset("high")
    insp = _insp(corr=3.0)
    s1 = score(a, insp, [_f("leakage", "confirmed")], history=[])
    s2 = score(a, insp, [_f("leakage", "confirmed"), Finding(finding_id="F-leakage-confirmed-2", inspection_id="INS-1", detector_id="d", defect_category="leakage", bbox=None, confidence=0.5, evidence_description="", limitations=[], human_review_status="confirmed")], history=[])
    assert s2.score > s1.score


def test_rationale_always_ends_with_human_decision_reminder():
    for crit in ["normal", "critical"]:
        for corr in [0.0, 10.0]:
            s = score(_asset(crit), _insp(corr), [_f("crack", "confirmed")], history=[])
            assert s.rationale
            assert "recommendation only" in s.rationale[-1]


def test_unreviewed_only_never_reaches_urgent_or_priority():
    for crit in ["normal", "high", "critical"]:
        for corr in [0.0, 5.0, 20.0]:
            s = score(_asset(crit), _insp(corr), [_f("crack", "unreviewed"), _f("leakage", "unreviewed")], history=[])
            assert s.category in ("routine_review", "follow_up")