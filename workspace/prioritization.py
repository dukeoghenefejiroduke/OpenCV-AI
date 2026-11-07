"""Transparent maintenance prioritization.

Never autonomous. Never declares an asset unsafe. Every score is a recommendation
for a human, with an ordered rationale list.

Inputs:
  - findings (with human_review_status)
  - asset criticality
  - recurrence (same category on same asset in the last N inspections)
  - authoritative (corrosion) report — treated as one input, not the sole signal

Output: PriorityScore with a category and per-factor rationale.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from workspace.models import Asset, Finding, Inspection, PriorityScore


CATEGORIES = (
    "routine_review",
    "follow_up",
    "priority_maintenance",
    "urgent_human_assessment",
)


def score(
    asset: Asset,
    inspection: Inspection,
    findings: List[Finding],
    history: List[Inspection],
    recurrence_window_days: int = 365,
) -> PriorityScore:
    rationale: List[str] = []
    factors: Dict[str, object] = {}
    points = 0

    confirmed = [f for f in findings if f.human_review_status == "confirmed"]
    rejected = [f for f in findings if f.human_review_status == "rejected"]
    unreviewed = [f for f in findings if f.human_review_status == "unreviewed"]
    needs_fu = [f for f in findings if f.human_review_status == "needs_followup"]

    factors["confirmed_count"] = len(confirmed)
    factors["unreviewed_count"] = len(unreviewed)
    factors["rejected_count"] = len(rejected)
    factors["needs_followup_count"] = len(needs_fu)

    if rejected:
        rationale.append(f"{len(rejected)} finding(s) were rejected by human review and do not contribute to priority.")

    # 1. Confirmed findings — only human-confirmed evidence pushes priority up.
    for f in confirmed:
        # severity: leak/spalling/crack more consequential than coating
        sev = 2 if f.defect_category in ("leakage", "spalling", "crack") else 1
        points += sev
        rationale.append(
            f"Human-confirmed {f.defect_category} finding ({f.finding_id}) adds {sev} point(s)."
        )

    # 2. Needs follow-up — moderate weight
    if needs_fu:
        points += 1
        rationale.append(f"{len(needs_fu)} finding(s) marked for follow-up add 1 point.")

    # 3. Recurrence — confirmed same-category finding on this asset within window
    cutoff = _parse(inspection.inspection_date) - timedelta(days=recurrence_window_days)
    recurring_categories = set()
    for prev_insp in history:
        if prev_insp.inspection_id == inspection.inspection_id:
            continue
        if _parse(prev_insp.inspection_date) < cutoff:
            continue
        for f in confirmed:
            # categories match if same defect_category is present in this prior inspection's findings
            # (simplified: the prior inspection's detector_results mention that category)
            if f.defect_category in str(prev_insp.detector_results):
                recurring_categories.add(f.defect_category)
    if recurring_categories:
        points += len(recurring_categories)
        rationale.append(
            f"Recurring confirmed categories within {recurrence_window_days}d: "
            f"{', '.join(sorted(recurring_categories))} (+{len(recurring_categories)})."
        )
    factors["recurring_categories"] = sorted(recurring_categories)

    # 4. Asset criticality
    crit_map = {"low": 0, "normal": 0, "high": 1, "critical": 2}
    cpts = crit_map.get(asset.criticality, 0)
    if cpts:
        points += cpts
        rationale.append(f"Asset criticality '{asset.criticality}' adds {cpts} point(s).")

    # 5. Corrosion authoritative signal (read-only)
    corr_pct = float(inspection.authoritative_report.get("corrosion_percentage", 0.0) or 0.0)
    if corr_pct >= 10.0:
        points += 2
        rationale.append(f"Authoritative corrosion estimate {corr_pct:.2f}% is at/above 10% critical threshold (+2).")
    elif corr_pct >= 2.0:
        points += 1
        rationale.append(f"Authoritative corrosion estimate {corr_pct:.2f}% is at/above 2% warning threshold (+1).")

    # 6. Unreviewed findings cannot push to urgent; they hold at follow_up max.
    if unreviewed and not confirmed:
        rationale.append(
            "Only unreviewed machine suggestions exist; priority is capped at 'follow_up' "
            "until a human reviews them."
        )

    # Category assignment
    if points >= 6 and confirmed:
        category = "urgent_human_assessment"
    elif points >= 4:
        category = "priority_maintenance"
    elif points >= 2:
        category = "follow_up"
    else:
        category = "routine_review"

    if unreviewed and not confirmed and category in ("priority_maintenance", "urgent_human_assessment"):
        category = "follow_up"

    if not rationale:
        rationale.append("No contributing factors identified; default routine review.")

    rationale.append(
        "This is a recommendation only. It does not declare the asset unsafe and does not "
        "dispatch maintenance. A qualified human must decide the next action."
    )

    return PriorityScore(
        category=category,
        score=points,
        rationale=rationale,
        contributing_factors=factors,
    )


def _parse(iso_date: str):
    try:
        return datetime.fromisoformat(iso_date.split("T")[0])
    except Exception:
        return datetime.now()