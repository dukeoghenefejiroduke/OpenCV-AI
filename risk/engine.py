"""Multi-Hazard Risk Fusion Engine (risk/engine.py).

Explicit rule table fusing confirmed visual findings with anomalies into LOW/MEDIUM/HIGH/CRITICAL
plus a per-rule contribution list, strict isolation rules, and approval requirements.
"""

from typing import Dict, Any, List, Optional


class RiskFusionEngine:
    """Fuses visual inspection and telemetry anomalies into structured risk tiers."""

    @staticmethod
    def evaluate_risk(
        visual_report: Optional[Dict[str, Any]] = None,
        anomalies: Optional[List[Dict[str, Any]]] = None,
        provenance: str = "simulated",
        human_approval: bool = False,
        is_real_asset: bool = True
    ) -> Dict[str, Any]:
        visual_report = visual_report or {}
        anomalies = anomalies or []

        corrosion = visual_report.get("corrosion_percentage", 0.0)
        rec = visual_report.get("recommendation", "PASS").upper()
        confirmed_visual = visual_report.get("confirmed", True) and rec in ["CRITICAL", "WARNING", "HIGH", "REJECTED"]

        anomaly_count = len(anomalies)
        has_imported_real = any(a.get("provenance") == "imported_real" for a in anomalies)

        contributions = []
        risk_level = "LOW"
        approval_required = False
        attention_flag = False

        # Rule Evaluation Table
        if confirmed_visual and (rec == "CRITICAL" or corrosion > 15.0):
            risk_level = "CRITICAL"
            approval_required = True
            contributions.append("Rule [CRITICAL_VISUAL]: Confirmed visual finding with CRITICAL recommendation or high corrosion (>15%).")
        elif confirmed_visual and rec in ["WARNING", "HIGH"]:
            risk_level = "HIGH"
            approval_required = True
            contributions.append("Rule [HIGH_VISUAL]: Confirmed visual finding with HIGH/WARNING recommendation.")
        elif has_imported_real and not confirmed_visual:
            attention_flag = True
            if anomaly_count >= 2:
                risk_level = "MEDIUM"
                contributions.append("Rule [IMPORTED_REAL_ATTENTION]: Imported real anomalies set attention flag; capped at MEDIUM without confirmed visual finding.")
            else:
                risk_level = "LOW"
                contributions.append("Rule [IMPORTED_REAL_LOW]: Minor imported real anomaly flagged for attention.")
        elif anomaly_count > 0 and provenance == "simulated":
            risk_level = "MEDIUM"
            contributions.append("Rule [SIMULATED_ANOMALY]: Simulated telemetry anomalies detected.")
        else:
            risk_level = "LOW"
            contributions.append("Rule [BASELINE_SAFE]: No significant confirmed findings or anomalies.")

        # Rule 3 constraint: imported_real cannot reach CRITICAL without confirmed visual or human approval
        if has_imported_real and risk_level == "CRITICAL" and not (confirmed_visual or human_approval):
            risk_level = "HIGH"
            approval_required = True
            contributions.append("Rule [IMPORTED_CRITICAL_GUARD]: Imported real data cannot reach CRITICAL without confirmed visual or human approval.")

        # Rule 4: CRITICAL sets approval_required; nothing is dispatched
        if risk_level == "CRITICAL":
            approval_required = True

        return {
            "risk_level": risk_level,
            "approval_required": approval_required,
            "attention_flag": attention_flag,
            "dispatched": False,  # Strictly NO dispatching / autonomous actions
            "contributions": contributions
        }
