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

    @staticmethod
    def evaluate_predictive_maintenance(
        visual_report: Optional[Dict[str, Any]] = None,
        anomalies: Optional[List[Dict[str, Any]]] = None,
        corrosion_history: Optional[List[Any]] = None,
        provenance: str = "simulated",
        human_approval: bool = False,
        is_real_asset: bool = True,
        critical_threshold: float = 10.0,
    ) -> Dict[str, Any]:
        """Combines rule-based visual/telemetry fusion with degradation velocity forecasting."""
        from app.forecasting import calculate_degradation_velocity, project_time_to_critical

        visual_report = visual_report or {}
        anomalies = anomalies or []

        # Simulation firewall: simulated anomalies never elevate a real asset to CRITICAL or alter real priority
        if is_real_asset and provenance == "simulated":
            effective_anomalies: List[Dict[str, Any]] = []
            firewall_note = "Simulation firewall active: simulated anomalies isolated from real asset risk escalation."
        else:
            effective_anomalies = [a for a in anomalies if not (is_real_asset and a.get("provenance") == "simulated")]
            firewall_note = None

        base = RiskFusionEngine.evaluate_risk(
            visual_report=visual_report,
            anomalies=effective_anomalies,
            provenance=provenance if not (is_real_asset and provenance == "simulated") else "real",
            human_approval=human_approval,
            is_real_asset=is_real_asset,
        )

        contributions = list(base["contributions"])
        if firewall_note:
            contributions.append(f"Rule [SIMULATION_FIREWALL]: {firewall_note}")

        current_corr = float(visual_report.get("corrosion_percentage", 0.0) or 0.0)
        history_pts = corrosion_history or []
        daily_rate = calculate_degradation_velocity(history_pts)
        days_to_critical = project_time_to_critical(current_corr, daily_rate, critical_threshold=critical_threshold)

        confirmed_visual = bool(visual_report.get("confirmed", False))
        anomaly_count = len(effective_anomalies)
        velocity_escalated = False

        if confirmed_visual and daily_rate > 0.0 and days_to_critical is not None and days_to_critical <= 14.0:
            velocity_escalated = True
            if base["risk_level"] in ("LOW", "MEDIUM"):
                base["risk_level"] = "HIGH"
                base["approval_required"] = True
            contributions.append(
                f"Rule [VELOCITY_ADVISORY]: Corrosion rate {daily_rate:.3f}%/day projects {days_to_critical} days to {critical_threshold}% critical threshold."
            )
        elif len(history_pts) < 3:
            contributions.append(
                "Rule [FORECAST_INSUFFICIENT_DATA]: Fewer than 3 dated observations; trend projection withheld."
            )

        if confirmed_visual and anomaly_count >= 2 and base["risk_level"] == "MEDIUM":
            base["risk_level"] = "HIGH"
            base["approval_required"] = True
            contributions.append(
                f"Rule [CORRELATED_MULTIMODAL]: {anomaly_count} persistent telemetry anomalies corroborate confirmed visual defect."
            )

        return {
            **base,
            "contributions": contributions,
            "degradation_velocity_pct_per_day": round(daily_rate, 4),
            "estimated_days_to_critical": days_to_critical,
            "velocity_escalated": velocity_escalated,
            "anomaly_count": anomaly_count,
            "provenance": provenance,
            "screening_disclaimer": "Deterministic screening fusion estimate only; not a certified physical RUL or autonomous dispatch trigger.",
        }

