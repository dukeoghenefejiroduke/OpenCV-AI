"""Enterprise CMMS Webhook Dispatch Simulation (app/cmms.py).

Formats structured JSON maintenance work-orders for integration with enterprise maintenance
management systems (CMMS) such as IBM Maximo, SAP PM, or Jira Service Management.
Explicitly indicates that payloads are prepared locally as drafts and are NOT dispatched externally.
"""

from typing import Dict, Any
from agent.decision import InspectionDecision


def format_cmms_work_order(asset_id: str, decision: InspectionDecision, corrosion_pct: float) -> Dict[str, Any]:
    """Generates a structured JSON work-order payload for CMMS dispatch.

    Args:
        asset_id: Identifier of the industrial asset/pipe.
        decision: InspectionDecision object containing decision state, action, and approval requirement.
        corrosion_pct: Measured corrosion percentage.

    Returns:
        Structured dictionary payload representing the enterprise work-order draft.
    """
    priority = "HIGH" if decision.decision == "CRITICAL" else ("MEDIUM" if decision.decision == "WARNING" else "LOW")

    payload = {
        "source": "CementSight AI Inspector v2.0",
        "asset_id": asset_id,
        "priority": priority,
        "operational_state": decision.decision,
        "corrosion_percentage": corrosion_pct,
        "proposed_action": decision.action,
        "approval_required": decision.approval_required,
        "approval_status": "APPROVED" if not decision.approval_required else "PENDING_SUPERVISOR_SIGN_OFF",
        "reason": decision.reason,
        "dispatch_status": "PAYLOAD PREPARED — NOT DISPATCHED",
        "integration_notice": "Local demonstration draft only. No external CMMS (Maximo/SAP) dispatch connected."
    }
    return payload
