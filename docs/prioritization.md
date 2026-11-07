# Prioritization & Risk Scoring Engine

## Scoring Methodology
CementSight AI calculates asset risk and maintenance prioritization scores based on multi-factor weighted heuristics:
1. **Asset Criticality Weight:** `low` (1x), `normal` (2x), `high` (3x), `critical` (5x).
2. **Confirmed Defect Severity & Count:** Weighted sum of confirmed corrosion, cracking, spalling, and leakage findings.
3. **Degradation Velocity & Temporal Trend:** Rate of corrosion growth calculated from historical inspection observations.
4. **Time to Critical Threshold:** Projected days until corrosion exceeds 20% critical threshold.

## Priority Categories
- **P1 - Critical (Immediate Action / Emergency Shutdown):** Active leakage, critical spalling, or rapid degradation trajectory threatening structural integrity.
- **P2 - High (Scheduled Intervention within 7 Days):** Confirmed severe corrosion or cracking with high asset criticality.
- **P3 - Medium (Monitor / Routine Maintenance):** Moderate findings or stable degradation within safe operational bounds.
- **P4 - Low (Normal Operations):** Minor or unconfirmed defects with low criticality.
