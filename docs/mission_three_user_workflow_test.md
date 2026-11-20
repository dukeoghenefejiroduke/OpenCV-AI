# CementSight AI — Mission Three User Workflow Test

**Date:** November 18, 2026  
**Scope:** End-to-end operator workflow test from asset selection to report generation.

## 1. End-to-End Operator Journey Test
1. **Asset Selection / Registration:** Operator selects pipeline asset `PIPE-01` (Material: `steel`) or registers new asset with declared material.
2. **Material Declaration:** Operator reviews or declares inspected surface material (`steel`, `concrete`, `brick`, `wood`, `insulation`, `unknown`, `mixed`).
3. **Image Capture / Upload:** Operator uploads field inspection photograph (`data/pipe_corrosion.jpg`).
4. **Inspection Execution:** System executes corrosion and multi-hazard detection, applying material scoping rules.
5. **Findings & Limitations:** Dashboard displays findings, corrosion percentage estimate, and explicit material applicability limitations.
6. **Human Review:** Supervisor reviews uncertain or borderline findings, toggling state to `confirmed` or `rejected`.
7. **Persistence & History:** Inspection record, findings, material provenance, and review status are securely stored in SQLite workspace.
8. **Report Generation:** Supervisor generates professional HTML report / CSV export preserving complete audit traceability, material provenance, and limitations disclaimers.

## 2. Usability & Mobile Observations
- Responsive design adapts cleanly to mobile viewports ($360\text{px}$ layout).
- Offline service worker caching ensures uninterrupted field operation in remote locations without cellular connectivity.
- Clear terminology prevents confusion between inspector-declared surface type and automated findings.
