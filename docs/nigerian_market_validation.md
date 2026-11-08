# Nigerian Market Validation & Customer Discovery Package

## B1. Problem Definition

### Positioning
CementSight AI is positioned as an **affordable, offline-first infrastructure inspection and maintenance intelligence platform** designed specifically for resource-constrained water and industrial infrastructure operators.

### Core Problems Addressed
1. **Inspection Documentation:** Eliminating disorganized paper logs and unindexed photo galleries by anchoring records to immutable asset IDs and SHA-256 image evidence hashes.
2. **Defect Screening:** Providing rapid automated multi-hazard screening (corrosion, cracks, spalling, leakage, insulation gaps) to highlight areas requiring physical attention.
3. **Asset Identification:** Streamlining asset lookup and field data entry using standardized QR tag URI payloads (`CEMENTSIGHT://ASSET/<asset_id>`).
4. **Maintenance Prioritization:** Offering transparent, non-autonomous prioritization scoring based on human review status, asset criticality, recurrence, and corrosion metrics.
5. **Repair Verification:** Linking initial defect findings to maintenance records, work performed, and follow-up inspection evidence to verify repair completion.
6. **Inspection Reporting:** Generating self-contained HTML inspection reports complete with statutory disclaimers ("Not a certification or engineering approval").
7. **Historical Evidence Management:** Maintaining robust local inspection history logs and SQLite audit trails without relying on continuous cloud connectivity.

### Product Scope vs. Broader Challenges
CementSight AI directly addresses **operational inspection documentation, screening workflow efficiency, and maintenance traceability**. It does not solve broader systemic water infrastructure challenges such as municipal power supply instability, raw water supply deficits, capital funding shortages, or complex political governance issues. Rather, it provides local engineering teams with digital tooling to maximize the lifespan and safety of existing assets.

---

## B2. Customer Segments

1. **State Water Corporations (e.g., Lagos Water Corporation, Greater Ibadan Waterworks):** Publicly owned utilities operating extensive water treatment plants, transmission mains, and distribution networks.
2. **Private Water Utilities & Bottling Plants:** Industrial facilities requiring rigorous plant hygiene, pressure vessel monitoring, and corrosion prevention.
3. **Water Infrastructure Inspection Contractors:** Specialized engineering service firms conducting condition assessments for municipal and industrial clients.
4. **Industrial Maintenance Teams (Manufacturing & Processing):** Facilities managers overseeing industrial piping, storage tanks, and structural concrete walls.
5. **Oil & Gas Infrastructure Operators (Upstream/Midstream facilities):** Operators managing surface flowlines, gathering pipes, and structural steel supports in corrosive environments.
6. **Engineering Consultants & Forensic Auditors:** Third-party engineering evaluators requiring verifiable visual evidence and audit trails.
7. **Universities & Research Institutions (Civil & Water Engineering Departments):** Academic groups conducting non-destructive evaluation (NDE) research and student training.

*Early-Adopter Focus:* Industrial maintenance teams and regional water inspection contractors operating in remote or intermittent-connectivity environments represent the most viable early-adopter segments due to their immediate need for offline-first inspection traceability.

---

## B3. Competitor Comparison

| Evaluation Dimension | CementSight AI | Jacobs Dragonfly | Siemens / Aerosophia | PuralinkOS | NASSCO (Standards Ecosystem) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Inspection Capabilities** | Multi-hazard visual screening (Corrosion, Crack, Spalling, Leakage, Insulation) | Advanced enterprise asset management & drone/LiDAR integration | Enterprise digital twins, SCADA & predictive analytics | Pipeline inspection data management | Pipeline assessment certification standards (PACP/MACP) |
| **AI-Assisted Analysis** | Classical OpenCV/numpy multi-hazard heuristics + deterministic agent decision layer | Proprietary cloud-hosted machine learning models | Industrial AI and machine learning suites | Cloud-based defect tagging | Not software — defines defect coding standards |
| **Asset Management** | SQLite-backed local workspace, QR asset identification | Enterprise GIS & asset databases | Enterprise asset management (Maximo/SAP integration) | Cloud database asset tracking | Standardized defect coding taxonomy |
| **Reporting** | Self-contained HTML reports with statutory disclaimers | Enterprise reporting dashboards | SCADA-integrated analytics | Web portal reports | Standardized PACP condition grading reports |
| **Maintenance Workflows** | Human-in-the-loop prioritization, CMMS payload preparation | Enterprise work order dispatch (Maximo/SAP) | Automated maintenance dispatch | Work order tracking | Standardized remediation coding |
| **Offline Operation** | **Fully offline-first** (Zero cloud/internet required; runs locally on Android/Termux) | Requires cloud connectivity | Requires robust enterprise cloud/edge infrastructure | Cloud-dependent | N/A (Standard specification) |
| **Target Customers** | Resource-constrained operators, field contractors, municipal water boards | Large enterprises, major municipal authorities | Heavy industry, large utilities | Municipal contractors | Municipal inspection agencies |
| **Deployment Model** | Local Python executable / Termux container | Enterprise SaaS / on-premise cloud | Enterprise server / Cloud | Cloud SaaS | N/A |

---

## B4. Customer Interview Questionnaire

*Designed for infrastructure managers, maintenance supervisors, and chief engineers.*

1. What is your current procedure for conducting routine condition assessments of water pipes, tanks, and structural concrete?
2. What hardware tools (e.g., DSLRs, smartphones, boroscopes, drones) do your inspection crews currently carry in the field?
3. How are inspection photographs and notes currently documented, cataloged, and archived after a site visit?
4. What software (if any) do you currently use to track asset maintenance history and prioritize repair work?
5. How frequently do your inspection sites experience unreliable internet connectivity or complete lack of cellular signal?
6. When machine vision or automated defect detection tools are evaluated, how important is it that an engineer retains final review authority over every finding?
7. What difficulties do you encounter when trying to verify whether a previously reported defect has been successfully repaired?
8. How long does it typically take your team to compile a formal condition assessment report following a major inspection campaign?
9. What are the primary cost or operational barriers preventing your organization from adopting advanced digital inspection software?
10. Who within your organization holds purchasing authority or budgetary approval for software tools valued under $5,000/year?
11. How do your field teams currently identify specific assets (e.g., physical metal tags, GPS coordinates, barcode/QR scanning)?
12. What specific reporting format (e.g., PDF, HTML, Excel, CMMS work order) do your stakeholders require?
13. How do you currently handle false positives generated by automated inspection tools?
14. Would your organization be open to participating in a 30-day supervised pilot of an offline-first, local inspection tool running on standard mobile devices?
15. What specific metrics or success criteria would you require to consider such a pilot successful?

---

## B5. Pilot Evaluation Framework

### Measurable Evaluation Criteria
1. **Inspection Time:** Reduction in time required to create a verified inspection record compared to legacy paper/manual methods.
2. **Documentation Completeness:** Percentage of inspections containing complete asset metadata, date stamps, and image evidence hashes.
3. **Inspector Agreement:** Rate at which certified human inspectors agree with machine-suggested findings (precision/recall tracking).
4. **False-Positive / False-Negative Tracking:** Systematic logging of rejected findings to measure screening noise.
5. **Review Efficiency:** Ease and speed of reviewing unreviewed machine suggestions using filter chips.
6. **Historical Traceability:** Time required to retrieve historical inspection records and maintenance logs for a specific asset ID.
7. **Maintenance Completion:** Percentage of prioritized recommendations that transition to completed maintenance records.
8. **Repair Verification Completeness:** Ratio of repairs backed by follow-up inspection evidence (`verified_resolved`) versus pending work.
9. **Offline Usability:** Frequency of successful inspection creation and report generation without active internet connectivity.
10. **User Satisfaction:** Qualitative feedback from field inspectors and maintenance supervisors regarding interface usability on mobile devices.

### Collecting Human-Reviewed Ground Truth
- Every machine-generated finding is initialized as `unreviewed`.
- Qualified human reviewers inspect each bounding box and explicitly mark findings as `confirmed`, `rejected`, or `needs_followup`.
- Ground-truth datasets are derived strictly from human-reviewed operational decisions, ensuring detector confidence scores are never conflated with calibrated accuracy.

---

## B6. Outreach Package

### Concise Introduction
> **CementSight AI** is an offline-first infrastructure inspection and maintenance intelligence platform. Designed specifically for resource-constrained environments, it operates entirely locally on standard mobile devices without requiring cloud connectivity. It assists engineering teams with rapid visual defect screening, standardized asset tracking, human-in-the-loop maintenance prioritization, and verifiable inspection reporting.

### Professional Pilot Invitation
> **Subject:** Invitation to Participate in a Supervised Offline Inspection Pilot
> 
> Dear [Engineering Director / Head of Operations],
> 
> We are inviting your team to evaluate **CementSight AI** in a 30-day supervised pilot. Our platform provides field engineers with a local, offline-first tool to document pipe and concrete inspections, screen for surface defects, prioritize maintenance actions, and generate verifiable HTML inspection reports—all without requiring cloud connectivity.
> 
> We would welcome the opportunity to demonstrate the platform and discuss how it can support your maintenance workflows.

### Demonstration Description
- **Duration:** 30 minutes (Live walkthrough on local hardware / mobile device).
- **Core Focus:** Demonstration of offline asset registration, image upload and multi-hazard defect screening, human review filtering, maintenance record linking, and self-contained report generation.

### Information Requested from Pilot Organizations
1. Typical volume of monthly infrastructure inspections.
2. Preferred mobile hardware operating environment (Android / Termux / local PC).
3. Specific asset categories of highest priority (e.g., transmission mains, storage reservoirs, pump stations).
4. Designated technical liaison for pilot feedback collection.
