# CementSight Release Checklist

## Pre-Release Validation & Trust Protocol
- [ ] **Test Suite Passing:** Run full pytest suite twice with zero failures (`python -m pytest`).
- [ ] **Offline Compliance:** Verify dashboard loads and operates with network blocked (no CDN dependencies, local CSS & system fonts only).
- [ ] **No Unvalidated Claims:** Ensure no accuracy claims are made for experimental detectors (e.g., crack detector explicitly documented with TUBERS ROC-AUC 0.3685, 95% CI 0.25–0.53).
- [ ] **Lockbox Integrity:** Ensure tuning harnesses and evaluation scripts do not access or report metrics on untouched lockbox groups.
- [ ] **Review Workflow Audit:** Verify that all findings require a reviewer name for state changes (`pending_review`, `confirmed`, `rejected`, `retake_requested`), and that audit logs are written atomically in append-only JSONL format.
- [ ] **Priority Safety:** Verify that unconfirmed or candidate findings (such as cracks with `crack_enabled_for_priority=False`) never raise asset severity or priority.
- [ ] **HTML Report Escaping:** Verify that all report fields are fully escaped using `html.escape` and include source image SHA-256 hashes, image-quality verdicts, review states, and limitations.
- [ ] **Corrosion Validation:** Verify corrosion dataset manifests with SHA-256 hash checking and sensitivity/specificity reporting using Wilson score intervals.
- [ ] **Git & Security:** Ensure no sensitive credentials, secrets, or .env files are tracked, and no git commits are made without explicit user instruction.
