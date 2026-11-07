# Inspection Workflow & API Guide

## Overview
CementSight AI provides a local, offline, human-reviewed inspection workspace built on top of a zero-dependency SQLite store (`WorkspaceStore`).

## Endpoints
1. **Asset Management (`/api/workspace/assets`)**
   - `GET`: List all registered assets.
   - `POST`: Register or update an asset (`asset_id`, `location`, `pipe_material`, `criticality`, etc.).
2. **Inspection & Multi-Hazard Orchestration (`/api/workspace/inspection`)**
   - `POST`: Upload an image (multipart or JSON path) along with asset metadata. Runs the multi-hazard vision orchestrator, records inspection findings, and stores the authoritative report.
3. **Asset Detail (`/api/workspace/asset/<asset_id>`)**
   - `GET`: Retrieve asset details, inspection history, and maintenance records.
4. **Inspection Details & Prioritization (`/api/workspace/inspection/<inspection_id>`)**
   - `GET`: Retrieve inspection results, individual findings, and calculated asset priority/risk scoring.
5. **Finding Review (`/api/workspace/review/<finding_id>`)**
   - `POST`: Human review sign-off (`confirmed`, `rejected`, `needs_followup`).
6. **Maintenance Logging (`/api/workspace/maintenance`)**
   - `POST`: Record corrective or preventive maintenance actions.
7. **HTML Report Export (`/api/workspace/report/<inspection_id>`)**
   - `GET`: Generate and download a standalone HTML inspection report.
