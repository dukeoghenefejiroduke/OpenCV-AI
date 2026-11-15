# Offline Sync Design & Architecture (Design Only)

## Overview
CementSight AI works offline; sync is a design only. The application operates in remote industrial facilities where cellular or Wi-Fi connectivity may be intermittent or absent. The offline sync architecture is a documented blueprint ensuring local field inspections, asset metadata updates, and maintenance records are fully captured locally.

## 1. Local Queue Format
- **Storage Mechanism:** SQLite-backed local workspace store (`data/workspace/cementsight.db`).
- **Outbox Queue Table (`sync_outbox`):**
  - `event_id` (TEXT PRIMARY KEY): UUIDv4 identifier for the offline transaction.
  - `event_type` (TEXT): `inspection`, `asset_upsert`, `maintenance_record`, `finding_review`.
  - `payload_json` (TEXT): Serialized JSON payload containing entity attributes and image/asset references.
  - `created_at` (TEXT): ISO 8601 timestamp.
  - `status` (TEXT): `pending`, `syncing`, `synced`, `failed`.
  - `retry_count` (INTEGER): Number of failed synchronization attempts.

## 2. Conflict Resolution
- **Strategy:** Last-Write-Wins (LWW) based on UTC timestamp with asset-level entity immutability.
- **Deduplication:** Inspections are immutable once generated (identified by unique `inspection_id`). If an asset metadata record is updated offline and online simultaneously, the higher timestamp prevails, with conflict logs written to `data/status/sync_conflicts.log`.

## 3. Authentication & Security
- **Token Storage:** Encrypted local token store (`AES-256` via device secure storage where available, fallback to restricted local sqlite).
- **Transport:** Mutual TLS (mTLS) or OAuth 2.0 Bearer tokens over HTTPS when syncing to cloud/edge servers.
- **Data Privacy:** Local image files are stored in `data/workspace/images/` with SHA-256 integrity hashes. No telemetry or imagery is transmitted without explicit user action or active sync dispatch.

---

## Multilingual Support (Pidgin & French)
*Status: Ready for translation string tables; marked "needs native review, disabled" pending professional review.*

### Pidgin Report Strings (Draft - Needs Native Review, Disabled)
```json
{
  "lang": "pcm",
  "status": "disabled",
  "notice": "Dis report na screening aid only. E no be engineering certification.",
  "corrosion_screening": "Corrosion screening percentage",
  "recommendation": "Wetin you suppose do"
}
```

### French Report Strings (Draft - Needs Native Review, Disabled)
```json
{
  "lang": "fr",
  "status": "disabled",
  "notice": "Ce rapport est une aide au criblage et ne constitue pas une certification d'ingenierie.",
  "corrosion_screening": "Pourcentage de criblage de corrosion",
  "recommendation": "Recommandation"
}
```
