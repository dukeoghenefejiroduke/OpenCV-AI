# Hosted Mode Architecture, Design & Configuration

> **Status:** **OFF BY DEFAULT — NOT DEPLOYED, NOT VERIFIED.**
> This document specifies the secure enterprise hosted deployment architecture for CementSight AI. All hosted features remain strictly disabled by default.

## 1. Security & Authentication
- **Token Authentication:** Strict JWT / Bearer token authentication required for all API endpoints when hosted mode is active. Unauthenticated requests are rejected with `401 Unauthorized`.
- **Transport Security (HTTPS):** Mandatory TLS 1.3 encryption in transit. Plain HTTP requests are rejected or redirected to HTTPS.
- **Rate Limiting:** Enforced rate limit of 60 requests per minute per authenticated client token to prevent denial-of-service and abuse.
- **Upload Caps:** Maximum request body size capped at 25MB per inspection payload.

## 2. Storage & Multi-Tenancy
- **Per-User Storage Isolation:** In hosted mode, user data, workspaces, and image stores are strictly isolated under partitioned directories: `data/hosted_storage/{user_id}/`.
- **Data Privacy & Encryption:** Storage encryption at rest (AES-256) for all hosted sqlite databases and object stores.

## 3. Containerization & Serverless Deployment
- **Docker Deployment:** Reference container packaging via Dockerfile (Python 3.11/3.14 slim image with OpenCV and numpy dependencies).
- **AWS Lambda Deployment:** Serverless invocation via `lambda_handler.py`, sharing the frozen OpenCV vision inspection pipeline (`vision/inspection.py`) with local edge mode.

## 4. Verification Statement
*All hosted mode parameters, configurations, and deployment topologies defined herein are for architecture design and reference only. They are not active in local edge mode and have not been deployed or verified in a production cloud environment.*
