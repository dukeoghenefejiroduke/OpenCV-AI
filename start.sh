#!/usr/bin/env bash
# ==============================================================================
# CementSight AI — Termux Edge Startup Script (start.sh)
# Single-Inspector Local Node Startup & Health Check
# ==============================================================================

set -e

echo "========================================================"
echo "Starting CementSight AI (Edge / Termux Mode)..."
echo "========================================================"

# 1. Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

# 2. Ensure data directories exist
mkdir -p data/uploads data/outputs data/status data/backups workspace

# 3. Check database / workspace store integrity
echo "Verifying workspace database..."
python3 -c "from workspace.store import WorkspaceStore; s = WorkspaceStore(); print('Database verified successfully. Assets:', len(s.list_assets()))"

# 4. Start Local Server on 127.0.0.1:8989
echo "Launching FastAPI / HTTP server on http://127.0.0.1:8989 ..."
python3 tools/server.py
