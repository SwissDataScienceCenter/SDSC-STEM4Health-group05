#!/usr/bin/env bash
# Start the Batch Monitoring Studio as the main app of a Renku session (custom launcher).
# Renku serves the session under a sub-path, given in RENKU_BASE_URL_PATH; Streamlit must use it.
set -e
cd "$(dirname "$0")/.."   # repository root, so .streamlit/config.toml is picked up
exec streamlit run frontend/app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-8501}" \
  --server.baseUrlPath "${RENKU_BASE_URL_PATH:-/}" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
