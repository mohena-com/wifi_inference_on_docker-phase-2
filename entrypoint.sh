#!/usr/bin/env bash
set -e

# Optional pre-start operations (e.g., create folders)
mkdir -p /app/checkpoints /app/data || true

# Try to warm load the model to surface errors early (calls /reload_model via python small runner)
if [ -n "$MODEL_PAYLOAD_PATH" ] && [ -f "$MODEL_PAYLOAD_PATH" ]; then
  echo "Model payload found at $MODEL_PAYLOAD_PATH; attempting a quick load test..."
  python - <<PY
import os, sys, torch
from app import load_model
p = os.environ.get("MODEL_PAYLOAD_PATH")
try:
    load_model(p)
    print("Model loaded OK")
except Exception as e:
    print("Model load failed:", e)
PY
else
  echo "No MODEL_PAYLOAD_PATH found or file missing; continue starting server"
fi

# Start Gunicorn (workers=1 default; set WORKERS env var to control)
WORKERS=${WORKERS:-1}
TIMEOUT=${GUNICORN_TIMEOUT:-120}
HOST=0.0.0.0
PORT=${PORT:-8000}

echo "Starting gunicorn on ${HOST}:${PORT} (workers=${WORKERS})"
exec gunicorn -w ${WORKERS} -b ${HOST}:${PORT} --timeout ${TIMEOUT} "app:app"
