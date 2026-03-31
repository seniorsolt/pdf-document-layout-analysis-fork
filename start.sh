#!/bin/bash
set -e

echo "=== Checking/downloading models ==="
PYTHONPATH=/app/src python -c "
from download_models import download_models
download_models('doclaynet')
download_models('fast')
"
echo "=== Models ready ==="

echo "=== Starting gunicorn ==="
exec gunicorn -k uvicorn.workers.UvicornWorker --chdir ./src app:app --bind 0.0.0.0:5060 --timeout 10000
