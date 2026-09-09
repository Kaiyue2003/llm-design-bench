#!/bin/sh
set -eu

mkdir -p "${RUN_RESULTS_DIR}"
python /app/scripts/write_environment_manifest.py \
  --output "${RUN_RESULTS_DIR}/container_environment.json"

exec "$@"
