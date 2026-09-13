#!/bin/sh
set -eu

for run in first replay; do
  output="${RUN_RESULTS_DIR}/${run}"
  python /app/scripts/write_environment_manifest.py --output "${output}/container_environment.json"
  llm-design-bench-publication \
    --all-methods \
    --method-config /app/configs/all_methods_smoke.json \
    --no-data-mixture --function ackley --function booth \
    --seed 38 --logged-samples 32 --recommendations 8 \
    --epochs 2 --particle-steps 2 --bdi-steps 2 --method-steps 3 \
    --torch-threads 1 --deterministic --no-resume \
    --results-dir "${output}"
done

python /app/scripts/verify_reproduction.py \
  "${RUN_RESULTS_DIR}/first" --compare "${RUN_RESULTS_DIR}/replay"
