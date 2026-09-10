#!/bin/sh
set -eu

DATA_RECIPES_URL="https://github.com/namkoong-lab/data-recipes.git"
DATA_RECIPES_COMMIT="37269969a0957448d51622e0c083977bc5d260e8"
DATA_RECIPES_ROOT="/opt/external/data-recipes"

if [ ! -d "${DATA_RECIPES_ROOT}/.git" ]; then
  git clone "${DATA_RECIPES_URL}" "${DATA_RECIPES_ROOT}"
fi

if ! git -C "${DATA_RECIPES_ROOT}" diff --quiet \
  || ! git -C "${DATA_RECIPES_ROOT}" diff --cached --quiet; then
  echo "The Data Recipes Docker volume is dirty; refusing to overwrite it." >&2
  exit 2
fi

if ! git -C "${DATA_RECIPES_ROOT}" cat-file -e "${DATA_RECIPES_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${DATA_RECIPES_ROOT}" fetch origin "${DATA_RECIPES_COMMIT}" --depth 1
fi
git -C "${DATA_RECIPES_ROOT}" checkout --detach "${DATA_RECIPES_COMMIT}"

exec llm-design-bench-publication "$@" \
  --data-recipes-root "${DATA_RECIPES_ROOT}" \
  --seed 38 --seed 39 --seed 40 --seed 41 \
  --seed 42 --seed 43 --seed 44 --seed 45 \
  --logged-samples 256 \
  --recommendations 128 \
  --epochs 100 \
  --particle-steps 100 \
  --bdi-steps 100 \
  --train-min-percentile 0 \
  --train-max-percentile 40 \
  --torch-threads 1 \
  --deterministic \
  --results-dir "${RUN_RESULTS_DIR}"
