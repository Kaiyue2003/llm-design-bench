#!/bin/sh
set -eu

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or is not available on PATH." >&2
  exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(dirname -- "$script_dir")
if [ "$#" -eq 0 ]; then
  set -- --help
fi
exec docker compose --project-directory "$project_dir" -f "$project_dir/compose.yaml" run --rm benchmark "$@"
