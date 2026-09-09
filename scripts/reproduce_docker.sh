#!/bin/sh
set -eu

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or is not available on PATH." >&2
  exit 1
fi

exec docker compose run --rm publication
