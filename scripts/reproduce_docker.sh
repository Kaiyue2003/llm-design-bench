#!/bin/sh
set -eu

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or is not available on PATH." >&2
  exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(dirname -- "$script_dir")

# Numeric IDs preserve bind-mount ownership. Rootless engines can explicitly
# override this with the IDs mapped to their host user (often 0:0).
LLMDM_CONTAINER_USER=${LLMDM_CONTAINER_USER:-"$(id -u):$(id -g)"}
case "$LLMDM_CONTAINER_USER" in
  *[!0-9:]*|:*|*:|*:*:*|"")
    echo "LLMDM_CONTAINER_USER must be a numeric UID:GID pair." >&2
    exit 2
    ;;
esac
case "$LLMDM_CONTAINER_USER" in
  *:*) ;;
  *) echo "LLMDM_CONTAINER_USER must be a numeric UID:GID pair." >&2; exit 2 ;;
esac

resolve_host_directory() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$project_dir" "$1" ;;
  esac
}

# Create mounts as the caller, never via the Docker daemon. Do not alter any
# existing files or permissions. Relative settings are project-relative.
RESULTS_DIR=$(resolve_host_directory "${RESULTS_DIR:-results/docker}")
ASSETS_DIR=$(resolve_host_directory "${ASSETS_DIR:-assets}")
mkdir -p -- "$RESULTS_DIR" "$ASSETS_DIR"
RESULTS_DIR=$(CDPATH= cd -- "$RESULTS_DIR" && pwd -P)
ASSETS_DIR=$(CDPATH= cd -- "$ASSETS_DIR" && pwd -P)
if [ ! -w "$RESULTS_DIR" ] || [ ! -x "$RESULTS_DIR" ]; then
  echo "RESULTS_DIR must be writable and accessible by the caller; choose a new directory or arrange access to existing results." >&2
  exit 2
fi
export LLMDM_CONTAINER_USER RESULTS_DIR ASSETS_DIR

service=benchmark
build=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --smoke) service=smoke; shift ;;
    --build) build=true; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [ "$service" = benchmark ] && [ "$#" -eq 0 ]; then
  set -- --help
fi
if [ "$build" = true ]; then
  set -- --build "$service" "$@"
else
  set -- "$service" "$@"
fi
exec docker compose --project-directory "$project_dir" -f "$project_dir/compose.yaml" run --rm "$@"
