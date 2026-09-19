# Docker: the same frozen LLM-DM workflow

The container runs `llm-design-bench`, the same formal CLI as a native installation.
It does not choose budgets, start a real campaign implicitly or download data.
Compose has two services: `benchmark` and `smoke`.

The image pins Python 3.11.15 and uv 0.11.8, installs `uv.lock`, and uses one
thread for common BLAS libraries. This is a CPU/Linux-container workflow, not
evidence of CUDA or cross-platform numerical parity. The ownership defaults below
target a **local Docker daemon**; remote daemons and user-namespace mappings need
explicit operator configuration.

## Recommended: use the wrappers

From the project checkout on Linux/macOS:

```bash
./scripts/reproduce_docker.sh --build methods
./scripts/reproduce_docker.sh --build --smoke
```

On Windows with Docker Desktop set to Linux containers:

```powershell
.\scripts\reproduce_docker.ps1 --build methods
.\scripts\reproduce_docker.ps1 --build --smoke
```

`methods` only lists the 27 non-SPADE formal methods; it does not load data or
train. `--smoke` selects the invented-data engineering check described below.
Leading `--build` and `--smoke` can appear in either order. Without `--build`, the
wrapper uses the selected image; with no arguments it prints CLI help.

Both wrappers locate the project from their own script path, prepare the mount
directories, and pass absolute paths to Compose. They read the following settings
from the **process environment**, not by parsing a project `.env` file:

| Variable | Wrapper default / behavior |
| --- | --- |
| `RESULTS_DIR` | `results/docker`, writable host output |
| `ASSETS_DIR` | `assets`, mounted read-only |
| `LLMDM_CONTAINER_USER` | Shell/POSIX PowerShell: caller's numeric UID:GID; Windows PowerShell: `1000:1000` for Docker Desktop Linux containers |

Relative directory overrides are resolved against the **project root**, not the
caller's working directory. Missing directories are created by the host caller;
the wrappers do not change permissions or ownership of existing files.

An explicit `LLMDM_CONTAINER_USER=uid:gid` overrides the defaults. For a rootless
daemon, `0:0` may be the appropriate explicit choice when container root maps to
your host account. The wrappers do not detect rootless mode or other UID mappings.
Do not use `0:0` blindly with a rootful daemon: it can create root-owned outputs.

Both services set `HOME=/tmp`, `USER=runner`, and `LOGNAME=runner`. These let Python/
PyTorch resolve a username and writable cache location even when the numeric UID
has no passwd entry; they do not change the actual configured UID/GID.

## What the smoke check proves

`scripts/container_smoke.py` creates invented logs and a fake oracle in a new
directory and exercises the real prepare/freeze/run CLI:

- all 27 methods use tiny engineering-test budgets for pilot seed 0;
- K=128 candidates and per-attempt artifacts are checked;
- only Best Logged's formal seed 38 is run and resumed without a new oracle call
  or attempt;
- the formal shard must remain incomplete: seven missing seeds and
  `rank_eligible=false`.

By default, output is under
`results/docker/container-smoke/invented-smoke-*/run/`, including
`smoke_verification.json`. No real checkpoint, archived result or approved
experimental budget is read or changed. This is not a full-budget pilot, a
27-method formal result or proof of real-data optimization quality.

Actual Compose/container execution is checked separately in Linux container CI.
Windows wrapper argument/directory tests are not a Docker Desktop execution test;
the current local development host has no Docker runtime. Unit tests alone do not
establish container, GPU or scientific-result reproduction.

## Supply inputs and run an approved campaign

`ASSETS_DIR` is mounted at `/assets` read-only; `RESULTS_DIR` is mounted at
`/results` writable. Place a trusted `data-recipes` checkout and approved methods
file under assets. A verified existing bundle/plan can also be supplied there.
The image and wrappers do not clone/update the checkout or fetch checkpoints.
Record its revision and trust its pickle/checkpoint files before loading them.

**After the code merge and budget approval**, use explicit CLI arguments. The
examples below use container paths; replace the checkpoint placeholder and repeat
`--oracle-checkpoint` for every checkpoint the oracle uses:

```bash
./scripts/reproduce_docker.sh prepare \
  --data-recipes-root /assets/data-recipes \
  --oracle-checkpoint /assets/data-recipes/path/to/trusted/checkpoint.pt \
  --output /results/shared-data

./scripts/reproduce_docker.sh freeze \
  --data-recipes-root /assets/data-recipes \
  --data-bundle /results/shared-data \
  --methods-file /assets/approved-methods.json \
  --experiment-id merged-llmdm-v1 --output /results/plan.json

./scripts/reproduce_docker.sh run \
  --data-recipes-root /assets/data-recipes \
  --data-bundle /results/shared-data --plan /results/plan.json \
  --phase pilot --setting multi_scale --device cpu \
  --results-dir /results/pilot

./scripts/reproduce_docker.sh run \
  --data-recipes-root /assets/data-recipes \
  --data-bundle /results/shared-data --plan /results/plan.json \
  --phase formal --setting multi_scale --device cpu \
  --pilot-results /results/pilot --results-dir /results/formal
```

PowerShell accepts the same CLI arguments through `reproduce_docker.ps1`; use
PowerShell line continuation or put them on one line. The methods-file schema is
in [Reproducing Results](REPRODUCING.md). Preparation/freezing do not query the
oracle. There is no implicit real-experiment budget preset.

Formal runs require matching verified pilots. `fixed_1b` or `both` need their
corresponding pilot coverage. Resume and explicit infrastructure-retry rules are
the same as the native CLI; changing code/configuration requires a new plan and
output directory. Native and container runs save the same manifests, candidate/
evaluation NPZs, environment records, checksums and failure evidence.

## Advanced: direct Compose calls

Compose deliberately requires `LLMDM_CONTAINER_USER`; omitting it fails with a
setup error, **not an unsafe root fallback**. Bind mounts use
`create_host_path: false`, so you must also create both host directories first.
Running a wrapper does not configure a later, separate shell's environment.

For direct Compose on a local POSIX host, from the project root:

```bash
export LLMDM_CONTAINER_USER="$(id -u):$(id -g)"
export RESULTS_DIR="$PWD/results/docker"
export ASSETS_DIR="$PWD/assets"
mkdir -p "$RESULTS_DIR" "$ASSETS_DIR"
docker compose run --build --rm benchmark methods
```

Choose any rootless/namespace override explicitly before running. On Windows,
prefer the PowerShell wrapper; for raw Compose, explicitly set the numeric user
appropriate to Docker Desktop (`1000:1000` by default above), absolute host paths,
and create those directories yourself.

With that same explicit environment/directory setup, reports can be derived into
a separate output directory without training:

```bash
docker compose run --rm --entrypoint llm-design-bench-report benchmark \
  from-unified --input-csv /results/formal/method_seed_results.csv \
  --results-dir /results/report
```

Keep the full run directory, including failed attempts, not just its CSV tables.
The former publication services and raw-run verification scripts are not active
parts of this workflow.

## Existing output and reproducibility limits

This fix does **not** repair historical root-owned output. Neither wrapper runs
`chown`/`chmod` or deletes existing results. If a directory is inaccessible, select
a new writable `RESULTS_DIR`; recover valuable old output separately with an
administrator's explicit assistance. Do not overwrite it or rerun a scientific
campaign merely to resolve file permissions.

Set `LLM_DESIGN_BENCH_IMAGE` to a published immutable image tag/digest, or build the
reviewed checkout; do not assume an unpublished image exists. A fixed image,
seeds, threads and data manifest
make runs auditable, not necessarily bitwise-identical across hardware.

Historical `reference_results/` instructions require their original code and
environment. They are unchanged and must not be relabeled as new 27-method
results. Budget approval, new plans/full-budget pilots, full scientific
reproduction and the deferred SPADE selection remain post-merge experimental
work, not prerequisites for merging this ownership fix.
