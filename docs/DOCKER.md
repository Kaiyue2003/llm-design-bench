# Docker: the same frozen LLM-DM workflow

The container uses `llm-design-bench`, the same formal CLI as a native installation.
It does not select experimental budgets, freeze a real campaign automatically, or
download data. There are two Compose services: `benchmark` and `smoke`.

The image pins Python 3.11.15 and uv 0.11.8, installs the frozen `uv.lock`, and sets
common BLAS thread counts to one. The provided container workflow is CPU-only;
it does not establish CUDA or cross-platform numerical parity.

## Build and inspect without training

```bash
docker compose run --build --rm benchmark
docker compose run --rm benchmark methods
```

The first command builds the current checkout and prints help. The second lists
the 27 non-SPADE formal methods. No data or oracle is loaded by these commands.
`llm-design-bench-llmdm` is a native alias for the same application, not another
Docker service or experiment protocol.

The shell/PowerShell wrappers only forward arguments to `benchmark`; with no
arguments they print help:

```bash
./scripts/reproduce_docker.sh methods
```

```powershell
.\scripts\reproduce_docker.ps1 methods
```

## Small engineering smoke check

```bash
docker compose run --build --rm smoke
```

`scripts/container_smoke.py` creates invented logs and a fake oracle in a fresh
directory. It exercises the real prepare/freeze/run CLI and:

- gives all 27 methods tiny engineering-test budgets for pilot seed 0;
- checks K=128 candidate outputs and saved attempt artifacts;
- runs only Best Logged's formal seed 38 and resumes it without another oracle
  call or a new attempt;
- verifies that the formal shard remains incomplete: seven missing seeds,
  `rank_eligible=false`.

Output is under
`results/docker/container-smoke/invented-smoke-*/run/`, including
`smoke_verification.json`. No real checkpoint, dataset, old result or approved
budget is read or changed. A successful smoke check is not a full-budget pilot,
a 27-method formal result, or evidence of real-data optimization quality.

## Supply external inputs explicitly

Compose maps these host directories:

| Host setting | Default | Container mount |
| --- | --- | --- |
| `ASSETS_DIR` | `./assets` | `/assets`, read-only |
| `RESULTS_DIR` | `./results/docker` | `/results`, writable and durable |

Create those directories and place a trusted `data-recipes` checkout under
`assets/data-recipes`, or set `ASSETS_DIR` to your own parent directory. An existing
verified bundle/plan can also be placed under assets. Neither the image nor
Compose automatically clones or updates the upstream checkout. Record the exact
revision and trust its pickle/checkpoint files before use.

After the code merge and budget approval, the following examples use container
paths. Replace the checkpoint placeholder with every real checkpoint required
by the oracle; repeat the option as necessary.

```bash
docker compose run --rm benchmark prepare \
  --data-recipes-root /assets/data-recipes \
  --oracle-checkpoint /assets/data-recipes/path/to/trusted/checkpoint.pt \
  --output /results/shared-data

docker compose run --rm benchmark freeze \
  --data-recipes-root /assets/data-recipes \
  --data-bundle /results/shared-data \
  --methods-file /assets/approved-methods.json \
  --experiment-id merged-llmdm-v1 --output /results/plan.json
```

The methods-file schema and approval policy are in
[Reproducing Results](REPRODUCING.md). There is no implicit full-experiment budget
preset. Preparation/freezing do not query the oracle. Output paths must be new;
do not overwrite an existing campaign.

## Run the approved pilot and formal campaign

```bash
docker compose run --rm benchmark run \
  --data-recipes-root /assets/data-recipes \
  --data-bundle /results/shared-data --plan /results/plan.json \
  --phase pilot --setting multi_scale --device cpu \
  --results-dir /results/pilot

docker compose run --rm benchmark run \
  --data-recipes-root /assets/data-recipes \
  --data-bundle /results/shared-data --plan /results/plan.json \
  --phase formal --setting multi_scale --device cpu \
  --pilot-results /results/pilot --results-dir /results/formal
```

The runtime requires matching verified pilots. Use `fixed_1b` or `both` only
with the corresponding pilot coverage. Selection, explicit resume and
infrastructure-retry rules are identical to the native CLI. A changed image
source/configuration cannot silently inherit old pilots or successful rows.

To derive reports into a separate directory:

```bash
docker compose run --rm --entrypoint llm-design-bench-report benchmark \
  from-unified --input-csv /results/formal/method_seed_results.csv \
  --results-dir /results/report
```

Native and container runs use the same per-attempt manifests, result JSON,
candidate/evaluation NPZs, environment records and integrity/resume checks.
Preserve the full results directory, including failed attempts, not just CSV
summaries. The former raw-run verification/environment scripts and publication
services are not part of this workflow.

## Images, reproducibility and historical results

For a reviewed image that has actually been published, set
`LLM_DESIGN_BENCH_IMAGE` to its immutable commit tag or digest. Do not assume an
unpublished integration image exists; build this checkout for local validation.
The mutable `main` tag is not an experiment identity.

A fixed image, seeds, threads and data manifest make runs auditable, but do not
guarantee bitwise equality across hardware or numerical libraries. Record the
image/release and inspect configuration, coverage, failures and numerical
tolerances when independently reproducing results.

Historical instructions under `reference_results/` refer to their original code,
environment and old services. Those archives are unchanged and must be replayed
from their recorded version, not relabeled as current 27-method results. Full
real-data reproduction, budget approval, new plans/pilots and SPADE selection
happen after code integration; they are not prerequisites for merging this work.
