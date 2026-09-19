# Reproducing Results

## Current entry points and release order

`llm-design-bench` and `llm-design-bench-llmdm` invoke the same frozen LLM-DM
application. `llm-design-bench-report` derives reports from saved rows without
training. The retired experiment CLIs are not alternate ways to run this release.

The code merge comes first. After merge, method owners approve their separate
training/search budgets; then freeze a new release plan, run full-budget seed-0
pilots, and run the eight formal seeds. The commands below describe that future
campaign, not an already approved budget or a reason to rerun historical results.

## 1. Install the reviewed revision

```bash
python -m pip install -e ".[dev]"
llm-design-bench --help
llm-design-bench methods
```

Use the merged release commit and a clean source tree before freezing. Keep its
dependency environment and source revision with the experiment. The formal
roster has 27 non-SPADE methods; listing methods neither trains nor loads data.
The generic synthetic task/Python API remains available for development checks.

## 2. Prepare one shared data bundle

Obtain a trusted `data-recipes` checkout and record its exact revision. Its logged
dataset and oracle checkpoints are external; do not substitute arbitrary pickle
or checkpoint files. Use real paths in place of the placeholders below:

```bash
llm-design-bench prepare \
  --data-recipes-root /path/to/data-recipes \
  --oracle-checkpoint /path/to/data-recipes/path/to/trusted/checkpoint.pt \
  --output /path/to/new/shared-data
```

Repeat `--oracle-checkpoint` for every checkpoint used by the oracle. Preparation
freezes the five-source order, source/asset hashes, visible row IDs, arrays and
reference identity without querying the oracle. The visible set combines the
0–40 utility percentiles within each model scale, including threshold ties.
All methods consume the same bundle; fixed-1B filters its visible 1B subset and
does not resplit. The reference range stays evaluator-only and is shared across
settings. Hashes establish identity, not the safety of third-party code.

## 3. Freeze approved method budgets

The owners supply a `methods.json` list of objects containing `method_id`, optional
`run_id`, and an explicit `kwargs` mapping. This small shape example is **not a
27-method experimental configuration**:

```json
[
  {"method_id": "best_logged", "kwargs": {}}
]
```

For a trained method, approve its settings before freezing. An explicit `{}`
means “use and freeze all current constructor defaults”; it is not a claim that
different methods share a budget. The final list must contain the selected
campaign's methods, not just the example reference above.

```bash
llm-design-bench freeze \
  --data-recipes-root /path/to/data-recipes \
  --data-bundle /path/to/shared-data \
  --methods-file /path/to/approved-methods.json \
  --experiment-id merged-llmdm-v1 \
  --output /path/to/new/plan.json
```

Freeze expands inherited defaults, checks reconstruction, and records package
and data identities. SPADE is rejected until its formal implementation is
selected. Preparation and freezing do not run model training.

## 4. Pilot, then formal runs

```bash
llm-design-bench run \
  --data-recipes-root /path/to/data-recipes \
  --data-bundle /path/to/shared-data --plan /path/to/plan.json \
  --phase pilot --setting multi_scale --device cpu \
  --results-dir /path/to/pilot

llm-design-bench run \
  --data-recipes-root /path/to/data-recipes \
  --data-bundle /path/to/shared-data --plan /path/to/plan.json \
  --phase formal --setting multi_scale --device cpu \
  --pilot-results /path/to/pilot --results-dir /path/to/formal
```

Pilot seed 0 uses the complete frozen method budget. Inspect cost, memory,
numerical stability and saved artifacts, not oracle scores for choosing settings.
Formal runs require verified pilots and use seeds 38–45. Each run returns K=128
candidates; the logged data does not change with the method seed. Target fidelity
is always 1B/19500 steps and utility is negative StackExchange cross-entropy.

Use `--setting fixed_1b` for the ablation or `both` for both settings; first obtain
the corresponding pilot coverage. `--run-id` and `--seed` can select shards.
`--device cuda` controls the method where a compatible PyTorch/CUDA runtime is
available; `--oracle-device` is independent. The existing precision policy uses
float64 for BDI, GA on GP and BO-qEI, and float32 for other methods; no mixed
precision is enabled.

`--resume` reuses identical verified successful attempts, not arbitrary CSV rows.
Inspect an interrupted attempt before supplying `--infrastructure-retry-reason`;
do not relabel algorithm failures or silently retry until a favorable result.
Changed code/data/settings require a new plan and output directory.

## 5. Inspect and report

Preserve the full run directory, not only downloaded CSV files. It contains
immutable per-attempt manifests/results, `candidates.npz`, `evaluation.npz`,
provenance and checksums, along with `method_seed_results.csv` and
`method_seed_summary.csv`. Failures remain recorded. Ranking requires complete
formal coverage; partial averages are not a completed benchmark.

```bash
llm-design-bench-report from-unified \
  --input-csv /path/to/formal/method_seed_results.csv \
  --results-dir /path/to/new/report
```

Reporting produces summaries and Markdown/LaTeX tables without retraining or
modifying its input CSV. Reports include mean, sample SD and SE with successful,
failed and missing coverage; additional confidence/range fields are preserved.
Report generation alone is not an independent audit of every original artifact.

Raw utility is maximized; `refnorm_*_score` uses the frozen full-logged utility
range. Scores are not clipped: above 1 means above logged best, not proof of a
global optimum. Do not concatenate different plans or historical splits into a
single ranking.

## Containers and historical archives

[Docker](DOCKER.md) runs this same workflow through `benchmark`, with a separate
small `smoke` check. It does not choose budgets or remove the pilot requirement.

Files under `reference_results/` are unchanged historical evidence. Reproducing
their old commands requires each archive's recorded original code and environment.
The old 14-method integration, publication, and 19-method forward tables do not
become new 27-method results by changing their labels. Historical configurations,
source pins, CSVs and documentation stay in their archive/history rather than
being installed as parallel active experiment workflows.
