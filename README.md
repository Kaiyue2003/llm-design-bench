# LLM Design Bench

`llm-design-bench` is a reproducible Python benchmark for offline design
optimization. It brings two experimental settings under one API:

- LLM pre-training data-mixture optimization through the simulator and logged
  runs from [`namkoong-lab/data-recipes`](https://github.com/namkoong-lab/data-recipes).
- Continuous black-box optimization over 47 synthetic functions organized by
  the SFU test-problem categories and backed, where available, by
  [`bayeso-benchmarks`](https://github.com/jungtaekkim/bayeso-benchmarks).

All tasks expose a maximization utility. For loss-minimization problems the
package uses `utility = -objective`, so a larger utility is always better.

## What Is Included

| Suite | Designs | Logged dataset | Methods |
| --- | --- | --- | --- |
| Data mixture | Five-domain simplex plus model scale and training steps | Published `data-recipes` runs | All 14 registered offline methods; legacy random/Sobol runners |
| Synthetic BBO | Box-bounded continuous vectors | Seeded uniform samples over each function's bounds | All 14 registered offline methods |

The synthetic suite covers Many Local Minima, Bowl-Shaped, Plate-Shaped,
Valley-Shaped, Steep Ridges/Drops, and Other test problems. COM is a
conservative objective-model implementation. BDI is a lightweight adaptation
that uses differentiable RBF kernel regression instead of the original legacy
JAX/Neural Tangents runtime; see [Method Notes](docs/METHODS.md) for the exact
mechanisms and limitations.

The runnable PyTorch registry contains `best_logged`, `offline_mlp`, `coms`,
`bdi`, `cbas`, `mins`, `ddom`, `gabo`, `gtg`, `rgd`, `bonet`, `demo`, `root`,
and `spade`. The ten additional methods are **continuous PyTorch adaptations**,
with source revisions and algorithmic substitutions recorded in
[Additional Methods](docs/ADDITIONAL_METHODS.md). They are not claims of
checkpoint compatibility or reproduction of the original papers' tables.

Run every method twice in Docker and verify the saved data and candidates:

```bash
docker compose run --build --rm all-methods-smoke
```

Run the eight-seed, ten-task suite with all methods:

```bash
docker compose run --build --rm all-methods
```

The full suite trains models and can take substantially longer than the smoke
check. Results, exact input arrays, candidates, configuration, and file hashes
are persisted under `results/docker/`.

## Installation

Install directly from GitHub:

```bash
pip install "llm-design-bench @ git+https://github.com/Kaiyue2003/llm-design-bench.git"
```

Or clone the repository for development:

```bash
git clone https://github.com/Kaiyue2003/llm-design-bench.git
cd llm-design-bench
python -m pip install -e ".[dev]"
python -m pytest -q
```

Python 3.11 and 3.12 are supported. PyTorch is installed because the MLP, COM,
and BDI implementations optimize differentiable surrogate models.

For the fully pinned container environment, run:

```bash
docker compose run --rm smoke
```

See [Docker Reproduction](docs/DOCKER.md) for the full eight-seed command,
external Data Recipes pin, result volume, and immutable GHCR tags.

## Quickstart

Run all 47 synthetic tasks with the reference configuration:

```bash
llm-design-bench-synthetic \
  --logged-samples 256 \
  --recommendations 64 \
  --epochs 100 \
  --particle-steps 100 \
  --bdi-steps 100 \
  --seed 38
```

Run a small reproducibility check first:

```bash
llm-design-bench-synthetic \
  --function ackley \
  --function booth \
  --logged-samples 64 \
  --recommendations 8 \
  --epochs 10 \
  --particle-steps 10 \
  --bdi-steps 10 \
  --results-dir results/smoke
```

The main synthetic outputs are:

- `results/synthetic_bo_results.csv`
- `results/synthetic_bo_summary.png`
- `results/synthetic_categories/`
- `results/synthetic_categories_top1/`

The normalized best-utility score is

```text
(generated best utility - logged minimum utility)
-------------------------------------------------
 (logged maximum utility - logged minimum utility)
```

Higher is better. `1.0` matches the maximum utility in the logged dataset;
values above `1.0` mean that the optimizer generated a candidate better than
every logged observation. The score is not clipped and does not claim that the
global optimum has been reached.

## Seeded Publication Table

Run the compact data-mixture and selected synthetic suite across the eight
publication seeds:

```bash
llm-design-bench-publication \
  --data-recipes-root ../data-recipes \
  --seed 38 --seed 39 --seed 40 --seed 41 \
  --seed 42 --seed 43 --seed 44 --seed 45 \
  --recommendations 128 \
  --results-dir results/publication
```

The command checkpoints raw per-seed rows and emits both mean +/- sample
standard deviation and Table 1-style mean +/- standard error summaries.
`task_summary.csv` also contains 95% Student-t confidence intervals
and observed seed ranges. Seed manifests, environment metadata, GitHub
Markdown tables, and Overleaf-ready LaTeX tables are written from the same raw
rows. The committed reference table and its raw inputs are in
[`reference_results/publication/`](reference_results/publication/).

The synthetic publication subset is the union of the prior single-seed COM
and BDI category winners. This makes the compact table useful for method
inspection, but performance claims over the full synthetic suite must use all
47 tasks instead of this selected subset.

## Data-Mixture Benchmark

The package does not redistribute the upstream simulator checkpoints or logged
runs. Clone `data-recipes` next to this repository, or set
`DATA_RECIPES_ROOT`:

```bash
git clone https://github.com/namkoong-lab/data-recipes.git
git clone https://github.com/Kaiyue2003/llm-design-bench.git
cd llm-design-bench
python -m pip install -e ".[dev]"
```

Run the online-style random/Sobol baselines:

```bash
llm-design-bench \
  --data-recipes-root ../data-recipes \
  --queries 256 \
  --reference-queries 2048 \
  --recommendations 128 \
  --seed 38
```

Run the logged-data-only MLP, COM, and BDI benchmark:

```bash
llm-design-bench-offline \
  --data-recipes-root ../data-recipes \
  --reference-queries 2048 \
  --recommendations 128 \
  --epochs 100 \
  --particle-steps 100 \
  --bdi-steps 100 \
  --train-min-percentile 0 \
  --train-max-percentile 40 \
  --seed 38
```

The default offline split exposes only logged observations between the 0th and
40th utility percentiles to the optimizers. Metric normalization still uses
the full logged dataset. Pass `--train-max-percentile 100` to expose all logged
observations.

For the fixed-1B ablation, keep the target fidelity unchanged and filter the
visible logged data to 1B runs:

```bash
llm-design-bench-offline \
  --data-recipes-root ../data-recipes \
  --logged-model-scale 1000 \
  --reference-queries 2048 \
  --recommendations 128 \
  --epochs 100 \
  --particle-steps 100 \
  --bdi-steps 100 \
  --train-min-percentile 0 \
  --train-max-percentile 40 \
  --seed 38
```

The Python registry also exposes this setting as `make("data-recipes-1b")`.

## Python API

```python
import numpy as np

from llm_design_bench import OfflineProblem, RunContext, make
from llm_design_bench.optimizers import make_method

task = make("synthetic-ackley", logged_samples=256, seed=38)
problem = OfflineProblem.from_task(task)
method = make_method("bdi", steps=100)
result = method.run(
    problem,
    RunContext(method_seed=38, dataset_seed=38, split_seed=38, candidate_budget=64),
)

# Oracle evaluation remains outside the method boundary.
batch = task.at_target_fidelity(result.candidates.detach().cpu().numpy())
utility = np.asarray(task.predict(batch))
print("best utility:", utility.max())
print("best objective:", -utility.max())
```

See [synthetic_quickstart.py](examples/synthetic_quickstart.py) and
[data_recipes_quickstart.py](examples/data_recipes_quickstart.py) for runnable
examples.

## Reproducing Results

The exact commands, expected files, seed policy, and comparison procedure are
documented in [REPRODUCING.md](docs/REPRODUCING.md). Compact reference outputs
are committed under [`reference_results/`](reference_results/) so that a fresh
run can be checked without relying on screenshots alone.

CSV utility columns follow two conventions:

- `raw_*_utility`: utility on the original maximization scale.
- `refnorm_*_score`: utility min-max normalized against the logged reference.

Generated files go to `results/`, which is intentionally ignored by Git.

## Development

The stable offline-method data boundary, candidate rules, seed policy, and
implementation-provenance requirements are defined in
[`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md).
The concrete tensor flow is documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and the PyTorch port
checklist and method skeleton are in
[`docs/ADDING_METHODS.md`](docs/ADDING_METHODS.md).

Using `pip`:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
python -m twine check dist/*
```

Using `uv`:

```bash
uv sync
uv run pytest -q
```

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). This project
is released under the [MIT License](LICENSE).
