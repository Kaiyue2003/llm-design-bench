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
| Data mixture | Five-domain simplex plus model scale and training steps | Published `data-recipes` runs | Best Logged, Random Search, Sobol, Offline MLP, Standard GA adaptation, CMA-ES adaptation, REINFORCE adaptation, COMs adaptation, BDI adaptation |
| Synthetic BBO | Box-bounded continuous vectors | Seeded uniform samples over each function's bounds | Best Logged, Random Search, Sobol, Offline MLP, Standard GA adaptation, CMA-ES adaptation, REINFORCE adaptation, COMs adaptation, BDI adaptation |

The synthetic suite covers Many Local Minima, Bowl-Shaped, Plate-Shaped,
Valley-Shaped, Steep Ridges/Drops, and Other test problems. The three standard
additions and COMs are PyTorch adaptations. BDI is a lightweight adaptation
that uses differentiable RBF kernel regression instead of the original legacy
JAX/Neural Tangents runtime; see
[Method Notes](docs/METHODS.md) for the exact mechanisms and limitations.

## Installation

For the agreed LLM data-mixture experiment (scale-stratified visible data,
fixed-1B subset ablation, frozen configs and per-candidate artifacts), use the
dedicated [LLM-DM protocol workflow](docs/LLMDM_PROTOCOL.md). Its prepare/freeze
commands do not train models; execution is a separate explicit command. The
historical publication table remains unchanged.

For the frozen first batch, open [the Colab notebook](notebooks/LLMDM_Colab.ipynb)
and follow the [Colab guide](docs/COLAB.md). The
[release inputs](experiments/llmdm_forward_v1/README.md) contain 184 shared visible
observations (26 in the fixed-1B subset) and complete budgets for our 19 methods.
The notebook pins code/data versions, defaults to training disabled, and backs
up explicitly launched runs to Drive.

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

The command checkpoints raw per-seed rows and emits mean +/- sample standard
deviation summaries, a seed manifest, environment metadata, a GitHub Markdown
table, and an Overleaf-ready LaTeX table. The committed reference table and
its raw inputs are in
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

Run registered methods through the versioned, oracle-separated suite runner:

```bash
llm-design-bench-suite \
  --data-mixture \
  --data-recipes-root ../data-recipes \
  --method best_logged \
  --method random_search \
  --method sobol \
  --method offline_mlp \
  --method standard_ga \
  --method cma_es \
  --method reinforce \
  --method bo_qei \
  --method ga_on_gp \
  --method mc_dropout \
  --method tri_mentoring \
  --method ict \
  --method roma \
  --method ltr \
  --method match_opt \
  --method pgs \
  --method coms \
  --method bdi \
  --seed 38 --seed 39 --seed 40 --seed 41 \
  --seed 42 --seed 43 --seed 44 --seed 45 \
  --candidate-budget 128 \
  --results-dir results/unified_data_mixture
```

Add `--fixed-1b` for the ablation. Both modes use the same unfiltered
data-recipes normalization reference and the same 1B/19,500-step target.
When `--method` is omitted, the suite runs these nineteen registered methods.
The result labels remain **Standard GA adaptation**, **CMA-ES adaptation**,
**REINFORCE adaptation**, **BO-qEI adaptation**, **GA on GP adaptation**,
**MC-Dropout adaptation**, **Tri-Mentoring adaptation**, **ICT adaptation**,
**RoMA adaptation**, **LTR adaptation**, **MATCH-OPT adaptation**, **PGS adaptation**, **COMs adaptation**,
and **BDI adaptation**; none
is presented as an exact reproduction of the cited implementation. Source
audits are in
[`docs/BASELINE_SOURCE_AUDIT.md`](docs/BASELINE_SOURCE_AUDIT.md) and
[`docs/GP_UNCERTAINTY_BASELINE_AUDIT.md`](docs/GP_UNCERTAINTY_BASELINE_AUDIT.md),
with RoMA/ICT/Tri-Mentoring details in
[`docs/FORWARD_METHOD_SOURCE_AUDIT.md`](docs/FORWARD_METHOD_SOURCE_AUDIT.md).
LTR, MATCH-OPT, and PGS are independently implemented adaptations.
Acceptance requirements and
deliberate deviations are recorded in the
[ranking and policy source audit](docs/RANKING_POLICY_METHOD_SOURCE_AUDIT.md).
PGS combines its [certified transition layer](docs/PGS_TRANSITION_DESIGN.md)
with native CQL/SAC policy training and fixed-target projected-gradient rollout.
These methods have integration tests and reduced-budget smoke runs, not new
formal eight-seed publication results.

**SPADE adaptation (official-core-derived)** is also registered as `spade`.
It uses an attributed MIT-licensed PyTorch diffusion core, native Torch kNN,
and design-only constrained evolution at fixed target fidelity. See the
[source audit](docs/SPADE_SOURCE_AUDIT.md) and [settings](docs/METHODS.md#spade).
It has integration smoke validation, not new formal publication results.

## Python API

```python
from llm_design_bench import make
from llm_design_bench.optimizers import BackwardDistillationOptimizer

task = make("synthetic-ackley", logged_samples=256, seed=38)
optimizer = BackwardDistillationOptimizer(
    recommendations=64,
    seed=38,
    steps=100,
)
trace = optimizer.optimize(task)

print("best utility:", trace.recommendation_utility.max())
print("best objective:", -trace.recommendation_utility.max())
```

The oracle-separated method API currently registers `best_logged`,
`random_search`, `sobol`, `offline_mlp`, `standard_ga`, `cma_es`, `reinforce`,
`bo_qei`, `ga_on_gp`, `mc_dropout`, `tri_mentoring`, `ict`, `roma`, `ltr`,
`match_opt`, `pgs`, `coms`, and `bdi`:

```python
from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import OfflineProblem, RunContext

problem = OfflineProblem.from_task(task)
method = make_method("offline_mlp", epochs=100, particle_steps=100)
result = method.run(
    problem,
    RunContext(method_seed=38, candidate_budget=128),
)

# Only the evaluator may send result.candidates to task.predict(...).
```

The legacy optimizer API remains available for the existing CLI and reference
result workflows.

New multi-task experiments compose `BenchmarkTaskSpec` trial factories with
the same seed runner, producing a single versioned result schema with raw
per-seed rows, mean/sample-SD/SE summaries, failures, runtime, candidate
diagnostics, provenance, Markdown, and LaTeX. The exact columns and legacy-v1
conversion command are documented in the
[unified result schema](docs/RESULT_SCHEMA.md).

The frozen three-method result set has a machine-checked
[publication v1 audit](docs/RESULTS_AUDIT.md). The planned PyTorch integration
of all 24 methods compared by the SPADE paper is tracked in the
[method catalog](docs/METHOD_CATALOG.md); catalog membership does not imply
that a method is already implemented.

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
