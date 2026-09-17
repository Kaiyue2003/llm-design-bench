# LLM Design Bench

`llm-design-bench` is a Python benchmark for offline black-box optimization.
Its primary experiment recommends LLM pre-training data mixtures using logged
observations and the simulator from
[`namkoong-lab/data-recipes`](https://github.com/namkoong-lab/data-recipes).
Methods only see the shared visible dataset; the evaluator scores their final
candidates at target fidelity.

All tasks expose a maximization utility. For loss-minimization problems,
`utility = -objective`, so a larger utility is always better.

## Current benchmark

The formal LLM data-mixture experiment uses:

- five mixture dimensions: Wikipedia, StackExchange, GitHub, ArXiv and Book;
- utility percentiles 0-40 within each model scale, merged into one shared
  visible dataset; model scale and training steps remain conditions;
- target fidelity 1B / 19,500 training steps and StackExchange cross entropy;
- exactly 128 candidates per method run;
- one full-budget pilot with seed 0, followed by formal seeds 38-45;
- a fixed-1B ablation that takes the 1B subset of the main visible data without
  splitting again.

The [GPyTorch v2 result archive](reference_results/llmdm_forward_gpytorch_v2/README.md)
contains 152 completed multi-scale results: 19 methods with eight seeds each.
It does **not** include fixed-1B results. These are simulator evaluations, not
new LLM pre-training runs. Historical tables in other archive directories use
different protocols and must not be merged into this comparison.

The registry includes Best Logged, Random Search, Sobol, Offline MLP, Standard
GA, CMA-ES, REINFORCE, BO-qEI, GA on GP, MC-Dropout, COMs, BDI, RoMA, ICT,
Tri-Mentoring, LTR, MATCH-OPT, PGS and SPADE. Published-method implementations
retain their adaptation labels; they are not automatically exact reproductions.
The GP methods use GPyTorch, while BDI remains a differentiable RBF kernel-ridge
adaptation, not the original JAX/Neural Tangents implementation. See
[Method Notes](docs/METHODS.md) and the [method catalog](docs/METHOD_CATALOG.md).

The package also retains 47 synthetic box-constrained tasks, organized by the
SFU test-problem categories and backed where available by
[`bayeso-benchmarks`](https://github.com/jungtaekkim/bayeso-benchmarks).
They are available through the same oracle-separated method API and the
development suite runner, not a second formal LLM-DM workflow.

## Installation

For development:

```bash
git clone https://github.com/Kaiyue2003/llm-design-bench.git
cd llm-design-bench
python -m pip install -e ".[dev]"
```

Python 3.11 and 3.12 are checked by CI. Reproducing a frozen result additionally
requires that release's recorded code, dependencies, data and runtime policy;
installing the latest branch is not a replacement for its pinned environment.

After updating an existing editable installation, rerun
`python -m pip install -e ".[dev]"` to refresh installed command entry points.
Do not use stale legacy launchers left in an old environment; if necessary,
install the current package into a clean environment.

The package does not redistribute upstream simulator checkpoints or logged
runs. Obtain a trusted `data-recipes` checkout and supply its path through the
formal workflow. Its pickle/checkpoint files and dynamically imported code are
trusted inputs, not safe formats for untrusted downloads.

## Formal LLM-DM workflow

**Use `llm-design-bench` (an alias of `llm-design-bench-llmdm`) for the formal
experiment.** Both commands expose the same prepare/freeze/run workflow:

```bash
llm-design-bench --help
llm-design-bench-llmdm --help
```

Follow the [LLM-DM protocol](docs/LLMDM_PROTOCOL.md) for the shared data bundle,
expanded method budgets, pilot review and formal run commands. Preparation and
freezing do not train models; `run` explicitly launches training/evaluation.
New experiments must freeze their own source identity and plan before running.

For the existing GPyTorch v2 release, use
[LLMDM_GPyTorch_Colab.ipynb](notebooks/LLMDM_GPyTorch_Colab.ipynb) and the
[automated Colab guide](docs/COLAB_GPYTORCH.md). The notebook pins code/data,
runs the selected full-budget pilots, requires an explicit pilot review, then
queues the formal seeds. It preserves interrupted attempts and skips verified
completed runs. The [batch guide](docs/COLAB_BATCH.md) describes recovery and
Drive backups.

Current-source refactoring changes the package fingerprint. It does not rewrite
the existing [release](experiments/llmdm_forward_gpytorch_v2/release.json), its
plan, pinned notebook or results, and does not require rerunning that archive.
To resume or reproduce that release, keep its pinned environment; never bypass
the fingerprint checks or substitute the current checkout into its old plan.

The old online search entry point and the separate `-offline`, `-publication`
and `-synthetic` execution commands have been retired. There is no second
legacy execution workflow in the current package. See
[reproduction and archive notes](docs/REPRODUCING.md) if you need to inspect
historical artifacts; preserving them does not imply they belong in the final
benchmark paper.

## API quickstart and development tasks

For a minimal data-mixture API demo, see [Quickstart](docs/QUICKSTART.md) and
[data_recipes_quickstart.py](examples/data_recipes_quickstart.py). The demo is
neither a frozen pilot nor a formal experiment.

For a small synthetic smoke run without upstream LLM data:

```bash
llm-design-bench-suite \
  --function ackley \
  --function booth \
  --method random_search \
  --seed 0 \
  --logged-samples 64 \
  --candidate-budget 8 \
  --results-dir results/smoke
```

`llm-design-bench-suite` is the general-purpose method/task development runner.
It does not load a frozen LLM-DM manifest or plan, so its outputs do not qualify
as the formal main experiment or ablation. Omitting `--method` selects all 19
registered methods; omitting `--seed` selects seeds 38-45. Select explicit
methods/seeds for development rather than inadvertently launching a full suite.
Synthetic tasks themselves remain supported after retirement of the old
synthetic command.

The Python API uses the same method boundary:

```python
from llm_design_bench import make
from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import OfflineProblem, RunContext

task = make("synthetic-ackley", logged_samples=64, seed=0)
problem = OfflineProblem.from_task(task)
method = make_method("random_search")
result = method.run(problem, RunContext(method_seed=0, candidate_budget=8))

# Only the evaluator may pass the final candidates to task.predict(...).
```

Multi-task development runs compose `BenchmarkTaskSpec` trial factories with
the seed runner. Methods never receive the oracle. See the
[benchmark contract](docs/BENCHMARK_PROTOCOL.md),
[method notes](docs/METHODS.md), and
[synthetic quickstart](examples/synthetic_quickstart.py).

## Results and reporting

Formal runs retain per-candidate artifacts, per-seed outcomes, resolved
configuration and provenance. Summaries include mean, sample standard
deviation, standard error, and success/failure/missing coverage. Pilot results
and incomplete formal coverage are not eligible for formal ranking.

The normalized utility score is

```text
(candidate utility - logged reference minimum utility)
/ (logged reference maximum utility - logged reference minimum utility)
```

It is not clipped. A score above 1 exceeds the historical logged reference
maximum; it is not a percentage improvement or proof of a global optimum.
Both the main experiment and fixed-1B ablation use the same full multi-scale
reference. CSV `raw_*_utility` columns retain the original maximization scale;
`refnorm_*_score` columns use this normalization.

`llm-design-bench-report` rebuilds reports from saved rows and supports a
read-only conversion of historical publication-v1 inputs into a new output
directory. It does not train methods, change old source files, or make results
from different protocols comparable. See the
[result schema](docs/RESULT_SCHEMA.md) and
[reproduction notes](docs/REPRODUCING.md).

Generated output belongs in the Git-ignored `results/` directory. Committed
snapshots under [`reference_results/`](reference_results/) remain unchanged.

## Development

See [code structure](docs/CODE_STRUCTURE.md) for the execution, statistics,
persistence and Colab module boundaries, and [CONTRIBUTING.md](CONTRIBUTING.md)
for lint, formatting and incremental strict type checks.

```bash
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
