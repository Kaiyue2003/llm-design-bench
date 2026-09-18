# LLM Design Bench

A Python benchmark for offline black-box optimization of LLM pre-training data
mixtures. Methods learn from logged experiments, return candidate mixtures, and
are evaluated by the separate
[data-recipes](https://github.com/namkoong-lab/data-recipes) simulator; this
workflow does not pre-train new LLMs. Every task exposes maximization utility:
for cross-entropy, `utility = -loss`.

## One formal workflow

`llm-design-bench` is the default frozen LLM-DM entry point.
`llm-design-bench-llmdm` is an explicit alias for the **same application**, not a
second experiment workflow. Reporting uses `llm-design-bench-report`.

The integrated formal roster contains **27 non-SPADE methods**: 18 reference,
forward-model and search methods, plus nine inverse/generative/other methods.
Their implementation ownership, adaptations and precision are documented in
[Integrated Methods](docs/INTEGRATED_METHODS.md) and [Method Notes](docs/METHODS.md).
SPADE remains available in the Python research API, but formal plans reject it
until the two development lines' implementations are reconciled.

| Setting | Agreed rule |
| --- | --- |
| Main task | Multi-scale logged data → target 1B data-mixture recommendation |
| Objective | StackExchange cross-entropy; maximize its negative |
| Visible data | Utility percentiles 0–40 **within each model scale**, then merged |
| Target fidelity | 1B, 19500 training steps |
| fixed-1B ablation | 1B subset of the same visible manifest; no new split |
| Candidates | K=128 per method/seed; duplicates allowed and counted |
| Pilot | Full configured budget, seed 0; cost/stability/artifact checks |
| Formal seeds | 38–45; logged data stays fixed |

The code integration does not select final training budgets. **After merge**, the
method owners agree their budgets, freeze a new plan against the merged release,
and run pilots before the eight-seed experiments. Full experimental reproduction
and the deferred SPADE decision are not prerequisites for merging this code.

## Installation

Install the checkout you intend to run:

```bash
git clone https://github.com/Kaiyue2003/llm-design-bench.git
cd llm-design-bench
python -m pip install -e ".[dev]"
llm-design-bench --help
llm-design-bench methods
```

Python 3.11 and 3.12 are supported. The package uses PyTorch, with GPyTorch for
GA on GP, BO-qEI and GABO; BDI uses kernel ridge regression, not a GP posterior.
During integration, check out the reviewed integration revision rather than
assuming the public default branch already contains these changes.

## Running a formal experiment

The only experiment sequence is:

```text
trusted data-recipes checkout + checkpoints
  → prepare shared visible-data manifest
  → freeze approved method settings and source identity
  → full-budget pilot (seed 0)
  → formal runs (seeds 38–45)
  → verify and summarize saved artifacts
```

`methods`, `prepare`, and `freeze` do not train or query the oracle. `run` does.
Use the complete path-based examples in [Reproducing Results](docs/REPRODUCING.md)
after the post-merge budget decision. Docker runs this same application:

```bash
docker compose run --build --rm benchmark methods
docker compose run --build --rm smoke
```

The first command only lists methods. The second checks a tiny invented-data
workflow, not a real LLM experiment. See [Docker](docs/DOCKER.md) for asset mounts,
explicit experiment arguments and smoke-test limits.

The upstream logged data and simulator checkpoints are external inputs. Use only
a trusted checkout: their pickle/checkpoint formats can execute code when loaded.
The manifest records hashes, and the simulator is only exposed to the evaluator.

## Python API and synthetic tasks

The generic simplex/box task API and 47 synthetic functions remain available for
component tests and research. They are not a second formal LLM-DM command line.

```python
from llm_design_bench import OfflineProblem, RunContext, make
from llm_design_bench.optimizers import make_method

task = make("synthetic-ackley", logged_samples=32, seed=0)
problem = OfflineProblem.from_task(task)
result = make_method("random_search").run(
    problem, RunContext(method_seed=0, candidate_budget=4)
)

# Evaluation is outside the method boundary, after candidate generation.
batch = task.at_target_fidelity(result.candidates.detach().cpu().numpy())
print(task.predict(batch))
```

See [synthetic_quickstart.py](examples/synthetic_quickstart.py) for a tiny API
demonstration and [data_recipes_quickstart.py](examples/data_recipes_quickstart.py)
for proposing from an existing frozen visible-data bundle without querying the
oracle. Neither example freezes a release or creates formal results.

## Results and history

New run output belongs in the Git-ignored `results/` directory. The runtime saves
per-attempt candidates/evaluations, configuration, provenance, checksums, failures,
and per-seed/summary tables. Resume only reuses verified compatible attempts;
changing code, data or settings requires a new plan and output directory.

Committed [reference results](reference_results/) are immutable historical
archives. Their documented old commands require the original recorded commit and
environment; the retired experiment entry points are not installed here. Those
tables, including the 14-method integration snapshot and the 19-method forward
campaign, must not be merged into a new 27-method formal ranking.

## Development

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```

Read the [protocol](docs/BENCHMARK_PROTOCOL.md), [architecture](docs/ARCHITECTURE.md),
[method integration guide](docs/ADDING_METHODS.md), and
[integration scope](docs/INTEGRATION.md). Contribution requirements are in
[CONTRIBUTING.md](CONTRIBUTING.md). Released under the [MIT License](LICENSE).
