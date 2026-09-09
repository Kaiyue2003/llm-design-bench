# Seeded Publication Benchmark

Normalized maximum score (100th percentile of K=128 recommendations), reported as mean +/- standard error across 8 independent seeds. Higher is better.

| Method | LLM-DM | Ackley | Schaffer N. 2 | Sum of Different Powers | Matyas | Power Sum | Rosenbrock | Michalewicz | Hartmann 6-D | Shekel | Mean rank | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D(best) | 0.881 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | -- | -- |
| Best Logged | <u>0.995 +/- 0.000</u> | 1.000 +/- 0.000 | **1.000 +/- 0.000** | 1.000 +/- 0.000 | <u>1.000 +/- 0.000</u> | **1.000 +/- 0.000** | **1.000 +/- 0.000** | **1.000 +/- 0.000** | <u>1.000 +/- 0.000</u> | 1.000 +/- 0.000 | 1.90 | 2.00 |
| COM | 0.989 +/- 0.005 | <u>1.185 +/- 0.130</u> | 0.512 +/- 0.034 | **1.001 +/- 0.001** | **1.000 +/- 0.000** | 0.965 +/- 0.006 | 0.931 +/- 0.005 | 0.238 +/- 0.031 | 0.176 +/- 0.027 | <u>1.557 +/- 0.102</u> | 2.40 | 3.00 |
| BDI | **1.003 +/- 0.000** | **1.277 +/- 0.116** | <u>0.527 +/- 0.036</u> | <u>1.000 +/- 0.001</u> | 0.999 +/- 0.000 | <u>1.000 +/- 0.000</u> | <u>1.000 +/- 0.000</u> | <u>0.968 +/- 0.077</u> | **1.306 +/- 0.081** | **1.711 +/- 0.258** | 1.70 | 2.00 |

Bold is best and underlining is second best within each task, based on the mean score.

## Experiment Contract

- Seeds: `38, 39, 40, 41, 42, 43, 44, 45`. The synthetic logged-dataset seed and optimizer seed both equal the trial seed.
- Data-mixture logged observations are fixed upstream data; only the optimizer seed varies by trial.
- Score: `(generated best utility - full logged minimum utility) / (full logged maximum utility - full logged minimum utility)`. Scores above 1 are allowed.
- Data-mixture training visibility: utility percentiles [0, 40].
- `D(best)` is the best optimizer-visible logged utility. LLM-DM logged runs retain their recorded model scale and training step, while method recommendations are evaluated at the target 1B/19,500-step fidelity.
- Synthetic functions: `ackley, schaffer2, sum_different_powers, matyas, power_sum, rosenbrock, michalewicz, hartmann6, shekel`.
- Displayed uncertainty: standard error.
- `task_summary.csv` also records sample SD, standard error, 95% Student-t CI, and observed seed range.
- Candidate count: `K=128` for every method and trial.

## Selection And Method Provenance

The synthetic subset is the union of the previously committed single-seed COM and BDI winner in each test-problem category. It is intentionally performance-selected for a compact descriptive table and must not be presented as an unbiased all-task comparison.

COM is a native PyTorch reimplementation of the conservative objective-model structure from [design-baselines](https://github.com/brandontrabucco/design-baselines).
BDI is the repository's native PyTorch/RBF-kernel adaptation of the forward/backward distillation structure in the [official BDI repository](https://github.com/GGchen1997/BDI). It does not claim numerical equivalence to the original JAX/Neural Tangents implementation.

## Reproduction Files

- [Raw per-seed runs](raw_runs.csv)
- [Aggregated task summary](task_summary.csv)
- [Rank summary](rank_summary.csv)
- [Seed manifest](seed_manifest.csv)
- [Run metadata](run_metadata.json)
- [Overleaf table](seeded_benchmark_table.tex)
- [Table 1-style mean +/- SE report](TABLE1_STYLE.md)
- [Table 1-style Overleaf table](seeded_benchmark_table_se.tex)
