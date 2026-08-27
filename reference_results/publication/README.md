# Seeded Publication Benchmark

Normalized maximum score (100th percentile of K=128 recommendations), reported as mean +/- sample SD across 8 independent seeds. Higher is better.

| Method | LLM-DM | Ackley | Schaffer N. 2 | Sum of Different Powers | Matyas | Power Sum | Rosenbrock | Michalewicz | Hartmann 6-D | Shekel | Mean rank | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D(best) | 0.881 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | -- | -- |
| Best Logged | <u>0.995 +/- 0.000</u> | 1.000 +/- 0.000 | **1.000 +/- 0.000** | 1.000 +/- 0.000 | **1.000 +/- 0.000** | **1.000 +/- 0.000** | **1.000 +/- 0.000** | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 2.10 | 2.50 |
| COM | 0.992 +/- 0.007 | <u>1.108 +/- 0.145</u> | <u>0.845 +/- 0.198</u> | <u>1.001 +/- 0.002</u> | <u>1.000 +/- 0.001</u> | <u>1.000 +/- 0.000</u> | <u>1.000 +/- 0.000</u> | **1.082 +/- 0.104** | <u>1.274 +/- 0.177</u> | **1.762 +/- 0.750** | 1.90 | 2.00 |
| BDI | **1.003 +/- 0.000** | **1.219 +/- 0.271** | 0.533 +/- 0.092 | **1.001 +/- 0.002** | 0.998 +/- 0.003 | 0.999 +/- 0.001 | 1.000 +/- 0.000 | <u>1.017 +/- 0.184</u> | **1.308 +/- 0.205** | <u>1.484 +/- 0.649</u> | 2.00 | 2.00 |

Bold is best and underlining is second best within each task, based on the mean score.

## Experiment Contract

- Seeds: `38, 39, 40, 41, 42, 43, 44, 45`. The synthetic logged-dataset seed and optimizer seed both equal the trial seed.
- Data-mixture logged observations are fixed upstream data; only the optimizer seed varies by trial.
- Score: `(generated best utility - full logged minimum utility) / (full logged maximum utility - full logged minimum utility)`. Scores above 1 are allowed.
- Data-mixture training visibility: utility percentiles [0, 40].
- `D(best)` is the best optimizer-visible logged utility. LLM-DM logged runs retain their recorded model scale and training step, while method recommendations are evaluated at the target 1B/19,500-step fidelity.
- Synthetic functions: `ackley, schaffer2, sum_different_powers, matyas, power_sum, rosenbrock, michalewicz, hartmann6, shekel`.
- Uncertainty: sample standard deviation, not standard error.
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
