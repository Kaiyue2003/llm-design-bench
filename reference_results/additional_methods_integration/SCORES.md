# Simulation integration scores (two-epoch configuration)

**Scope: short-budget integration runs of continuous PyTorch adaptations. These scores do not establish original-paper performance parity.**


Normalized maximum score (100th percentile of K=8 recommendations), reported as mean +/- sample SD across 8 independent seeds. Higher is better.

| Method | LLM-DM | Ackley | Schaffer N. 2 | Sum of Different Powers | Matyas | Power Sum | Rosenbrock | Michalewicz | Hartmann 6-D | Shekel | Mean rank | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D(best) | 0.881 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | -- | -- |
| Best Logged | 0.976 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | **1.000 +/- 0.000** | **1.000 +/- 0.000** | **1.000 +/- 0.000** | **1.000 +/- 0.000** | **1.000 +/- 0.000** | 1.000 +/- 0.000 | 3.30 | 2.00 |
| COM | 0.979 +/- 0.002 | 0.903 +/- 0.235 | 0.859 +/- 0.306 | 0.993 +/- 0.009 | 0.999 +/- 0.004 | 0.999 +/- 0.001 | 1.000 +/- 0.001 | 0.905 +/- 0.251 | 0.973 +/- 0.113 | 0.932 +/- 0.262 | 5.55 | 5.00 |
| BDI | 0.979 +/- 0.002 | 0.952 +/- 0.230 | 0.810 +/- 0.276 | 0.996 +/- 0.005 | <u>1.000 +/- 0.007</u> | <u>1.000 +/- 0.001</u> | <u>1.000 +/- 0.001</u> | <u>0.930 +/- 0.245</u> | <u>0.975 +/- 0.103</u> | 1.075 +/- 0.257 | 4.00 | 3.00 |
| Offline MLP | 0.979 +/- 0.002 | 0.903 +/- 0.235 | 0.859 +/- 0.306 | 0.993 +/- 0.009 | 0.999 +/- 0.004 | 0.999 +/- 0.001 | 1.000 +/- 0.001 | 0.905 +/- 0.251 | 0.973 +/- 0.113 | 0.932 +/- 0.262 | 5.45 | 5.25 |
| CbAS | 0.985 +/- 0.011 | 0.795 +/- 0.873 | 0.896 +/- 0.464 | 0.959 +/- 0.067 | 0.993 +/- 0.012 | 0.996 +/- 0.007 | 0.992 +/- 0.007 | 0.662 +/- 0.353 | 0.469 +/- 0.260 | 0.836 +/- 0.363 | 7.20 | 7.50 |
| MINs | 0.967 +/- 0.002 | 1.136 +/- 0.476 | <u>1.983 +/- 1.174</u> | **1.007 +/- 0.017** | 0.999 +/- 0.005 | 0.983 +/- 0.011 | 0.996 +/- 0.003 | 0.244 +/- 0.258 | 0.307 +/- 0.210 | **1.603 +/- 1.387** | 6.00 | 4.50 |
| DDOM | <u>1.003 +/- 0.002</u> | 0.066 +/- 0.034 | 0.534 +/- 0.237 | -0.179 +/- 0.189 | 0.950 +/- 0.014 | 0.639 +/- 0.462 | 0.831 +/- 0.053 | 0.017 +/- 0.048 | 0.048 +/- 0.039 | 0.211 +/- 0.212 | 12.20 | 13.00 |
| GABO | 0.969 +/- 0.003 | **1.220 +/- 0.710** | **2.255 +/- 2.052** | <u>1.006 +/- 0.021</u> | 0.998 +/- 0.005 | 0.986 +/- 0.009 | 0.995 +/- 0.006 | 0.334 +/- 0.288 | 0.281 +/- 0.162 | <u>1.321 +/- 0.697</u> | 6.20 | 7.00 |
| GTG | 1.003 +/- 0.003 | 0.117 +/- 0.047 | 0.535 +/- 0.239 | -0.139 +/- 0.215 | 0.950 +/- 0.013 | 0.885 +/- 0.040 | 0.839 +/- 0.036 | 0.097 +/- 0.274 | 0.036 +/- 0.040 | 0.260 +/- 0.169 | 11.20 | 12.00 |
| RGD | **1.004 +/- 0.001** | 0.057 +/- 0.060 | 0.536 +/- 0.239 | -0.169 +/- 0.202 | 0.950 +/- 0.014 | 0.882 +/- 0.043 | 0.820 +/- 0.071 | 0.000 +/- 0.000 | 0.070 +/- 0.040 | 0.213 +/- 0.221 | 11.70 | 13.00 |
| BONET | 0.970 +/- 0.010 | 0.591 +/- 0.626 | 1.004 +/- 0.629 | 0.839 +/- 0.235 | 0.841 +/- 0.300 | 0.995 +/- 0.006 | 0.971 +/- 0.050 | 0.382 +/- 0.400 | 0.216 +/- 0.142 | 1.290 +/- 0.759 | 8.90 | 9.50 |
| DEMO | 0.985 +/- 0.007 | <u>1.189 +/- 0.861</u> | 0.774 +/- 0.332 | 0.947 +/- 0.082 | 0.992 +/- 0.013 | 0.993 +/- 0.006 | 0.998 +/- 0.003 | 0.915 +/- 0.258 | 0.919 +/- 0.464 | 0.739 +/- 0.373 | 6.70 | 6.50 |
| ROOT | 0.979 +/- 0.008 | 0.483 +/- 0.507 | 0.843 +/- 0.457 | 0.804 +/- 0.109 | 0.948 +/- 0.071 | 0.994 +/- 0.009 | 0.924 +/- 0.101 | 0.670 +/- 0.407 | 0.219 +/- 0.191 | 0.642 +/- 0.335 | 9.70 | 10.50 |
| SPADE | 0.977 +/- 0.009 | 0.890 +/- 0.493 | 1.710 +/- 2.302 | 0.916 +/- 0.095 | 0.998 +/- 0.005 | 0.998 +/- 0.003 | 0.967 +/- 0.064 | 0.856 +/- 0.480 | 0.716 +/- 0.287 | 0.937 +/- 0.366 | 6.90 | 6.00 |

Bold is best and underlining is second best within each task, based on the mean score.

## Experiment Contract

- Seeds: `38, 39, 40, 41, 42, 43, 44, 45`. The synthetic logged-dataset seed and optimizer seed both equal the trial seed.
- Data-mixture logged observations are fixed upstream data; only the optimizer seed varies by trial.
- Score: `(generated best utility - full logged minimum utility) / (full logged maximum utility - full logged minimum utility)`. Scores above 1 are allowed.
- Data-mixture training visibility: utility percentiles [0, 40].
- `D(best)` is the best optimizer-visible logged utility. LLM-DM logged runs retain their recorded model scale and training step, while method recommendations are evaluated at the target 1B/19,500-step fidelity.
- Synthetic functions: `ackley, schaffer2, sum_different_powers, matyas, power_sum, rosenbrock, michalewicz, hartmann6, shekel`.
- Displayed uncertainty: sample SD.
- `task_summary.csv` also records sample SD, standard error, 95% Student-t CI, and observed seed range.
- Candidate count: `K=8` for every method and trial.

## Selection And Method Provenance

Performance-selected union of prior single-seed COM and BDI category winners; descriptive comparison only.

Additional methods are continuous PyTorch adaptations. Published-result parity is not established. Exact source revisions, settings and substitutions are in METHOD_PROVENANCE.md and run_metadata.json.

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
