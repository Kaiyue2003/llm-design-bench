# GPyTorch v2: formal multi-scale data-mixture results

This snapshot contains 152 successful final results: 19 methods, each with
formal seeds 38-45. It covers only `multi_scale` data-mixture optimization
at target 1B / 19500 training steps. Pilot seed 0 and the fixed-1B ablation
are not included. These are simulator evaluations, not new LLM pretraining runs.

## Files and provenance

- [method_seed_results.csv](method_seed_results.csv): 152 per-seed records,
  including method configuration, runtime environment, diagnostics, artifact
  references and infrastructure-retry reasons.
- [method_seed_summary.csv](method_seed_summary.csv): 19 per-method summaries,
  including mean, sample standard deviation, standard error and coverage.
- [run_metadata.json](run_metadata.json): frozen experiment identity, CSV byte
  counts and SHA256 hashes, runtime policy, retry notes and audit scope.

Both CSVs are byte-for-byte copies of the supplied Colab exports. Their values
and embedded metadata were not rewritten. The local `.gitattributes` disables
Git line-ending conversion for these two files so their recorded hashes remain
portable across checkouts.

The run used code commit `9d70e1458239142353e34cc596ba2b8b8764871d`, not
whatever code happens to be on the current branch. Its plan ID is
`1e097b0f5f554982ac5029a28b66337dfa22cfd9a309da200523202eea786b8c`.
See the immutable [release](../../experiments/llmdm_forward_gpytorch_v2/release.json),
[method plan](../../experiments/llmdm_forward_gpytorch_v2/plan.json),
[data manifest](../../experiments/llmdm_forward_gpytorch_v2/data/manifest.json)
and [Colab guide](../../docs/COLAB_GPYTORCH.md). The plan contains the full
per-method training and search budgets; this archive does not refreeze them.

Do not merge these rows with [the historical publication table](../publication/README.md)
or v1 results: they are separate experimental snapshots.

## Shared experiment settings

- Objective: `eval/RedPajamaStackExchange/CrossEntropyLoss`; maximize
  `utility = -loss`.
- Data: 454 usable logged observations; 184 visible observations, selected
  within each model scale using inclusive utility percentiles 0-40, linear
  quantiles and all cutoff ties. Model scale and training steps are conditions.
  The visible dataset is identical for all methods and seeds.
- Mixture order: `RedPajamaWikipedia`, `RedPajamaStackExchange`,
  `RedPajamaGithub`, `RedPajamaArXiv`, `RedPajamaBook`.
- Target context: `[1000.0, 19500.0]` (1B model, 19500 steps).
- Each seed returns K=128 candidates. Duplicates are allowed; distinct mixtures
  are counted after rounding to 12 decimal places.
- Formal seeds: 38-45. Fixed-1B, when run separately, must use the exact
  26-row 1B subset of the main visible data without resplitting. No fixed-1B
  result is included here.

The recorded Colab runtime used Python 3.13.15, PyTorch 2.11.0+cu128,
CUDA 12.8 and an NVIDIA A100-SXM4-80GB for neural methods, with one Torch
CPU thread. Best Logged, Random Search, Sobol, BDI and both GP methods ran
on CPU; the oracle also ran on CPU. BDI and both GP methods used float64;
the remaining methods used float32, without mixed precision. GPyTorch 1.15.2
and linear_operator 0.6.1 back `ga_on_gp` and `bo_qei`. BDI remains the
independent RBF kernel-ridge adaptation, not an original JAX/Neural Tangents
reproduction. Recorded `deterministic_algorithms` is false; matching seeds
does not promise bitwise equality across hardware or library versions.

## Results

For each seed, select the best oracle-evaluated candidate among that seed's
128 candidates; then report the mean and sample SD (`ddof=1`) over eight
seeds. The table is **mean +/- SD, not SE**. The CSV also retains SE.
`Best refnorm score` is `refnorm_max_score`; `Best loss` is `raw_min_loss`.
Rank is descending mean best refnorm score, not a statistical-significance test.

Normalization uses `(utility - reference_min_utility) /
(reference_max_utility - reference_min_utility)`, with fixed extrema
`-3.630530834197998` and `-0.8202966451644897`. These come from the full
logged reference at its recorded fidelities, not from reevaluating all 454
logged mixtures at target fidelity. A score above 1 exceeds that historical
reference maximum; it is not a percentage improvement or evidence from
actual LLM training.

| Rank | Method | Best refnorm score (mean +/- SD) | Best loss (mean +/- SD) | Method seconds (mean) |
| ---: | --- | ---: | ---: | ---: |
| 1 | BO-qEI adaptation | 1.004580 +/- 0.000187 | 0.807426 +/- 0.000525 | 4.65 |
| 2 | Offline MLP | 0.999012 +/- 0.005216 | 0.823073 +/- 0.014658 | 2.92 |
| 3 | Standard GA adaptation | 0.996830 +/- 0.005234 | 0.829205 +/- 0.014708 | 3.29 |
| 4 | BDI adaptation | 0.995904 +/- 0.000000 | 0.831806 +/- 0.000000 | 1.76 |
| 5 | SPADE adaptation (official-core-derived) | 0.995369 +/- 0.001470 | 0.833311 +/- 0.004130 | 25.83 |
| 6 | Random Search | 0.994822 +/- 0.003433 | 0.834849 +/- 0.009647 | 0.001 |
| 7 | COMs adaptation | 0.992530 +/- 0.003058 | 0.841289 +/- 0.008593 | 16.36 |
| 8 | Sobol | 0.990850 +/- 0.002366 | 0.846010 +/- 0.006650 | 0.002 |
| 9 | MC-Dropout adaptation | 0.988403 +/- 0.001252 | 0.852886 +/- 0.003518 | 7.08 |
| 10 | CMA-ES adaptation | 0.988128 +/- 0.003345 | 0.853659 +/- 0.009399 | 7.61 |
| 11 | LTR adaptation | 0.987674 +/- 0.000110 | 0.854935 +/- 0.000310 | 5.67 |
| 12 | RoMA adaptation | 0.986745 +/- 0.001853 | 0.857547 +/- 0.005207 | 1084.86 |
| 13 | ICT adaptation | 0.985829 +/- 0.000200 | 0.860119 +/- 0.000562 | 9.24 |
| 14 | Tri-Mentoring adaptation | 0.985734 +/- 0.000321 | 0.860388 +/- 0.000903 | 211.27 |
| 15 | GA on GP adaptation | 0.984859 +/- 0.006292 | 0.862845 +/- 0.017681 | 2.88 |
| 16 | REINFORCE adaptation | 0.984832 +/- 0.003985 | 0.862921 +/- 0.011199 | 7.74 |
| 17 | MATCH-OPT adaptation | 0.984497 +/- 0.003194 | 0.863863 +/- 0.008977 | 11.87 |
| 18 | Best Logged | 0.983934 +/- 0.000000 | 0.865445 +/- 0.000000 | 0.004 |
| 19 | PGS adaptation | 0.979567 +/- 0.009301 | 0.877719 +/- 0.026138 | 159.89 |

Method time is the recorded method execution time, excluding oracle evaluation,
backup overhead and interrupted work. It is not end-to-end Colab elapsed time.
CPU and CUDA methods follow the device policy above; this is not a uniform
GPU throughput comparison.

### Interpretation and recovery notes

- Adaptation labels are retained. `standard_ga` means standard gradient ascent,
  not a genetic algorithm.
- Best Logged and BDI have the same recorded candidate-file hash across all
  eight seeds, with effectively zero score variation. These eight completed
  seed records should not be described as eight independently varying samples.
- PGS returns 4-126 distinct mixtures per seed (mean 53.625). The protocol
  permits duplicates, so these runs remain included without retuning or
  replacing seeds. The table's best-candidate metric does not describe the
  quality or diversity of the whole candidate batch.
- RoMA seeds 41 and 45 were explicitly retried after reported connection/runtime
  interruptions, keeping the original seed and frozen configuration. The exact
  reasons and final artifact-relative directories are in `run_metadata.json`
  and the per-seed CSV. Eight final successes do not imply that no earlier
  attempt was interrupted.

## Validation and retained artifacts

Local CSV checks cover byte hashes, unique 19 x 8 coverage, frozen configuration
and provenance, utility direction, target fidelity, device/dtype, all 18 numeric
metrics' mean/SD/SE/count summaries, ranks, table values and recorded retries.
Run the archive-only regression tests from the repository root:

```console
python -m pytest tests/test_gpytorch_reference_results.py -q
```

This is **CSV-level consistency verification only**. The full Colab/Drive
snapshot, candidate/evaluation NPZ files, result manifests, logs and attempt
journals are not included in this directory and were not independently
revalidated here. In particular, hashes declared inside a CSV do not substitute
for checking the corresponding artifact files. Full raw archives remain in the
operator's plan-specific Drive backup and should be retained for a subsequent
artifact audit. `artifact_dir` in the CSV is an original Colab path;
`artifact_relative_dir` is relative to the restored formal results directory.
See [the result schema](../../docs/RESULT_SCHEMA.md) for artifact layout.
