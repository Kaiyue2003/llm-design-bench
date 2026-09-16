# Unified Result Schema

New experiments use schema version 1 from
`llm_design_bench.evaluation.unified_report`. The historical
`publication_runner` remains available only to reproduce publication v1; it is
not the execution path for newly registered methods.

## Execution layers

```text
BenchmarkTaskSpec.trial_factory(seed)
    -> BenchmarkTrial(evaluator task, OfflineProblem, reference utility)
    -> run_method_seed_benchmark(...)
         method receives OfflineProblem only
         evaluator receives returned candidates
    -> one unified per-seed row
    -> write_unified_report(...)
```

The factory is called once for every task/seed pair. All methods evaluated for
that pair share the same offline problem and reference. This supports both:

- fixed logged data with eight optimizer seeds, as in data-recipes; and
- paired synthetic trials where dataset seed and method seed are identical.

The evaluator task, oracle, hidden reference utilities, and `D(best)` summary
never enter `OfflineBBOMethod.run`.

## Per-seed artifact

`method_seed_results.csv` contains exactly one row per
`(experiment_id, suite, task_id, run_id, method_seed)`.

The main field groups are:

| Group | Representative fields |
| --- | --- |
| Schema | `schema_version`, `result_source`, `experiment_id` |
| Task | `suite`, `task_id`, `task_display_name`, category, objective and target context |
| Method | `run_id`, `method_id`, display name, family, implementation kind, adaptations |
| Provenance | source URL/commit, package commit, requested and fully resolved method config |
| Seeds | method, dataset, and split seeds |
| Data/reference | train size, normalization reference ID/range, `D(best)` utility and score |
| Status | success/failure, error type/message |
| Cost | method, oracle-evaluation, and total wall-clock seconds |
| Utility | raw and reference-normalized maximum, median, and mean |
| Candidate diagnostics | unique count/fraction, diversity, novelty, and simplex diagnostics |
| Method diagnostics | JSON training summary and method-local diagnostics |

Failed rows are retained. Their unavailable scores and diagnostics are null in
CSV rather than filled with a favorable or unfavorable value.

## Summary artifact

`method_seed_summary.csv` contains one row per task and method run ID. Every
numeric metric has four suffixes:

```text
<metric>_n       successful finite observations
<metric>_mean    mean over successful observations
<metric>_std     sample standard deviation, ddof=1
<metric>_se      sample standard deviation / sqrt(n)
```

It also reports requested, successful, and failed run counts. The default
Markdown and LaTeX tables display mean +/- SE to match the SPADE paper, while
the CSV retains both SD and SE.

`rank_summary.csv` ranks method mean normalized-maximum scores within each
task, then reports mean and median task rank. A method with no successful run
on a task is absent from that task's ranking; its failure counts remain visible.

For frozen LLM-DM runs, `phase` and `required_seeds_json` impose a stricter rule:
pilots are not ranked, and a formal method must have successful seeds 38-45
before receiving a rank. The summary also retains `missing_runs` and marks
incomplete coverage in both Markdown and LaTeX.

## Durable frozen-protocol attempts

The dedicated [LLM-DM workflow](LLMDM_PROTOCOL.md) enables evaluator-owned
per-attempt persistence. This adds `phase`, `required_seeds_json`,
`provenance_json`, `logical_fingerprint`, `artifact_relative_dir`,
`environment_json`, and raw loss statistics to per-seed rows. The additive
fields do not reconstruct missing artifacts for historical results.

Each attempt contains `manifest.json`, `candidates.npz` saved before any oracle
call, `evaluation.npz`, and `result.json`. The completion row records candidate
and evaluation file hashes. Reuse verifies hashes, shapes, dtype, target context,
utilities, refnorm values and aggregate consistency. Failed/interrupted attempts
remain available and never silently become a different seed or configuration.
The relative artifact path supports copying an entire results directory between
machines. Aggregate reports are updated under a single-writer suite lock;
independent shards may be appended sequentially under the same frozen plan.

### Read-only statistics verification

`verify_successful_attempt(attempt_directory)` also recomputes these required
successful-result fields from the saved arrays:

- For `utility_transform="negative_loss"`, `raw_min_loss`, `raw_median_loss`
  and `raw_mean_loss` must match reductions of `-utility`. The saved `raw_loss`
  array must already equal `-utility`. Other objective transforms do not require
  loss statistics.
- `unique_candidate_count` must exactly match the number of distinct rows after
  `np.round(candidates, decimals=12)`, preserving the saved candidate dtype.
- `unique_candidate_fraction` must match that count divided by the candidate
  budget and lie in `[0, 1]`.

Missing, nonnumeric, boolean or non-finite values are rejected with the field
name. Floating summaries use `rtol=1e-12, atol=1e-12`; counts use exact equality.
The validator never rewrites results, reruns a method or queries the oracle.
Matching CSV and JSON statistics alone are insufficient if both disagree with
the raw arrays. Verification of these fields does not independently recompute
all other diagnostics, such as novelty or method-local training diagnostics.

Auditing an old attempt with the current validator is separate from resuming
its experiment. This API reads the existing manifest, JSON and NPZ directly and
does not load a method plan or require the data-recipes runtime. It needs all
four attempt files; the two archived CSVs alone are insufficient. Missing
statistics are not backfilled automatically.

All package `.py` files, including this validator, contribute to the frozen
source fingerprint. Keep published plans and notebook pins unchanged when
updating validation code: new code must not resume training under an old source
fingerprint. Continuing the historical run requires its pinned code; running
new experiments with changed code requires a separately frozen plan and pilot.
Read-only verification of existing artifacts requires neither retraining nor
editing their recorded provenance.

## D(best)

`d_best_summary.csv` is computed once per task/seed after verifying that every
method row in the paired trial uses identical reference values. `D(best)` is
kept separate from method rankings.

For multi-fidelity LLM-DM, `D(best)` is the best optimizer-visible utility at
its recorded fidelity. A Best Logged method can differ because its selected
mixtures are evaluated at target fidelity.

## Human-readable artifacts

Each unified report directory contains:

- `README.md`: mean +/- SE table, run/failure totals, and provenance;
- `benchmark_table.tex`: Overleaf-ready mean +/- SE table;
- `run_metadata.json`: schema, seeds, tasks, methods, uncertainty convention,
  candidate budget, and caller-supplied experiment metadata.

## Legacy publication v1

`load_legacy_publication_results(...)` maps the frozen v1 `raw_runs.csv` into
schema version 1. The conversion is lossless for fields present in v1 and
leaves runtime, diversity, novelty, source commit, and other unavailable fields
empty. It never fabricates those values.

Conversion must target a new directory:

```bash
llm-design-bench-report from-legacy \
  --publication-dir reference_results/publication \
  --results-dir results/publication_v1_unified
```

To rebuild reports from an existing unified per-seed CSV:

```bash
llm-design-bench-report from-unified \
  --input-csv results/experiment/method_seed_results.csv \
  --results-dir results/experiment_report
```

The CLI rejects attempts to overwrite its source artifact.

## Running the built-in task specs

The suite CLI uses the same output contract for synthetic and data-mixture
tasks. For the primary LLM-DM setting:

```bash
llm-design-bench-suite \
  --data-mixture \
  --data-recipes-root ../data-recipes \
  --method best_logged --method random_search \
  --method sobol --method offline_mlp \
  --method standard_ga --method cma_es --method reinforce \
  --method bo_qei --method ga_on_gp --method mc_dropout \
  --method tri_mentoring --method ict --method roma \
  --method ltr \
  --method match_opt \
  --method pgs \
  --method coms --method bdi \
  --seed 38 --seed 39 --seed 40 --seed 41 \
  --seed 42 --seed 43 --seed 44 --seed 45 \
  --candidate-budget 128 \
  --results-dir results/unified_data_mixture
```

Use `--fixed-1b` for the fixed-scale ablation. The task-spec builder changes
the visible logged rows but deliberately retains the unfiltered data-recipes
reference range, so refnorm scores remain comparable across the two settings.
