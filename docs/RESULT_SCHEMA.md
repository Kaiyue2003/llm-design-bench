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
  --method coms --method bdi \
  --seed 38 --seed 39 --seed 40 --seed 41 \
  --seed 42 --seed 43 --seed 44 --seed 45 \
  --candidate-budget 128 \
  --results-dir results/unified_data_mixture
```

Use `--fixed-1b` for the fixed-scale ablation. The task-spec builder changes
the visible logged rows but deliberately retains the unfiltered data-recipes
reference range, so refnorm scores remain comparable across the two settings.
