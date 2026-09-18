# Unified PyTorch Architecture

The benchmark separates method-visible data from evaluator-only state. This is
the central reproducibility and leakage-prevention rule.

```text
task adapter -> frozen visible manifest -> OfflineProblem
                                              |
                           registered OfflineBBOMethod
                             /                       \
                   direct fit/search            Prepared pipeline
                                                   (visible split,
                                                   train-only transforms)
                             \                       /
                               MethodResult(candidates)
                                         |
                                  save candidates
                                         |
hidden oracle/reference ------------> evaluator -> saved results/reports
```

## Tensor contract

The data arrays in `OfflineProblem` are floating-point PyTorch tensors:

| Tensor | Shape | Meaning |
| --- | ---: | --- |
| `train_designs` | `[N, D]` | optimizer-visible physical designs |
| `train_context` | `[N, C]` | fidelity or task context for each logged row |
| `train_utility` | `[N]` | maximization utility |
| `target_context` | `[C]` | context at which final candidates are evaluated |

The data-mixture task uses simplex designs and the context
`[model_scale_millions, training_steps]`. Synthetic functions use
box-bounded designs and constant context. The same method API handles both.

For methods using preparation, `OfflineTensorDataset` preserves original row indices.
`split_offline_dataset` uses a dedicated CPU
`torch.Generator`, making the train/validation assignment independent
of model randomness and auditable across devices.

## Fitted transforms

`prepare_offline_problem` is used by the additional methods, not imposed on every
forward method. It fits context and utility standardizers on training rows only.
Validation values never influence the fitted mean or scale, and both partitions
come solely from the shared optimizer-visible rows.

- Positive model scale is transformed with `log1p` before standardization.
- Utility is standardized for model training but remains a maximization target.
- Simplex designs use additive log-ratio coordinates with the final component
  as reference.
- Box designs use unit-box coordinates.
- Candidate outputs are decoded back to physical coordinates and validated
  before evaluation.

The optimization `to_unconstrained` and `from_unconstrained` maps
remain available for gradient search. Model coordinates and optimization
parameters are deliberately separate concepts.

## Method boundary

Every runnable method subclasses `OfflineBBOMethod` and returns a
`MethodResult`. A result contains unevaluated candidates, training
summaries, and diagnostics. Oracle utilities cannot appear in this object.

`FitThenProposeMethod` supports direct visible-data fitting and proposal.
`PreparedFitThenProposeMethod` supplies an optional preparation lifecycle:

1. deterministically split the visible dataset;
2. fit transformations on training rows;
3. call `fit_prepared`;
4. call `propose_prepared`;
5. validate the exact candidate count, dtype, device, and search-space bounds.

The runnable registry contains executable implementations. The integration
catalog separately tracks `planned`, `implemented_adaptation`, and
`parity_validated` states. Equation checks and executable contract tests do not
establish numerical equivalence to an upstream implementation.

## Current implementations and runtime

The formal registry roster contains 27 non-SPADE methods: 18 reference/forward/
search methods from the forward branch and nine additional methods from main.
The four overlapping baseline IDs select the forward implementations. The raw
SPADE Python method remains available for research but is rejected by formal
freeze/load/run. [Integrated Methods](INTEGRATED_METHODS.md) records ownership;
[Method Notes](METHODS.md) and [Additional Methods](ADDITIONAL_METHODS.md) describe
the selected algorithms and adaptations.

The new methods isolate module initialization with `torch.random.fork_rng`
and use the run's generator for stochastic operations. Trajectory and bridge
pairs stay within observed fidelity groups. The evaluator saves candidates after
optimization returns and before oracle evaluation, then saves evaluation arrays.

The public execution route is `llm-design-bench` (alias
`llm-design-bench-llmdm`) → frozen protocol → unified suite → seed runner → method
boundary → evaluator → artifact-backed reports. Docker uses the same entry point.
The old experiment runners/optimizer compatibility layer are removed. Generic
synthetic tasks and Prepared APIs remain available, not as a parallel formal CLI.

Within `evaluation/`, data manifests and plans establish input identity;
`seed_runner.py` executes trials, with contract/persistence/statistics/rendering
helpers; `run_artifacts.py` protects per-attempt evidence; `unified_report.py`
assembles task/method coverage and reports. `llmdm_cli.py` is the thin formal CLI,
and `report_cli.py` provides read-input/write-report operations only.

## Statistical outputs

The seed runner writes raw results before aggregation. For every metric it
reports the number of successful runs, mean, sample SD, standard error,
two-sided 95% Student-t confidence interval, observed minimum, observed
maximum, and range. Failed seeds remain in the raw table.

Tables derive from the same per-seed rows and identify the uncertainty statistic
being shown; incomplete formal coverage is not ranked. Historical result formats
remain historical evidence, not inputs to a mixed-plan leaderboard. Approving
the next campaign's budgets and running full-budget pilots happens after the code
merge, not as an architectural prerequisite.
