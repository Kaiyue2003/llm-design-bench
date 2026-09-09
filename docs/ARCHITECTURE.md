# Unified PyTorch Architecture

The benchmark separates method-visible data from evaluator-only state. This is
the central reproducibility and leakage-prevention rule.

```text
Task adapter
  |-- visible logged rows --> OfflineProblem
  |                            |
  |                            +--> OfflineTensorDataset
  |                            +--> deterministic OfflineDataSplit
  |                            +--> train-only fitted transforms
  |                            +--> registered OfflineBBOMethod
  |                                      |
  |                                      +--> MethodResult(candidates)
  |
  +-- hidden oracle/reference ----------------> evaluator and reports
```

## Tensor contract

An `OfflineProblem` contains only floating-point PyTorch tensors:

| Tensor | Shape | Meaning |
| --- | ---: | --- |
| `train_designs` | `[N, D]` | optimizer-visible physical designs |
| `train_context` | `[N, C]` | fidelity or task context for each logged row |
| `train_utility` | `[N]` | maximization utility |
| `target_context` | `[C]` | context at which final candidates are evaluated |

The data-mixture task uses simplex designs and the context
`[model_scale_millions, training_steps]`. Synthetic functions use
box-bounded designs and constant context. The same method API handles both.

`OfflineTensorDataset` preserves original row indices.
`split_offline_dataset` uses a dedicated CPU
`torch.Generator`, making the train/validation assignment independent
of model randomness and auditable across devices.

## Fitted transforms

`prepare_offline_problem` fits context and utility standardizers on
training rows only. Validation values never influence the fitted mean or scale.

- Positive model scale is transformed with `log1p` before standardization.
- Utility is standardized for model training but remains a maximization target.
- Simplex designs use additive log-ratio coordinates with the final component
  as reference.
- Box designs use unit-box coordinates.
- Candidate outputs are decoded back to physical coordinates and validated
  before evaluation.

The legacy `to_unconstrained` and `from_unconstrained` maps
remain available for gradient search. Model coordinates and optimization
parameters are deliberately separate concepts.

## Method boundary

Every runnable method subclasses `OfflineBBOMethod` and returns a
`MethodResult`. A result contains unevaluated candidates, training
summaries, and diagnostics. Oracle utilities cannot appear in this object.

`PreparedFitThenProposeMethod` supplies a standard lifecycle:

1. deterministically split the visible dataset;
2. fit transformations on training rows;
3. call `fit_prepared`;
4. call `propose_prepared`;
5. validate the exact candidate count, dtype, device, and search-space bounds.

The runnable registry contains implementations that can execute. The separate
integration catalog describes planned methods. A planned method is promoted to
the registry only after its PyTorch implementation and parity tests pass.

## Current implementations

| ID | Kind | PyTorch status |
| --- | --- | --- |
| `best_logged` | reference | native |
| `offline_mlp` | forward surrogate | native |
| `coms` | forward surrogate | lightweight PyTorch adaptation of TensorFlow code |
| `bdi` | bidirectional surrogate | lightweight PyTorch/RBF adaptation of JAX code |

CbAS, MINs, DDOM, GABO, GTG, RGD, BONET, DEMO, ROOT, and SPADE are cataloged
for later integration. They are intentionally not runnable placeholders.

## Statistical outputs

The seed runner writes raw results before aggregation. For every metric it
reports the number of successful runs, mean, sample SD, standard error,
two-sided 95% Student-t confidence interval, observed minimum, observed
maximum, and range. Failed seeds remain in the raw table.

The publication runner emits both:

- a compatibility table using mean +/- sample SD; and
- a Table 1-style table using mean +/- standard error.

Both tables are derived from the same raw per-seed rows.
