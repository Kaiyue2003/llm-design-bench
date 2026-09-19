# Main integration: shared contract, methods, and formal workflow

## Scope and provenance

Integration branch: `integration/unified-benchmark`.

- Base: main `09991781f82b5f2f5ae1d249ac1e211c4e00c225`.
- Shared runtime and forward methods:
  `feat/pytorch-methods-results-audit` at
  `05b35452c6325080705a16dd21009a8f9a536421`.
- The independent integration worktree preserves the original checkout and
  research work, both Git histories, archived results and old frozen plans.
- The formal roster combines **27 non-SPADE methods**: 18 from the forward branch
  and nine from main. The four overlapping baseline IDs select the forward
  implementations. See [Integrated Methods](INTEGRATED_METHODS.md).

This integration changes code and infrastructure, not approved experiment budgets
or historical result values. Regression/smoke tests use invented data and small
budgets; they are not publication experiments.

## One default workflow

`llm-design-bench` now invokes the frozen LLM-DM CLI.
`llm-design-bench-llmdm` is an alias for the same application; its module entry is
`python -m llm_design_bench.llmdm_cli`. Reporting retains
`llm-design-bench-report from-unified`.

The old experiment command-line entry points, wrappers, task-based optimizer
classes, compatibility factories and old runners are removed from the current
runtime. Docker's `benchmark` service invokes the same new CLI, with a separate
invented-data `smoke` check. There is no installed parallel publication or
synthetic experiment workflow. Generic synthetic tasks, spaces, Prepared APIs and
research methods remain available through Python.

See [Reproducing Results](REPRODUCING.md) for the only formal command sequence and
[Docker](DOCKER.md) for container mounts and commands.

## Shared method boundary

```text
visible data + target context -> OfflineProblem
method seed + candidate budget + device/dtype -> RunContext
method.run(problem, context) -> MethodResult (unevaluated candidates)
evaluator only -> oracle scores -> immutable attempt + aggregate reports
```

Methods receive visible designs, maximization utilities, fidelity context, target
context and a design space, not an oracle or hidden/reference data. Runs receive
independent tensors and deep-copied metadata. They return exactly K feasible
candidates with the requested shape, dtype and device. Duplicates are allowed and
counted.

Both templates remain supported:

- `FitThenProposeMethod`: direct fitting/search with method-specific visible-only
  preprocessing.
- `PreparedFitThenProposeMethod`: a visible-data train/validation partition and
  transforms fitted on its training partition. The validation data is not the
  benchmark's hidden high-utility region; this preprocessing is not imposed on
  the forward methods.

Main's simplex/box encodings and extended provenance metadata remain.
`configuration()` exports reconstructible constructor arguments, including
inherited defaults, but not learned state. Freeze validates reconstruction.

## Frozen LLM-DM protocol

| Item | Rule |
| --- | --- |
| Objective | StackExchange cross-entropy, `utility = -loss` |
| Visible data | Per-model-scale utility percentiles 0–40, including threshold ties |
| Design order | Fixed five-source adapter `DOMAIN_ORDER` |
| Conditioning | Model scale and training steps |
| Target | 1B (1000M), 19500 steps |
| fixed-1B | 1B subset of the same visible rows; no resplitting |
| Candidates | K=128; duplicates allowed and counted |
| Pilot | Full configured budget, seed 0; artifacts/cost/stability checks |
| Formal | Seeds 38–45 with verified matching pilot coverage |
| Precision | BDI/GA on GP/BO-qEI float64; others float32; no mixed precision |
| Reference | Frozen full-logged utility range, evaluator-only |
| Statistics | Per-seed scores and mean/SD/SE; incomplete formal coverage is not ranked |

Source/data/checkpoint and plan identities are verified. Candidates are saved
before oracle evaluation; failed/interrupted attempts remain visible. Resume
requires compatible verified artifacts. Locks and atomic writes protect output;
reports must not overwrite their input CSV. These are filesystem safeguards, not
a claim that multiple files form one database transaction.

The shared report API retains main's Markdown/LaTeX helpers, confidence/range
columns, provenance and the `resolved_method_config_json` alias. Canonical rows
store requested overrides in `requested_method_config_json` and complete resolved
settings in `method_config_json`. Accepting an older already-unified row schema
does not restore an old training workflow or authorize mixed-plan rankings.

## Deliberate implementation decisions

- GPyTorch is the unified GP backend. GA on GP and BO-qEI use their learned exact
  RBF model. GABO reuses the model/settings with fixed kernel/noise, latent inputs,
  target standardization, a variance floor and analytic EI. Equation/gradient
  checks do not promise bitwise-identical full runs across platforms.
- BDI remains a finite RBF kernel-ridge/distillation adaptation, not GP inference.
- Neither competing SPADE implementation is selected for formal use. The raw
  main-branch Python implementation remains available, but formal freeze/load/run
  reject it, including run-ID aliases. SPADE selection is not a merge blocker.
- No historical artifact is rewritten or relabeled. Reproducing an archive's old
  commands requires its original recorded commit and environment.

## Merge now; experimental planning afterward

The integration is reviewed as a code change: contracts, non-SPADE methods,
default CLI, reporting, packaging, documentation and small runtime/container
checks. The PR should report which checks actually ran and any unavailable
platform checks. **It need not wait for new budgets, a frozen experimental plan,
real-data pilots, full Docker numerical reproduction, or SPADE selection.**

After merge:

1. Each method owner approves its method-specific training/search budget.
2. Prepare or verify the shared data manifest and freeze a **new** plan against
   the merged release and approved configurations.
3. Run full-budget seed-0 pilots and inspect cost/stability/artifacts.
4. Run formal seeds 38–45, then the intended fixed-1B ablation and reports.

Source/constructor identities have changed. Old plans and pilots cannot silently
be attached to the new source; historical results remain valid records of their
original configurations, not results for the integrated 27-method campaign.

## Verification scope

Earlier integration checkpoints exercised the shared contract, all 27 non-SPADE
constructor round trips, component-level GP comparisons, additional and forward
method tests, failure/resume behavior, report safety, and wheel entry loading.
These are development checks, not a claim of upstream-paper parity or a completed
new formal experiment. The final PR records the current test counts and installed
wheel/container checks after removing the retired entry points.
