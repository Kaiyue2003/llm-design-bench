# Offline BBO Benchmark Protocol

This document defines the stable contract for methods evaluated by
`llm-design-bench`. Implementations may use different models and search
procedures, but they must obey the same data-access, candidate, and reporting
rules.

The agreed LLM-DM experiment is specified in [LLMDM_PROTOCOL.md](LLMDM_PROTOCOL.md),
including the frozen-data/config workflow and durable attempt artifacts. The
historical publication-v1 settings and results remain separate.

## Objective convention

Every task exposes a maximization utility. If the original objective is a loss,
the task adapter converts it once with

```text
utility = -objective
```

Methods receive utility and must maximize it. A method must not negate or
otherwise reinterpret the task direction. Reports may convert utility back to
the original objective for presentation.

## Offline data boundary

A method receives only an `OfflineProblem` constructed from the visible logged
dataset. It contains:

- logged designs;
- optional logged context such as model scale and training steps;
- logged maximization utilities;
- the target context at which candidates must be proposed; and
- a design-space object used to sample and validate candidates.

The hidden logged region, simulator, oracle utilities, and evaluation reference
are not part of `OfflineProblem`. A method must not call `Task.predict`, inspect
the hidden split, or use final evaluation results during training or candidate
search.

Only the evaluator may call the oracle, after a method has returned its final
candidate batch.

`OfflineBBOMethod.run` gives each invocation an isolated input copy via
`OfflineProblem.to(..., copy=True)`: tensors and the design space are cloned,
and `ProblemMetadata` is deep-copied, including nested dictionaries and lists
in `extra`. Method-local edits therefore do not change the evaluator's metadata
or the inputs to subsequent methods/seeds, even if the method raises an error.
Metadata should contain descriptive values compatible with Python `deepcopy`;
copy failures are not silently replaced by shared references.
Direct calls with `copy=False` retain shared metadata and design-space objects;
they are not an isolation boundary. This is protection against accidental
in-place edits, not a security sandbox for untrusted method code.

## Data-mixture settings

The primary data-mixture experiment uses all available logged model scales,
selecting the 0th through 40th utility percentiles **within each scale** and then
merging in source row order. Linear interpolation and inclusive boundaries keep
all cutoff ties. It evaluates mixture recommendations at the target fidelity:

```text
model scale:    1B (1000 million parameters)
training steps: 19,500
```

The fixed-1B experiment takes only the 1B rows of the already-frozen main visible
set, without a new split. It uses the same utility convention and target
fidelity. It is an ablation rather than the primary setting.

For methods originally defined without context variables, adding model scale
and training steps is a multi-fidelity adaptation and must be identified as
such in method metadata.

## Candidate budget

The paper-aligned final protocol uses `K = 128` candidates per method run.
Development smoke tests may use a smaller budget, but their results are not
comparable with final benchmark tables.

A method must return a tensor with exactly `K` rows and the task's design
dimension. The framework does not copy candidates to fill a short batch and
does not silently remove duplicates. Duplicate and unique candidate counts are
reported as method behavior.

`D(best)` is an observed-data reference, not a method. The separate Best Logged
method returns exactly 128 candidates, repeating visible designs when necessary;
its recommendations are evaluated at target fidelity like other methods.

## Random seeds

Randomness is separated into three namespaces:

- `dataset_seed` controls construction of a generated logged dataset;
- `split_seed` controls a stochastic offline split, when applicable; and
- `method_seed` controls model initialization, data-loader order, sampling,
  mutation, dropout, diffusion noise, and other method randomness.

The data-recipes dataset and percentile split are deterministic. Final method
comparisons use the paired method seeds

```text
38, 39, 40, 41, 42, 43, 44, 45
```

Each seed is a complete independent method run that returns `K` candidates.
The per-seed statistic is computed first. Across the eight runs, reports include
the mean, sample standard deviation, and standard error

```text
standard error = sample standard deviation / sqrt(8)
```

Seed 0 is a full-configuration pilot, excluded from formal statistics/ranking.
Deterministic methods are marked as such; eight identical executions must not
be presented as eight independent stochastic trainings.

## Normalization and comparison

The primary statistic is each seed's maximum reference-normalized utility;
raw target-fidelity utility/loss and median scores are retained. The score is

```text
(utility - reference minimum) / (reference maximum - reference minimum)
```

and is not clipped. Scores above one are allowed.

Comparisons between multi-scale and fixed-1B settings use the same full
multiscale logged utility reference. Setting-local normalized scores may be
reported for diagnostics but must not be compared across settings as if their
denominators were identical. The hidden reference is evaluation-only and is
never exposed to a method.

## Hyperparameters

Method hyperparameters must come from one of the following sources:

1. a cited official configuration;
2. a configuration fixed before final evaluation; or
3. tuning performed only on visible logged data.

Hyperparameters must not be selected using hidden-region or final oracle
utilities. Each final result records the complete resolved configuration.

## PyTorch boundary

The method-facing data contract uses PyTorch tensors, devices, dtypes, and
random generators. Neural models, differentiable kernels, gradient search, and
stochastic sampling should use PyTorch. Mature PyTorch-ecosystem dependencies,
such as BoTorch and GPyTorch, are allowed when they improve fidelity to the
published method.

An implementation that replaces a published model or learning mechanism must
be labeled an adaptation rather than an exact reproduction.

## Method provenance

Every registered method declares an implementation kind:

- `official_wrapper` for a thin wrapper around official code;
- `faithful_pytorch_port` for a PyTorch port intended to preserve the published
  algorithm;
- `multi_fidelity_adaptation` for a method extended with context/fidelity
  variables;
- `lightweight_adaptation` for a method that replaces a material component;
- `native_baseline` for a benchmark-native reference implementation.

Orthogonal changes such as adding multi-fidelity context are recorded as
adaptation tags. This allows, for example, an official wrapper to also declare
that its benchmark integration is a `multi_fidelity` adaptation.

Reports record the method name, implementation kind, source URL, source commit,
resolved configuration, package commit, task metadata, seeds, device, and
runtime.

## Failure policy

Invalid shapes, non-finite candidates, and out-of-space candidates are hard
failures. Timeouts, out-of-memory failures, and numerical failures are retained
in per-seed results rather than silently discarded. Summary tables report both
successful-run statistics and the number of failed seeds.

## Seed-run artifacts

The unified method runner writes two machine-readable files:

- `method_seed_results.csv` contains one row per method configuration and seed,
  including failures, timing, configuration, provenance, candidate diagnostics,
  raw utility, and reference-normalized scores;
- `method_seed_summary.csv` aggregates successful runs with a mean, sample
  standard deviation, standard error, and contributing-run count for each
  metric, while also reporting requested, successful, and failed runs.

The runner creates a fresh method instance for every seed. It evaluates no
candidate when method training or proposal fails.

Multi-task experiments use `BenchmarkTaskSpec` and `BenchmarkTrial` to feed the
same seed-level evaluator boundary into this runner, then write one unified
report. The complete versioned column contract, report artifacts, and legacy-v1
conversion rules are documented in [RESULT_SCHEMA.md](RESULT_SCHEMA.md).

The current formal CLI is `llm-design-bench`, with
`llm-design-bench-llmdm` as an equivalent explicit name. It enforces the frozen
LLM-DM data/plan workflow. `llm-design-bench-suite` remains a general-purpose
method/task development runner, not a replacement for that formal workflow.

The historical online/offline/publication/synthetic execution runners and
task-taking optimizer classes have been retired from the current package.
Synthetic tasks themselves remain available through the current method API
and suite runner. `llm-design-bench-report` retains read-only loading/conversion
of old results into new output directories; conversion does not make them
comparable with current formal results. Git history and unchanged result
archives retain historical provenance; see [REPRODUCING.md](REPRODUCING.md).

Refactoring the current source changes its fingerprint, not the source pinned
by an existing frozen plan. Existing releases must still use their own pinned
code/environment for resume or reproduction. Do not alter their plan hashes,
notebook pins or archived results to accommodate a new checkout. A new
experiment requires its own freeze and pilot; no retraining of an existing
archive is implied by this source cleanup.
