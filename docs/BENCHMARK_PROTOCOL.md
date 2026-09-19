# Offline BBO Benchmark Protocol

This document defines the stable contract for methods evaluated by
`llm-design-bench`. Implementations may use different models and search
procedures, but they must obey the same data-access, candidate, and reporting
rules.

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

## Data-mixture settings

The primary data-mixture experiment uses all available logged model scales.
Within each scale it exposes the 0th through 40th utility percentiles, including
threshold ties, then merges the visible rows into one shared frozen manifest.
It evaluates mixture recommendations at the target fidelity:

```text
model scale:    1B (1000 million parameters)
training steps: 19,500
```

The fixed-1B experiment takes the 1B subset of that exact visible manifest,
without rerunning the split. It keeps the same utility convention, target fidelity
and full-logged normalization reference. It is an ablation, not the main setting.

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

`D(best)` is a scalar evaluator-side logged reference. The separately registered
`best_logged` method returns K candidates and obeys the same output contract;
it may repeat visible designs to fill the requested budget. Do not confuse its
candidate statistics with the scalar `D(best)` reference.

## Random seeds

Randomness is separated into three namespaces:

- `dataset_seed` controls construction of a generated logged dataset;
- `split_seed` controls a visible-data training/validation partition where a
  method's preparation pipeline uses one; and
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

The formal runner records all eight seeds for each selected method, including
deterministic references, to maintain uniform coverage. Identical deterministic
results must not be described as eight independent stochastic discoveries.
Pilot seed 0 uses the full configured budget and is checked before formal runs.

## Normalization and comparison

Raw target-fidelity utility and the corresponding original objective are the
primary cross-setting metrics. A reference-normalized score is

```text
(utility - reference minimum) / (reference maximum - reference minimum)
```

and is not clipped. Scores above one are allowed.

Comparisons between multi-scale and fixed-1B settings use the same frozen
full-logged utility reference while evaluating all candidates at the target
fidelity. Setting-local normalized scores may be
reported for diagnostics but must not be compared across settings as if their
denominators were identical. The hidden reference is evaluation-only and is
never exposed to a method.

## Hyperparameters

Method hyperparameters must come from one of the following sources:

1. a cited official configuration;
2. a configuration fixed before final evaluation; or
3. tuning performed only on visible logged data.

Hyperparameters must not be selected using hidden-region, pilot-oracle, or final
oracle utilities. Each final result records the complete resolved configuration.
For the integrated campaign, owners agree method-specific budgets after the code
merge, then freeze a new plan and run full-budget pilots. Those experimental
decisions are not code-merge prerequisites.

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

The frozen workflow saves per-attempt manifests/results, candidate and evaluation
NPZ files, provenance/checksums and environment metadata. Candidates are committed
before oracle evaluation. Verified matching attempts support explicit resume;
failed or interrupted attempts are never silently discarded or retried.

The method/report APIs also write machine-readable and rendered summaries:

- `method_seed_results.csv` contains one row per method configuration and seed,
  including failures, timing, configuration, provenance, candidate diagnostics,
  raw utility, and reference-normalized scores;
- `method_seed_summary.csv` aggregates successful runs with a mean, sample
  standard deviation, standard error, 95% Student-t confidence interval,
  observed minimum and maximum, range, and contributing-run count for each
  metric, while also reporting requested, successful, and failed runs;
- `method_seed_table.md` is a GitHub-readable mean +/- SE table; and
- `method_seed_table.tex` is the equivalent Overleaf-ready table.

The runner creates a fresh method instance for every seed. It evaluates no
candidate when method training or proposal fails.
Incomplete formal seed coverage is retained in reports but is not rank-eligible.

## Supported entry points and archives

`llm-design-bench` is the default `prepare`/`freeze`/`run` application;
`llm-design-bench-llmdm` is its explicit alias. The report command
`llm-design-bench-report from-unified` only creates reports. Docker invokes the
same formal application.
Generic task/method Python APIs remain available for research, not as an alternate
formal protocol. SPADE remains outside the formal roster until resolved.

Retired experiment CLIs/runners are not installed. Historical commands and results
under `reference_results/` require their original recorded commit/environment and
must not be combined into a new 27-method formal ranking.
