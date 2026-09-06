# Method Notes

## Common task contract

Each task provides logged designs `logged_x`, logged maximization utilities
`logged_y`, candidate validation, evaluation through `predict`, and a target
fidelity. Data-mixture designs live on a five-dimensional simplex. Synthetic
designs live inside each function's box bounds.

For synthetic minimization functions, the adapter defines
`utility = -objective`. This lets every optimizer maximize the same quantity.

## Offline MLP

The MLP fits a two-hidden-layer ReLU network to standardized logged utilities.
Candidate optimization starts from the best unique logged designs and updates
unconstrained parameters with Adam. A softmax maps parameters to simplex
mixtures; a sigmoid and the task bounds map parameters to continuous synthetic
designs.

## Standard GA

The registered `standard_ga` method is the paper-facing gradient-ascent
baseline. It trains one probabilistic MLP, treats its mean prediction as the
deterministic search objective, starts from high-utility logged designs, and
uses plain gradient-ascent updates. This differs from the native `offline_mlp`
control, which trains an MSE surrogate and uses Adam for candidate search.

It is reported as **Standard GA adaptation** because the original TensorFlow
implementation is replaced, constrained coordinates are introduced, and the
surrogate additionally receives model-scale and training-step context.

## CMA-ES

The registered `cma_es` method trains five bootstrapped Gaussian MLPs by
default, freezes them, and runs one independent full-covariance CMA-ES search
per requested candidate. Search starts from the best unique logged designs;
random feasible starts fill the candidate budget when there are too few unique
logs. Only the ensemble's standardized mean prediction is optimized.

The CMA population lives in unconstrained coordinates. Softmax maps it to the
data-mixture simplex, while scaled sigmoid maps it to box tasks. The resulting
**CMA-ES adaptation** is fully PyTorch-native and does not depend on `pycma`.

## REINFORCE

The registered `reinforce` method fits the same frozen probabilistic ensemble,
then learns the mean of a fixed-variance Gaussian policy with the score-function
estimator. Each update standardizes the surrogate rewards within its sampled
batch. Final policy samples are mapped through the task design space and only
then returned for evaluator-owned oracle evaluation.

This is reported as **REINFORCE adaptation**: TensorFlow is replaced by
PyTorch, fidelity context is appended to surrogate inputs, and policy samples
are constrained through the common simplex/box mapping.

Source provenance and the boundary on paper-parity claims for all three new
methods are recorded in the
[standard baseline source audit](BASELINE_SOURCE_AUDIT.md).

## COM

The Conservative Objective Model fits the same MLP while constructing
adversarial designs that increase the predicted utility. Its training loss
penalizes predicted improvement on those adversarial designs relative to
logged designs. A learned non-negative multiplier adjusts the strength of the
penalty around an overestimation limit. Final candidates are optimized through
the conservative surrogate from top logged starting points.

This is a compact PyTorch implementation of the COM idea, not a byte-for-byte
port of the original
[`design-baselines`](https://github.com/brandontrabucco/design-baselines)
implementation.

The registered `coms` method receives only `OfflineProblem`, standardizes the
logged design/fidelity features and utility, and returns unevaluated
candidates. Oracle evaluation is performed later by the benchmark runner. The
legacy `ConservativeObjectiveModelOptimizer` remains only for reproducing old
result paths.

## BDI

The BDI adaptation first fits RBF kernel ridge regression to standardized
logged utilities. Candidate support points are initialized from top logged
designs. During optimization, their loss combines:

1. high utility under the forward kernel surrogate; and
2. a backward distillation term that asks the support points with optimistic
   labels to reconstruct the logged utility landscape.

The RBF length scale defaults to the median non-zero pairwise distance in the
logged feature set. This preserves a differentiable forward/backward
distillation mechanism while avoiding the original JAX, Neural Tangents, and
Design-Bench dependencies. It should therefore be described as a BDI
adaptation when reporting results. The reference implementation is the
authors' [`GGchen1997/BDI`](https://github.com/GGchen1997/BDI) repository.

The registered `bdi` method uses the same oracle-separated contract and
supports both simplex and box spaces. It records the RBF substitution in
method provenance. The legacy `BackwardDistillationOptimizer` remains only
for reproducing old result paths.

## Reference-normalized utility

For a statistic `u` and full logged reference utilities `D`, the score is

```text
(u - min(D)) / (max(D) - min(D))
```

The best, median, and mean candidate utilities are normalized independently.
The score is not clipped. A best-utility score above one means the generated
best candidate exceeds the maximum logged utility, not that it exceeds the
known global optimum.

## Diversity and novelty

Candidate diversity is the mean pairwise Euclidean distance among normalized
candidate designs. Candidate novelty is the mean nearest-neighbor Euclidean
distance from each normalized candidate to the logged dataset. Synthetic
coordinates are mapped to `[0, 1]` by their box bounds before these metrics are
computed.
