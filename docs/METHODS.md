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
