# PGS transition/action consistency gate

Status: transition construction implemented and tested; **PGS is not yet a
registered method**. CQL/SAC training and policy rollout remain to be written.
This is not a PGS reproduction or a formal benchmark result. See the pinned
[source audit](RANKING_POLICY_METHOD_SOURCE_AUDIT.md#3-pgs-offline-rl-learns-gradient-search-step-sizes).

## Why this gate comes first

The public PGS code infers a coordinate-wise action by dividing a logged
displacement by a surrogate gradient, then rescales/truncates it. Deployment
uses another scale. An inferred action can therefore describe a different
next state from the one whose utility supplies the reward. Copying those
conventions would make a replay tuple internally inconsistent.

Our adapter instead requires every retained replay action to reconstruct its
logged next design, within a declared numerical tolerance, using the exact
same function that future policy rollout will call. No oracle, synthetic
reward, action clipping, or movement of logged designs into the interior is
used to pass this check.

## State, action, and shared transition map

Let `x` be a raw feasible design, `c` its fixed fidelity context, and `f` the
frozen, visible-data-trained surrogate. Differentiate `f` through its feature
normalization to obtain `g(x,c) = grad_x f(x,c)` in **raw design units**.
The action is a signed vector `a` in `[-1, 1]^d`, with one entry per design
coordinate. Fidelity is never an action coordinate. Define

```text
next_x = project_domain(x + step_scale * a * g(x, c))
```

Multiplication is element-wise. `project_domain` is Euclidean projection onto
the probability simplex or coordinate-wise box clipping. This preserves
logged boundary designs such as zero mixture weights. Both replay checking
and future rollout use `projected_gradient_step`; there must not be a second
rollout-only step multiplier.

The surrogate must be row-independent. Input differentiation neither updates
its weights nor writes parameter gradient buffers. For future finite-horizon
CQL/SAC, policy and critic state must concatenate visible-statistics-normalized
design/context features with `remaining_steps / config.max_horizon`. The next
state's remaining steps are one less. Use this identical representation during
training and rollout. Limit rollout horizon to the maximum retained fragment
horizon (or a smaller predeclared value); do not request 50 search steps when
every certified training fragment contains only one step. Target-fidelity
generalization remains an explicit adaptation, not a guarantee of accuracy.

## Visible trajectory construction

1. Group visible rows by exact equality of **all** logged fidelity fields.
   With no context, use one group. Never join different scales or step counts.
2. Select each group's top utility quantile, default top 20%, using only
   visible labels. Ties at the threshold are retained. This is a per-fidelity
   adaptation; it does not reveal any part of the benchmark's hidden region.
3. Skip groups whose selected pool has fewer than two rows, reporting counts.
   Do not silently expand a pool. If none remain, fail explicitly.
4. Sample trajectories without replacement within each trajectory, using the
   run's generator. Different trajectories can reuse rows. They are random,
   not sorted by utility. Bound the horizon by `min(max_horizon, pool_size-1)`.

For each logged edge `(x_i, u_i) -> (x_j, u_j)`, keep its original endpoints
and reward `u_j - u_i` in raw maximization utility units. A future explicit
reward standardization may use visible data only and must be recorded; it
must not replace these rewards with surrogate predictions.

## Action calibration and rejection

For each coordinate with nonzero displacement, require gradient magnitude at
least `gradient_floor`. A zero-gradient coordinate is admissible only when
its logged displacement is exactly zero, in which case its inferred action
is zero. Do not replace small gradients with epsilon and pretend the resulting
action follows the original gradient dynamics.

Infer the unscaled gain `b = (x_j-x_i)/g_i` for usable coordinates. Reject an
edge if any gain is non-finite or exceeds `max_step_scale/action_margin`.
From the surviving visible edges, fit one diagonal scale per run:

```text
step_scale[k] = max(1, action_margin * max_edges(abs(b[edge,k])))
a[edge,k] = b[edge,k] / step_scale[k]
```

This produces bounded actions without clipping. Store the scale with the
future model checkpoint and reuse it unchanged at target-fidelity rollout.
Reconstruct each next design with the shared map and reject edges whose
maximum coordinate error exceeds `reconstruction_tolerance`. Only certified
edges receive their logged utility-difference reward in the returned replay.
Scale fitting may include an edge subsequently rejected by the numerical
reconstruction check; it still uses visible data only and is not refitted.

Removing an edge splits its trajectory into contiguous fragments. Do not
bridge the gap or reuse a reward for a new edge. Each fragment's final edge
is terminal and its remaining horizon is recomputed. These are explicitly
finite-horizon fragments, not an assertion that the physical design state is
an absorbing terminal. Horizon-conditioned policy/critics are necessary to
make this convention unambiguous.

Defaults for this gate (not frozen experimental settings): 32 trajectories
per eligible group, maximum horizon 50, top fraction 0.2, gradient floor
`1e-8`, action margin `1.05`, maximum step scale `1e6`, reconstruction
tolerance `1e-5`. All are validated by `PGSTransitionConfig`.

## Tests and remaining gates

Tests cover known simplex/box projections, boundary points, raw gradient
units, exact replay reconstruction in float32/float64, fixed fidelity,
unmodified labels/model/input tensors, random-generator isolation, small
pools, ties, bounded horizons, action/gain rejection, partial trajectory
filtering, terminal masks, and failure with no eligible data. CUDA has a
conditional test; it is not locally verified when CUDA is unavailable.

Before registering `pgs`, still implement and validate:

- visible-only surrogate fitting followed by freezing;
- a horizon- and fidelity-conditioned tanh-Gaussian actor, twin critics and
  target critics, SAC entropy handling, CQL conservative penalty, isolated
  actor/critic gradients, and target updates;
- policy rollout using this exact transition map and saved scale, fixed target
  fidelity, valid remaining-horizon inputs, and an exact candidate budget;
- full method reproducibility, oracle isolation, synthetic smoke reports, and
  measured rejection/group coverage on actual data-recipes logs.

Filtering can bias the retained transition distribution, and diagonal gain
calibration can produce very different search scales from the original code.
Report accepted/rejected counts, pool sizes, reconstruction errors and scales;
freeze these choices before formal experiments. Do not tune them against
hidden oracle scores or claim source parity. The future method remains
**PGS adaptation**, not an inverse-generative method or a REINFORCE alias.
