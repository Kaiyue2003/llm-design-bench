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

## BO-qEI

The registered `bo_qei` method fits a native exact Gaussian Process with an
ARD RBF kernel to standardized design, model-scale, and training-step
features. It jointly optimizes the requested candidate batch using a Monte
Carlo estimate of q-Expected Improvement over the best logged standardized
utility. Fixed base samples make each seeded acquisition optimization
deterministic and reduce gradient noise between optimization steps.

## GA on GP

The registered `ga_on_gp` method shares the exact GP surrogate with BO-qEI,
but performs plain constrained gradient ascent on posterior mean. Mixed top
logged and random starts prevent all candidates from starting at duplicated
logged designs.

## MC-Dropout

The registered `mc_dropout` method trains an MLP with generator-controlled
dropout. Dropout remains active when scoring candidates, yielding Monte Carlo
mean and epistemic standard-deviation estimates. Candidate search maximizes
`mean - uncertainty_weight * std`, a conservative lower confidence bound for
a maximization task.

All three are reported as adaptations because the public SPADE artifacts do
not provide the final LLM-DM preprocessing, task-specific hyperparameters, or
baseline evaluation scripts. Their source and implementation boundaries are
recorded in the
[GP and uncertainty source audit](GP_UNCERTAINTY_BASELINE_AUDIT.md).

## Tri-Mentoring

`tri_mentoring` independently trains three MSE proxies. Each candidate starts
with a fresh copy of the pretrained ensemble. At each step, it samples feasible
neighbors by perturbing unconstrained design coordinates, takes a majority
vote on all unique pairwise comparisons, and mentors each proxy on the pairs
where it disagrees. Ties use the paper's strict-greater-than indicator (vote 0).

For each disagreeing proxy, a virtual SGD step on pairwise BCE-with-logits
links soft labels to model parameters. Logged-data MSE through that virtual
model supplies the label meta-gradient. Updated labels are clipped to [0, 1],
detached, and used for one real SGD update. Adam then maximizes the adapted
ensemble mean in constrained design coordinates. No oracle is available to
either the inner or outer optimization.

The implementation uses `torch.func.functional_call` and native autograd;
`higher` is not required. Defaults retain a width-2048 two-hidden-layer ReLU
network, 200 training epochs with cosine-decayed Adam, 200 search steps,
10 neighbors, and learning rates 0.001 (search/mentoring) and 0.1 (labels).
Each proxy selects a checkpoint using its own 10% split of the visible data:
Pearson correlation, falling back to MSE for one-row or constant-label
validation sets. Fewer than three visible rows use final-epoch models.

Neighborhood geometry, context conditioning, small-data behavior, and
initialization are benchmark adaptations; see the
[forward source audit](FORWARD_METHOD_SOURCE_AUDIT.md). The method is reported
as **Tri-Mentoring adaptation**. It adapts proxies separately for every
candidate, so its search cost grows with both candidate budget and search steps.

## ICT

`ict` fits three independent MSE proxies using the shared mentoring trainer.
It first adapts a single ensemble along a moving trajectory initialized at the
best visible logged design. Each adaptation step advances that design using
Adam, samples feasible neighbors, and rotates teacher roles in order 0, 1, 2.
The teacher creates detached pseudo-labels; the other two proxies each select
`remember_count` low-loss samples for the OTHER student. Both selections are
computed before either student updates, with stable index-order tie breaking.
Each rotation regenerates labels from the current teacher.

For each student, weights start at one. A differentiable virtual SGD update
on the selected, row-wise weighted MSE yields a logged-data MSE meta-gradient.
The paper's raw weight gradient is used; updated weights are clipped to [0, 2]
and detached before one real SGD update from the original parameters. Setting
`reweighting=False` disables weight adaptation but retains small-loss exchange.
No true scores are requested for pseudo-labeled designs.

Final candidate search starts afresh from top unique logged designs, filling
any shortage with random feasible starts. The adapted ensemble is frozen and
shared across these independent Adam trajectories. Unlike Tri-Mentoring,
ICT does NOT continue adapting a separate ensemble for every final candidate.

Defaults: width 2048, 200 pretraining epochs, batch size 128, 100 adaptation
steps, 100 final search steps, 128 neighbors, eight remembered samples,
noise 0.1, and learning rates 0.001 (design/proxy) and 0.1 (weights).
Inputs retain logged fidelity context during training/meta-supervision;
neighbors and final candidates always use target fidelity. Both mentoring
methods use unit scale for constant/near-constant feature columns, avoiding
epsilon-amplified out-of-support features in very small datasets.

This is **ICT adaptation**, independently implemented using native PyTorch
functional updates, not a port of the original `higher`/Adam runtime. Search
geometry, initialization, fresh teacher labels, raw meta-gradients, and
small-data handling are explicit differences. See the
[forward source audit](FORWARD_METHOD_SOURCE_AUDIT.md).

## RoMA

`roma` has a width-64, two-hidden-layer Softplus Gaussian proxy. Pretraining
uses noisy normalized design features with unchanged logged fidelity context.
For each minibatch, projected gradient ascent finds adversarial weights that
increase Gaussian negative log-likelihood (NLL). The outer Adam update treats
that perturbation as fixed and updates the base parameters through
`torch.func.functional_call`; the base model is never overwritten with the
adversarial weights. Input normalization uses only visible rows, with unit
scale for constant columns. There is no oracle-based checkpoint selection.

Each candidate maintains its own adapted parameter dictionary. Before every
candidate update, the temporary weights reset to the pretrained reference.
Projected descent then minimizes the norm of the candidate score's input
gradient plus consistency NLL against the previous adapted model's detached
mean at the CURRENT candidate. The first target is the pretrained prediction
at target fidelity, not a logged score from a potentially different fidelity.
The input gradient is taken with respect to unconstrained design coordinates;
it includes the softmax/sigmoid map but never the fixed fidelity coordinates.
This smoothness update uses second-order autograd through temporary weights.

Every temporary parameter tensor is bounded by
`||theta_t - theta_base|| <= weight_radius * ||theta_base||`.
The projection also covers biases and variance parameters; zero-norm tensors
stay fixed. Candidate ascent treats adapted weights as constants and optimizes
`q - (q - q_initial)**2 / (2 * region)`, where
`q = mean - uncertainty_weight * logstd`. This is a penalty on SCORE change,
not distance from the initial mixture. The uncertainty term uses log standard
deviation, not a Gaussian lower confidence bound; it defaults to zero.

Defaults are 50 pretraining epochs, batch size 128, pretraining rate 0.001,
20 adversarial steps, relative weight radius 0.0005, input noise 0.2,
100 adaptation steps per candidate update, 500 candidate updates at rate
0.003, consistency weight 1, region 4, and global gradient clipping at 1.
`adaptation_steps=0` disables local adaptation; `weight_radius=0` disables
both adversarial perturbations and local weight changes (Gaussian smoothing
remains unless `input_noise_std=0`). All resolved settings enter result metadata.

This is **RoMA adaptation**, not TensorFlow parity: Gaussian NLL, explicit
tensor-wise projection, independent candidate state, constrained-coordinate
smoothness, all-visible-data training, and multi-fidelity conditioning are
declared choices. See the [source audit](FORWARD_METHOD_SOURCE_AUDIT.md).
Default search is computationally expensive because it differentiates through
input gradients for each candidate. Use reduced smoke settings to test the
pipeline, not to report final performance.

## LTR

`ltr` is **LTR adaptation (RaM-ListNet)**: an independently implemented
two-hidden-layer ReLU ranking model. ListNet minimizes cross entropy between
the softmax of standardized visible utilities and the model's log-softmax,
averaging over lists. Each list samples rows without replacement; different
lists may overlap. Training lists are generated lazily each epoch. Fixed
validation lists use the same visible row pool and select a deep-copied best
checkpoint by ranking loss; they do not measure unseen-row generalization.

Features include logged fidelity context, standardized using only visible
data. Search freezes the model and maximizes its output normalized by the
mean and population standard deviation of predictions on visible logs.
Constant prediction/feature columns use unit scale. Candidate Adam updates
only unconstrained design coordinates, mapped to the simplex or box, while
holding fidelity at its target. No hidden elites or oracle are accessible.
Top unique logged designs initialize search; missing starts are sampled
feasibly. With one row or utility standard deviation below `minimum_std`,
training and search are skipped explicitly and these initial candidates are
returned. Such a run is not evidence that a ranking model learned anything.

Compact defaults: `hidden_size=64`, `surrogate_epochs=50`, `list_length=32`,
`lists_per_epoch=256`, `batch_size=32` lists, `validation_lists=32`, training
Adam rate `3e-4`, `weight_decay=1e-5`, `solver_steps=200`, search rate `1e-3`,
and `minimum_std=1e-6`. Effective list length is bounded by the visible row
count; short final batches are retained. All settings are configurable and
recorded by the runner. These are not frozen formal experiment budgets or
the original paper settings. RankCosine is not implemented yet. See the
[source audit](RANKING_POLICY_METHOD_SOURCE_AUDIT.md).

## MATCH-OPT

`match_opt` is **MATCH-OPT adaptation**, independently implemented with a
row-independent LeakyReLU (slope 0.3) MLP. It combines supervised utility MSE
with MSE between observed pair differences `utility_v - utility_u` and a
five-node left-endpoint approximation of the surrogate-gradient line integral
from `x_u` to `x_v`. The latter remains differentiable with respect to model
weights. Summing row outputs before input differentiation avoids constructing
a quadratic batch Jacobian; batch-coupled layers are not supported here.

Pair construction groups rows by exact equality of all logged context fields
(scale AND steps), orders each group by utility, partitions into non-empty
buckets, and samples monotone trajectories through the buckets each epoch.
Adjacent trajectory points supply matching pairs. Each supervised minibatch
draws a matching minibatch without replacement within that draw; pairs can be
reused between updates. All visible rows enter value fitting once per epoch,
including singleton contexts that cannot supply matching pairs. If every
context is a singleton (or there is only one row), the method raises an explicit
error before training; the runner records a failed run, not an MSE fallback.
Without context, all rows form one group. Ties and duplicate designs are
retained, with zero-displacement pair counts reported explicitly.

Feature/utility normalization uses visible logs only. Paths hold fidelity
constant and linear interpolation stays feasible on the simplex or box.
The final epoch's frozen model guides Adam ascent of utility at fixed target
fidelity, starting from unique top logged designs plus feasible random fills.
The method cannot access the oracle or hidden-region diagnostics.

Compact defaults are `embedding_dim=8` (hidden widths 128/32/8),
`surrogate_epochs=50`, `batch_size=128`, `bucket_count=32`,
`quadrature_nodes=5`, training Adam rate `1e-4`, `matching_weight=1`,
`solver_steps=150`, search rate `1e-3`, and `minimum_std=1e-6`.
Matching weight must be positive; a zero-weight MSE baseline is a different
method. These defaults are not frozen formal experiment budgets. The source
uses embedding width 32, 128 buckets, and 201 epochs; context grouping and
separate supervised/pair minibatches are additional declared adaptations.
See the [source audit](RANKING_POLICY_METHOD_SOURCE_AUDIT.md).

## PGS

PGS's shared projected-gradient transition map and visible logged replay
construction are implemented in `optimizers/pgs_transitions.py`. The module
checks that inferred actions reconstruct their logged next designs before
attaching utility-difference rewards, and rejects incompatible edges instead
of clipping actions or fabricating rewards. See the
[transition design](PGS_TRANSITION_DESIGN.md) for exact equations, defaults,
same-fidelity pool selection, boundary handling, finite-horizon fragments,
filtering caveats, and tests.

The registered **PGS adaptation** fits one two-hidden-layer ReLU surrogate on
all visible standardized data, using the shared cosine learning-rate schedule
and final epoch (no validation/oracle selection). It freezes that proxy before
building certified replay. State inputs concatenate normalized design/fidelity
features and remaining horizon divided by `max_horizon`. Replay rewards remain
logged utility differences, divided by the visible utility population standard
deviation (floored by `minimum_std`); no reward centering or proxy rewards.

Native CQL/SAC uses a tanh-Gaussian actor with learned log-standard-deviation
multiplier/offset and log-std clamped to [-20, 2]. Sampling uses the run-local
generator and a stable tanh log-Jacobian. Two independent critics have frozen
Polyak-updated targets. The Bellman backup uses minimum target Q; like the
pinned source, `backup_entropy=False` by default. Enabling it subtracts
`alpha * log_pi` in the backup. Terminal transitions receive only their reward.
The actor always minimizes `alpha * log_pi - min(Q1,Q2)`; automatic entropy
tuning uses target entropy `-design_dim` and initial alpha 1.

CQL uses equal counts of uniform [-1,1], current-policy and next-policy
proposal actions, ALL evaluated at the current state. Its per-critic penalty
is `T * logsumexp((Q - proposal_log_density) / T) - Q_logged`, using the
source's unnormalized logsumexp convention. Densities/actions are detached
for critic updates. Actor updates freeze critic parameters while retaining
action derivatives; entropy updates detach policy log densities. There is no
learned CQL Lagrange multiplier or oracle-based checkpoint selection.

At rollout, the actor uses tanh of its Gaussian location (deterministic
evaluation), not a sampled action. The surrogate supplies raw design gradients
at fixed target fidelity. The exact replay transition map and saved diagonal
scale are reused. Search starts preserve exact feasible logged boundaries and
fill missing candidates with feasible random designs. The horizon is bounded
by both `solver_steps` and the maximum retained fragment length; it is recorded
in diagnostics along with scales, replay coverage and rejection counts.

Compact defaults: hidden width 64 for surrogate, actor and critics; 50 surrogate
epochs; batch size 128; 1,000 RL updates; surrogate/RL Adam rates `3e-4`;
discount 0.99; target update rate 0.005; CQL weight 5, ten samples per proposal
distribution, temperature 1; automatic entropy enabled; 50 requested search
steps; `minimum_std=1e-6`. Transition defaults are in the linked design.
`automatic_entropy=False` holds alpha fixed; `backup_entropy=True` is a
declared ablation. Only float32/float64 replay is supported. These are not
frozen formal budgets or the source's 401,000-update configuration.
Empty/invalid replay fails explicitly, without pool expansion or a substitute
optimizer. GPU and real data-recipes coverage require separate validation.

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
