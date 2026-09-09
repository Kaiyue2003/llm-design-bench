# Ranking and Policy Method Source Audit

This source/design audit now also records the integrated `ltr` and `match_opt`
adaptations. `pgs` remains planned and is not registered.
Implementation order: **LTR and MATCH-OPT (integrated) -> PGS**. These have only
tests and reduced-budget smoke validation, not formal eight-seed experiments.

## Authoritative sources

| Method | Paper | Author repository and pinned revision | Source runtime |
| --- | --- | --- | --- |
| LTR / RaM | [ICLR 2025](https://arxiv.org/html/2410.11502) | [Offline-RaM](https://github.com/lamda-bbo/Offline-RaM/tree/389e4bcf68c3e645e3a36e0f84ebdf05a76c235f), `389e4bcf68c3e645e3a36e0f84ebdf05a76c235f` | PyTorch 1.13.1 core; legacy Design-Bench / TensorFlow utilities |
| MATCH-OPT | [ICML 2024](https://proceedings.mlr.press/v235/hoang24a.html) | [MatchOpt](https://github.com/azzafadhel/MatchOpt/tree/aae3f579a04400206eaa7abe836961c7af94b508), `aae3f579a04400206eaa7abe836961c7af94b508` | PyTorch core; Design-Bench / TensorFlow data pipeline; no dependency lock |
| PGS | [AAAI 2024](https://arxiv.org/html/2405.05349) | [PGS](https://github.com/yassineCh/PGS/tree/54837299b33f986563b15176695e2c83472ffdda), `54837299b33f986563b15176695e2c83472ffdda` | PyTorch 1.7.1 core; TensorFlow 2.3.2 and legacy Design-Bench dependencies |

No explicit project-wide license file was found at these revisions. Do not
copy or vendor source into this package pending license clarification. Plan
independent implementations from the published algorithms, using the source
only to audit behavior. Third-party assets in a repository can have their own
licenses; this observation is not a claim that every file is unlicensed.
Do not install these repositories' full environments or execute their research
entry points. In particular, MatchOpt executes CUDA setup, cache loading, and
training at module import. The source checkouts are not runtime dependencies.

## Shared integration requirements

- Use `OfflineBBOMethod` / `OfflineProblem` / `RunContext` / `MethodResult`;
  return exactly the requested number of feasible, unevaluated candidates.
- Consume only visible logged data. Utilities already maximize performance;
  never negate loss again or access hidden labels, oracle callbacks, or
  pretrained checkpoints of unknown training provenance.
- Preserve the outer low-utility offline split. Any top-data selection below
  means top data **within the visible split**, not the full logged dataset.
- Train on logged fidelity context, search mixtures at the fixed target
  1B/19,500-step context, and record all multi-fidelity adaptations. Use the
  same interface for fixed-1B ablations.
- Isolate random generators; support CPU and configured dtype/device. Keep
  W&B, global CUDA setup, dataset downloads, and research-script filesystem
  side effects out of methods. Evaluation belongs to the benchmark evaluator.
- Source defaults below describe research code, not approved LLM-DM settings.
  Declare smoke and full-run configurations separately; passing a smoke test
  is not numerical reproduction or admission to the eight-seed results table.

## 1. LTR: ranking surrogate with calibrated gradient search

The RaM mechanism combines sampled training lists, a learning-to-rank loss,
and surrogate-output normalization before search. It is not merely an MSE
proxy with a different name. The public training script runs both ListNet and
RankCosine, although `config/default.yaml` defaults to MSE. The integrated
adapter: **LTR adaptation (RaM-ListNet)**. RankCosine can be a separately named
configuration/ablation, not a second paper baseline. Only ListNet is currently
implemented; RankCosine is not an exposed configuration yet.

Audited files: `model.py`, `utils.py`, `main_from_scratch.py`,
`config/default.yaml`, and `run_from_scratch.sh` in the pinned repository.
The network has two 2,048-wide ReLU hidden layers. Common defaults include
10,000 sampled lists of length 1,000, batch size 128 lists, 100 training epochs,
Adam learning rate `3e-4`, weight decay `1e-5`, and continuous Adam search
for 200 steps at `1e-3` from 128 top logged starts.

Retain ListNet's target-softmax/prediction-log-softmax cross entropy with stable
numerics and explicit list dimensions. RankCosine, if exposed, centers both
vectors and minimizes one minus their cosine similarity. Normalize search
scores using the mean and standard deviation of predictions on visible logs;
this is not sigmoid calibration. Handle zero prediction variance explicitly.

Integration hazards and tests:

- Source list sampling uses CUDA and materializes every list. Sample indices
  lazily with the run generator and cap list length at visible row count.
  Test tiny datasets, ties, one-element dimensions, and both supported dtypes.
- Source validation splits sampled lists, which can share underlying rows;
  it is not evidence of generalization to unseen observations. Declare the
  adapter's validation split unit and use only visible data for selection.
- Snapshot best model weights by value. The source's direct `state_dict()`
  assignment can retain references to subsequently mutated tensors.
- The optional plain-gradient search negates the score before adding its
  gradient, unlike the correctly signed Adam minimization path. Test utility
  ascent on an analytic objective rather than porting that sign convention.
- Remove hidden-elite diagnostics (`eval_elites`) and final `task.predict`
  calls. These are reporting paths in the source, not permission for an
  offline adapter to access the held-out region.
- Condition ranking predictions on fidelity and constrain final design
  search; record these deviations from the original Design-Bench setup.

## 2. MATCH-OPT: line-integral gradient matching

The method constructs utility-ordered trajectories from logged samples and
fits both observed values and adjacent-pair utility differences. For a pair
`(x_u, u_u), (x_v, u_v)`, use the explicit orientation
`delta_u = u_v - u_u`, `delta_x = x_v - x_u`. Match `delta_u` to the dot product
of `delta_x` with the surrogate gradient integrated along their straight path.
The pinned implementation uses five left-endpoint quadrature nodes `i/5`,
`i=0..4`, plus ordinary supervised MSE with coefficient one. Preserve this
finite-quadrature objective; replacing it with an exact difference of network
outputs is a different training objective. Explicit orientation also avoids
ambiguity in signs in the paper's HTML presentation.

Audited files: `gm_surrogate_traj_sampling.py`, `nets.py`, `util.py`, and
`data_grabber.py`. Defaults include 128 utility buckets, a LeakyReLU MLP with
hidden widths 512/128/32, 201 epochs, Adam `1e-4`, and 150 candidate Adam steps
at `1e-3`. The source runs four seeds; that is not the project's eight-seed
protocol and must not determine our seed count.

Integration hazards and tests:

- The source builds a full batch-by-batch Jacobian before taking its diagonal.
  For a row-independent MLP, differentiate the sum of outputs to obtain the
  per-row input gradients without the quadratic batch dimension. Preserve
  the higher-order graph and test equivalence and parameter gradients.
- Bucket count must be bounded by available rows; empty buckets can otherwise
  eliminate training. Test ties, small groups, finite gradients, and positive
  utility direction on a linear function with a known line integral.
- Remove periodic oracle reporting and absolute cache paths. Train directly
  from `OfflineProblem`, without a cache produced by running other baselines.
- **Multi-fidelity rule:** the adapter uses same-(scale, steps) trajectory
  pairs for mixture-gradient matching, while supervised fitting may use all
  visible rows. A cross-fidelity utility difference is not solely a mixture
  effect. Report eligible group/pair counts and reject unsupported data when
  no pairs exist; do not silently run only MSE and label it MATCH-OPT.
- A full joint-input path that differentiates context as well as mixtures is
  a possible explicit alternative, not the default: it interpolates fidelity
  and changes the interpretation of the gradient-matching term.

## 3. PGS: offline RL learns gradient-search step sizes

PGS first fits a frozen MSE surrogate. It then constructs logged-data
transitions with utility-difference rewards and trains a policy to choose
coordinate-wise step sizes for surrogate gradient search. The public core
uses conservative offline RL (CQL with SAC), twin critics, target critics,
and a tanh-Gaussian actor. A noisy gradient optimizer or the existing
REINFORCE method would not constitute PGS. Its actor does not make this the
inverse-generative workstream assigned separately to a teammate.

Audited files: `surrogate.py`, `generate_trajectories.py`,
`conservative_sac.py`, `evaluate_policy.py`, and `pgs.py`. Source defaults use
a two-layer 2,048-wide surrogate; the top 20% of visible observations form
the random, not necessarily ascending, trajectory pool. The configuration
creates 20,000 trajectories of 50 transitions (one million transitions),
uses 256-wide policy/critic networks and batches of 256, and requests
401 epochs of 1,000 RL updates. These defaults need an explicit runtime
budget; they are not appropriate as an unannounced CPU smoke run.

Integration hazards and tests:

- Reconstructing a step-size action divides coordinate displacement by the
  surrogate gradient. Near-zero gradients and action clipping can make the
  stored action inconsistent with its next state. Define a shared transition
  map and record reconstruction residuals; do not hide this with `nan_to_num`.
- Source trajectory construction scales actions by division by 10, whereas
  continuous deployment uses `0.05 * sqrt(d)` as its step scale. The adapter
  must explicitly reconcile training and rollout units, including simplex
  constraints; test transition reconstruction before training a policy.
- The terminal mask is hardcoded for length 50, and sampling without
  replacement can fail on small pools. Bound the horizon by available data,
  generate correct terminal flags, and reject pools with fewer than two states.
- A source rollout passes `vec=True` to `step`, whose signature has no such
  parameter. This is a static integration defect, not a measured run failure.
- Test CQL loss, actor/critic gradient isolation, target updates, termination,
  reproducibility, and feasibility. Remove intermediate oracle evaluations
  and never select the policy checkpoint using hidden performance.
- **Multi-fidelity rule:** propose same-fidelity logged transitions, with
  fidelity-conditioned actor and critics and fixed target context at rollout.
  Audit eligible pool sizes first. Do not join different fidelities and call
  their utility difference a reward for a mixture-only action. If unsupported,
  report that result rather than inventing oracle or surrogate reward labels.

## Next implementation gate

LTR is registered with ranking-loss, search-sign, small-data, checkpoint-copy,
oracle-isolation, exact-budget, and reproducibility tests. It uses resampled
training lists and fixed validation lists sharing the visible row pool, not a
claim of unseen-row generalization. Constant/near-constant utilities and one-row
datasets return logged/random starts with explicit skipped-training diagnostics.
Constant feature and prediction scales use one; prediction calibration sees
only visible rows. The model is frozen before target-fidelity search.
See [method settings](METHODS.md#ltr) for compact configurable defaults.

MATCH-OPT retains left-node quadrature with differentiable input gradients
computed by summing independent row predictions. Each epoch builds monotone
trajectories separately within exact logged-context groups. All visible rows,
including singleton contexts, enter value fitting; matching pairs are sampled
for each value minibatch. The final epoch is retained without oracle selection.
Eligible rows/groups, effective buckets, pair counts, zero-displacement pairs,
and both training-loss histories are reported. No eligible group raises a
clear error before training; there is no silent MSE-only fallback. Repeated
designs are retained as logged observations and diagnosed: conflicting labels
at identical designs create irreducible matching error, not a usable gradient.
Source import-time CUDA/cache/reporting code is not imported or copied.
See [method settings](METHODS.md#match-opt) for compact configurable defaults.

PGS's [transition/action-consistency gate](PGS_TRANSITION_DESIGN.md) is now
implemented and tested: projected raw-design gradient steps, visible-only
diagonal action calibration, same-fidelity pools, reconstruction checks, and
contiguous finite-horizon fragments. CQL/SAC training and policy rollout remain
to be implemented; PGS is not registered. All three are or will be labeled adaptations; none of
these pinned repositories establishes exact parity with the SPADE LLM-DM table.

Complete the agreed method roster before freezing and running formal
experiments. Colab preparation changes the execution environment, not this
order; reduced-budget integration smoke runs do not count as full trial seeds.

### LTR integration validation (2026-09-09)

The CPU smoke run used Ackley and Branin, 32 logged rows per task, seed 38,
and eight candidates. Reduced settings were hidden width 8, three training
epochs, list length 8, eight training lists per epoch, list batch size 4,
four validation lists, and five candidate-search steps. Both task runs
completed through the unified evaluator and report writer. Local artifacts
are under ignored `results/ltr_integration_smoke_20260909/`; they are not
publication reference results, and no performance-based settings were chosen.
CUDA and real data-recipes execution remain unverified in this local environment.

### MATCH-OPT integration validation (2026-09-09)

CPU tests cover linear integral orientation, the analytic quadratic left-node
formula, equivalence to a full batch Jacobian, and higher-order parameter
gradients checked by finite differences. Integration tests cover simplex/box,
fixed target context, exact same-fidelity grouping, singleton exclusion from
matching only, duplicate designs, constant labels, reproducibility, and
oracle isolation, including failed runs with no eligible pairs.

The reduced-budget smoke used Ackley and Branin with 32 logged rows each,
seed 38, and eight candidates. Settings: embedding width 2, three epochs,
batch size 8, eight buckets, five quadrature nodes, and five search steps.
Both runs completed through the unified evaluator/report writer. Local outputs
are under ignored `results/match_opt_integration_smoke_20260909/`, not the
publication reference table. CUDA and actual data-recipes group coverage
remain unverified; verify eligible same-fidelity counts before formal runs.
