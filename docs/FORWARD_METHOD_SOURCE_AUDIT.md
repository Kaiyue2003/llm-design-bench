# Forward Offline Method Source Audit

This audit covers the integrated `tri_mentoring` adaptation and planned
`ict` and `roma` method IDs.
It fixes authoritative sources, implementation constraints, and the order in
which shared PyTorch components should be introduced.

## Repositories and pinned revisions

| Method | Paper | Public repository | Audited commit | Original runtime |
| --- | --- | --- | --- | --- |
| RoMA | [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/24b43fb034a10d78bec71274033b4096-Paper.pdf) | [`sihyun-yu/RoMA`](https://github.com/sihyun-yu/RoMA) | `08b1ab54ce3daf383e3ca3a1227d75a471afe023` | TensorFlow 2.3 / TensorFlow Probability |
| ICT | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/ae8b0b5838ba510daff1198474e7b984-Paper-Conference.pdf) | [`mila-iqia/Importance-aware-Co-teaching`](https://github.com/mila-iqia/Importance-aware-Co-teaching) | `6d51fcc04e7a0a60c7ad578ae4c4f743bc678ae4` | PyTorch 1.7 and `higher` 0.2.1 |
| Tri-Mentoring | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/f189e7580acad0fc7fd45405817ddee3-Paper-Conference.pdf) | [`GGchen1997/parallel_mentoring`](https://github.com/GGchen1997/parallel_mentoring) | `db1a055073cc0c22afc12c5eace55f8f030caa52` | PyTorch 1.7 and `higher` 0.2.1 |

No explicit license file was present in any of the three audited repositories
at the pinned revision. The project must not vendor or copy their source. New
implementations should be clean-room implementations from the published
algorithms, with source behavior used only for compatibility auditing. A
license added by an upstream project later does not retroactively change this
revision; any such change must be audited and pinned separately.

## Shared benchmark boundary

All three adapters must receive only `OfflineProblem`, operate on maximization
utility, and return unevaluated candidates. The original research scripts call
Design-Bench's `task.predict` while reporting candidate progress. Those calls
must not enter an `OfflineBBOMethod`; only the unified evaluator may call the
oracle after candidate generation completes.

For multi-scale LLM-DM, surrogate inputs include data-mixture coordinates,
model scale, and training steps. Candidate optimization fixes the context to
1B and 19,500 steps and updates only the mixture coordinates. Softmax enforces
the simplex. These context and constraint changes are material adaptations.

The papers and scripts reuse `K` for neighborhood or retained pseudo-labeled
sample counts. The benchmark separately uses `K=128` for the number of final
candidates. Implementations must use unambiguous names such as
`neighbor_samples`, `remember_count`, and `candidate_budget`.

## Tri-Mentoring

The source trains three independently initialized two-hidden-layer ReLU MLPs.
At each candidate-search step it samples a local neighborhood, obtains a
ranking from each proxy, and derives pairwise labels by majority vote. Each
proxy is mentored only on pairs where it disagrees with the consensus. A
differentiable inner update adjusts soft pair labels against the logged-data
regression objective, after which the proxy is updated by pairwise binary
cross entropy. Candidate ascent uses the mean of the three proxies.

Audited common script defaults are width 2048, 200 proxy epochs, batch size
128, 200 continuous-task search steps, search rate `1e-3`, 10 neighborhood
points, and 128 top logged starts. They are Design-Bench defaults, not public
LLM-DM settings.

The clean-room implementation should use native PyTorch functional parameter
updates instead of adding the old `higher` dependency. It must use isolated
generators and constrained candidate coordinates.

The registered adaptation now implements the paper's one-step inner and outer
updates using `torch.func.functional_call`. Its pairwise comparisons use the
strict-greater-than indicator, and mentoring uses only disagreement pairs as
recorded in the source audit. Each candidate gets independent copies of all
three pretrained models; adapted weights are never shared between candidates.
It uses all visible logged rows for the outer MSE, retaining logged context.
Neighborhood and search inputs always use the fixed target context. Local
Gaussian perturbations are applied in unconstrained coordinates and mapped to
feasible designs; this changes the original neighborhood geometry.

Training uses separate seeded 90/10 splits with Pearson checkpoint selection.
Small/constant-label validation sets use MSE, and fewer than three observations
disable the split. This is explicit small-data behavior, not paper parity.
Unique high-utility logged starts are supplemented by random feasible starts
if there are fewer unique observations than the requested candidate budget.

### Integration validation

The integration was checked with Python 3.12 and PyTorch 2.14 on CPU: 217
pytest cases passed and two data-recipes integration cases were skipped because
the sibling checkout was unavailable. Focused tests include a finite-difference
check of the soft-label meta-gradient, candidate-specific model isolation,
float32/float64 reproducibility, simplex/box neighborhoods, and runner-owned
oracle evaluation. Wheel and source distributions passed package validation.

An eight-seed Branin smoke run (38--45) returned 128 candidates per seed with
width 16, three surrogate epochs, two search steps, and four neighbors. All
runs succeeded and exercised both disagreement updates and nonzero soft-label
changes. These reduced settings validate execution, not publication scores.

## ICT

ICT also begins with three independently initialized MLP proxies. It samples
designs around the current optimization point, rotates each proxy through the
pseudo-labeler role, and co-trains the other two proxies. Each student selects
the pseudo-labeled examples on which the other student has the smallest loss,
then exchanges those examples for its update. A differentiable meta-learning
step reweights pseudo-labeled losses using logged data as supervision.

The public repository reports 200 proxy epochs and, for its TFBind8 command,
100 search steps, 128 neighborhood samples, 8 remembered co-teaching samples,
proxy rate `1e-3`, reweighting rate `3e-1`, and `full` logged-data supervision.
Other tasks require task-specific changes, and no LLM-DM configuration is
published.

The source relies on saved proxy checkpoints, hard-coded CUDA assumptions,
and several fixed shapes. The adapter should train proxies inside `fit`, make
all sizes explicit, remove filesystem checkpoint coupling, and never evaluate
intermediate candidates with the oracle.

## RoMA

RoMA has two distinct stages. Robust pretraining uses adversarial weight
perturbations while fitting a probabilistic surrogate. Candidate-specific
model adaptation then adjusts a temporary surrogate to reduce the candidate
objective's input-gradient norm while retaining score consistency. Candidate
updates optimize a pessimistic probabilistic objective with a local-region
penalty relative to the initial logged solution.

The continuous-task scripts generally use a width-64 probabilistic network,
50--200 warm-up epochs, 500 candidate updates, 20 inner adaptation steps, 128
initial solutions, and task-specific inner/search learning rates. These are
not LLM-DM settings. The original implementation is TensorFlow, so a native
PyTorch implementation is necessarily a framework port plus a multi-fidelity
and constrained-space adaptation.

## Implementation order and promotion criteria

1. Implement shared three-proxy training, local-neighborhood generation,
   pair construction, and differentiable one-step updates.
2. Implement `tri_mentoring` first because it directly exercises every shared
   pairwise and soft-label component.
3. Implement `ict` by adding rotating pseudo-labelers, small-loss exchange,
   and meta-weighting.
4. Implement RoMA separately with a probabilistic surrogate, adversarial
   weight perturbation, and candidate-specific adaptation.

All three result names must retain **adaptation** until an LLM-DM configuration
is published and pinned, preprocessing and optimization semantics match, and
candidate-level or aggregate parity is demonstrated.
