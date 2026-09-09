# Offline BBO Method Catalog

This catalog is the implementation backlog for the 23 baselines and SPADE in
[arXiv:2605.11246](https://arxiv.org/abs/2605.11246). The machine-readable
inventory is [`method_catalog.json`](method_catalog.json). A catalog entry is
not evidence that the method has been implemented or reproduced.

## Shared integration contract

Every method must subclass `OfflineBBOMethod` (or its `FitThenProposeMethod`
template), receive only an `OfflineProblem`, and return exactly `K` candidates
in a `MethodResult`. The evaluator alone owns the oracle. All task labels are
maximization utilities; LLM cross entropy is converted once with
`utility = -loss` by the task adapter.

For the primary multi-scale LLM-DM experiment, a forward model is conditioned
on model scale and training steps and is optimized at 1B/19,500 steps. An
inverse model must condition its generator on fidelity as well as target
utility. Adding these context variables is a multi-fidelity adaptation and
must be recorded in metadata. The fixed-1B experiment exercises the same
implementation without cross-scale conditioning and serves as an ablation.

The target is a PyTorch-facing implementation. BoTorch and GPyTorch are
acceptable PyTorch ecosystem dependencies. A material replacement of an
original component is labeled an adaptation, not a faithful reproduction.

## Status vocabulary

- `integrated_unified`: available through the strict oracle-separated registry.
- `integrated_adaptation`: available through the strict unified registry but
  replaces a material part of the published method.
- `legacy_adaptation`: runnable only through the legacy path.
- `planned_official_adapter`: verified official PyTorch core exists, but the
  benchmark adapter has not been written.
- `planned`: not implemented; source audit may be complete, but registration
  requires implementation and validation. Any outstanding source/config audit
  must be completed before implementation begins.

## Paper method matrix

| ID | Method | Family | Current state | PyTorch integration |
| --- | --- | --- | --- | --- |
| `cma_es` | CMA-ES adaptation | Standard | integrated adaptation | Full-covariance PyTorch CMA-ES over a frozen probabilistic ensemble |
| `reinforce` | REINFORCE adaptation | Standard | integrated adaptation | Fixed-variance Gaussian policy with frozen-surrogate rewards |
| `bo_qei` | BO-qEI adaptation | Standard | integrated adaptation | Native exact RBF GP + joint Monte Carlo qEI |
| `standard_ga` | Standard GA adaptation | Forward | integrated adaptation | Probabilistic MLP mean + plain gradient ascent under a distinct result ID |
| `ga_on_gp` | GA on GP adaptation | Forward | integrated adaptation | Native exact RBF GP predictive mean + design ascent |
| `mc_dropout` | MC-Dropout adaptation | Forward | integrated adaptation | Dropout surrogate + Monte Carlo lower-confidence-bound search |
| `coms` | COMs | Forward | integrated adaptation | Unified compact PyTorch adaptation; retain adaptation label until parity audit passes |
| `roma` | RoMA adaptation | Forward | integrated adaptation | Native Gaussian proxy, projected adversarial weights, and candidate-local smoothness adaptation |
| `ict` | ICT adaptation | Forward | integrated adaptation | Independent PyTorch rotating co-teaching, functional meta-weighting, and frozen-ensemble search |
| `tri_mentoring` | Tri-Mentoring adaptation | Forward | integrated adaptation | Independent PyTorch three-proxy voting, pairwise mentoring, and adaptive soft labels |
| `bdi` | BDI | Forward | integrated adaptation | Unified RBF version as `BDI adaptation`; faithful port requires a separate result ID |
| `ltr` | LTR adaptation | Forward | integrated adaptation | RaM-ListNet: sampled lists, ranking loss, normalized-output search |
| `match_opt` | MATCH-OPT adaptation | Forward | integrated adaptation | Value regression + line-integral gradient matching on same-fidelity pairs |
| `pgs` | PGS adaptation | Forward | integrated adaptation | Certified replay, horizon-conditioned CQL/SAC, fixed-target projected-gradient rollout |
| `cbas` | CbAS | Inverse | planned | Conditional adaptive sampling |
| `mins` | MINs | Inverse | planned | Conditional GAN inverse model |
| `ddom` | DDOM | Inverse | planned | Score-conditioned design diffusion |
| `gabo` | GABO | Inverse | planned | GAN latent-space BO with source critic |
| `gtg` | GTG | Inverse | planned | Guided diffusion over synthetic trajectories |
| `rgd` | RGD | Inverse | planned | Official PyTorch core + benchmark adapter |
| `bonet` | BONET | Inverse | planned | Autoregressive generative pretraining |
| `demo` | DEMO | Inverse | planned | Design-distribution diffusion/editing |
| `root` | ROOT | Inverse | planned | Probabilistic-bridge distribution translation |
| `spade` | SPADE | Forward (proposed) | source audited; adapter planned | Attributed official-core-derived PyTorch adaptation; constrained search and fixed context |

The counts are 3 standard methods, 11 forward-surrogate baselines, 9 inverse
generative baselines, and SPADE, for 24 paper methods total.

## Existing project controls

The unified registry currently contains the controls `best_logged`,
`random_search`, `sobol`, and `offline_mlp`. Standard GA, CMA-ES, REINFORCE,
BO-qEI, GA on GP, MC-Dropout, Tri-Mentoring, ICT, RoMA, LTR, MATCH-OPT, PGS, COMs, and BDI are separately registered and
explicitly labeled adaptations.
The controls should not be silently renamed to a paper method:

- `offline_mlp` can share components with Standard GA, but a result is called
  Standard GA only after its architecture, preprocessing, initialization, and
  gradient-search configuration match the declared protocol.
- `best_logged` is a deterministic reference and should be separated from
  stochastic method ranks.
- Random Search and Sobol measure whether learned methods beat simple feasible
  exploration of the simplex or box.

## Provenance decisions already made

- COMs currently means a compact native PyTorch adaptation based on
  [`design-baselines`](https://github.com/brandontrabucco/design-baselines).
- Standard GA, CMA-ES, and REINFORCE are PyTorch adaptations audited against
  Design-Baselines commit `785dbcfa58107bfcc426257a1c2e69d7f71c3c27`.
  See the [baseline source audit](BASELINE_SOURCE_AUDIT.md) for retained
  mechanisms, defaults, and deliberate differences.
- BO-qEI, GA on GP, and MC-Dropout are native PyTorch adaptations. The GP
  methods share an exact ARD-RBF surrogate; BO-qEI optimizes joint Monte Carlo
  expected improvement, while GA on GP optimizes posterior mean. MC-Dropout
  optimizes a lower confidence bound estimated with inference-time dropout.
  See the [GP and uncertainty source audit](GP_UNCERTAINTY_BASELINE_AUDIT.md).
- BDI currently means the RBF-kernel adaptation. The cited source is the
  [official BDI repository](https://github.com/GGchen1997/BDI), whose
  JAX/Neural Tangents mechanism has not yet been faithfully ported.
- RGD has a verified
  [official PyTorch repository](https://github.com/GGchen1997/RGD), but no
  project adapter yet.
- SPADE's MIT-licensed official source is pinned to
  `586151bbb56e246f93ca97ce33f79887a13161bd`. The intended result label is
  **SPADE adaptation (official-core-derived)**, not an unchanged wrapper:
  search-domain, target-context, candidate-budget and RNG changes are required.
  Its public release omits final LLM-DM preprocessing/configuration. See the
  [SPADE source audit](SPADE_SOURCE_AUDIT.md) for verified hazards, reuse terms,
  defaults and the acceptance gate. It is not registered yet.
- RoMA, ICT, and Tri-Mentoring now have pinned author/official repositories,
  but none contains an explicit license file at the audited revision. Their
  algorithms are implemented independently from the papers, not by copying
  source (all three integrated as adaptations). See the
  [forward-method source audit](FORWARD_METHOD_SOURCE_AUDIT.md).
- LTR, MATCH-OPT, and PGS now have pinned paper-linked/author repositories.
  All three are independently implemented adaptations.
  No explicit project-wide license file was found at these revisions; use
  independent implementations, not source vendoring.
  See the [ranking and policy source audit](RANKING_POLICY_METHOD_SOURCE_AUDIT.md)
  for algorithms, dependency hazards, multi-fidelity rules, and acceptance tests.

For every other planned method, locating the authoritative code repository,
license, version/commit, original framework, and final paper configuration is
a blocking checklist item. The catalog intentionally leaves an unverified code
URL empty rather than guessing one.

## Recommended implementation waves

1. **Low-risk controls and classical methods:** Standard GA, CMA-ES,
   REINFORCE, BO-qEI, GA on GP, and MC-Dropout are integrated adaptations.
2. **Forward offline methods:** Tri-Mentoring, ICT, RoMA, LTR, MATCH-OPT, and PGS
   are integrated. PGS uses [certified transitions and native CQL/SAC](PGS_TRANSITION_DESIGN.md).
   The unified COMs/BDI
   adaptations are already available as comparison points.
3. **SPADE:** source audit complete; implement the attributed PyTorch core and
   constrained adapter next, without claiming exact table reproduction.
4. **Inverse generative methods:** CbAS, MINs, DDOM, GABO, GTG, RGD, BONET,
   DEMO, and ROOT.

Each method enters the final eight-seed table only after interface, no-oracle,
reproducibility, design-space validity, smoke-run, and provenance tests pass.
