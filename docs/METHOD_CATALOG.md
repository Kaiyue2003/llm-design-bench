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
- `planned`: not implemented; its primary code/config source must be audited
  before implementation begins.

## Paper method matrix

| ID | Method | Family | Current state | PyTorch integration |
| --- | --- | --- | --- | --- |
| `cma_es` | CMA-ES adaptation | Standard | integrated adaptation | Full-covariance PyTorch CMA-ES over a frozen probabilistic ensemble |
| `reinforce` | REINFORCE adaptation | Standard | integrated adaptation | Fixed-variance Gaussian policy with frozen-surrogate rewards |
| `bo_qei` | BO-qEI | Standard | planned | BoTorch/GPyTorch qEI |
| `standard_ga` | Standard GA adaptation | Forward | integrated adaptation | Probabilistic MLP mean + plain gradient ascent under a distinct result ID |
| `ga_on_gp` | GA on GP | Forward | planned | GPyTorch predictive mean + design ascent |
| `mc_dropout` | MC-Dropout | Forward | planned | Dropout surrogate + Monte Carlo risk-aware acquisition |
| `coms` | COMs | Forward | integrated adaptation | Unified compact PyTorch adaptation; retain adaptation label until parity audit passes |
| `roma` | RoMA | Forward | planned | Robust local smoothness/model-adaptation port |
| `ict` | ICT | Forward | planned | Importance-aware co-teaching and pseudo-label exchange |
| `tri_mentoring` | Tri-Mentoring | Forward | planned | Three-surrogate mentoring with pairwise filtering |
| `bdi` | BDI | Forward | integrated adaptation | Unified RBF version as `BDI adaptation`; faithful port requires a separate result ID |
| `ltr` | LTR | Forward | planned | Learning-to-rank surrogate and paper-aligned search |
| `match_opt` | MATCH-OPT | Forward | planned | Surrogate/data-support gradient matching |
| `pgs` | PGS | Forward | planned | Policy-guided, perturbation-smoothed gradient search |
| `cbas` | CbAS | Inverse | planned | Conditional adaptive sampling |
| `mins` | MINs | Inverse | planned | Conditional GAN inverse model |
| `ddom` | DDOM | Inverse | planned | Score-conditioned design diffusion |
| `gabo` | GABO | Inverse | planned | GAN latent-space BO with source critic |
| `gtg` | GTG | Inverse | planned | Guided diffusion over synthetic trajectories |
| `rgd` | RGD | Inverse | planned | Official PyTorch core + benchmark adapter |
| `bonet` | BONET | Inverse | planned | Autoregressive generative pretraining |
| `demo` | DEMO | Inverse | planned | Design-distribution diffusion/editing |
| `root` | ROOT | Inverse | planned | Probabilistic-bridge distribution translation |
| `spade` | SPADE | Forward (proposed) | planned official adapter | Official PyTorch core + LLM-DM/simplex/context adapter |

The counts are 3 standard methods, 11 forward-surrogate baselines, 9 inverse
generative baselines, and SPADE, for 24 paper methods total.

## Existing project controls

The unified registry currently contains the controls `best_logged`,
`random_search`, `sobol`, and `offline_mlp`. Standard GA, CMA-ES, REINFORCE,
COMs, and BDI are separately registered and explicitly labeled adaptations.
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
- BDI currently means the RBF-kernel adaptation. The cited source is the
  [official BDI repository](https://github.com/GGchen1997/BDI), whose
  JAX/Neural Tangents mechanism has not yet been faithfully ported.
- RGD has a verified
  [official PyTorch repository](https://github.com/GGchen1997/RGD), but no
  project adapter yet.
- SPADE should be reported as **official SPADE core + llm-design-bench
  adapter**. Its [official repository](https://github.com/HarryYoung2018/spade)
  explicitly omits LLM-DM preprocessing, final task configurations, and the
  paper's full evaluation scripts.

For every other planned method, locating the authoritative code repository,
license, version/commit, original framework, and final paper configuration is
a blocking checklist item. The catalog intentionally leaves an unverified code
URL empty rather than guessing one.

## Recommended implementation waves

1. **Low-risk controls and classical methods:** Standard GA, CMA-ES, and
   REINFORCE are integrated adaptations; BO-qEI, GA on GP, and MC-Dropout
   remain next in this wave.
2. **Forward offline methods:** RoMA, ICT, Tri-Mentoring, LTR, MATCH-OPT, and
   PGS; the unified COMs/BDI adaptations are now available as comparison
   points.
3. **SPADE:** integrate the verified official PyTorch core early enough to
   establish the target paper method, but do not claim exact table reproduction.
4. **Inverse generative methods:** CbAS, MINs, DDOM, GABO, GTG, RGD, BONET,
   DEMO, and ROOT.

Each method enters the final eight-seed table only after interface, no-oracle,
reproducibility, design-space validity, smoke-run, and provenance tests pass.
