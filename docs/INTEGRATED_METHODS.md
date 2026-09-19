# Integrated non-SPADE methods

The default formal entry point is `llm-design-bench`; `llm-design-bench-llmdm`
is an alias for the same application. `llm-design-bench methods` lists the
27 integrated IDs and protocol dtypes without loading data or training.
Docker's `benchmark` service invokes this same CLI, not a separate runner.

## Implementation ownership

| Group | IDs | Selected implementation |
| --- | --- | --- |
| Reference baselines | `best_logged`, `random_search`, `sobol` | Forward branch |
| Forward/search methods | `offline_mlp`, `standard_ga`, `coms`, `bdi`, `ga_on_gp`, `bo_qei`, `cma_es`, `reinforce`, `mc_dropout`, `roma`, `ict`, `tri_mentoring`, `ltr`, `match_opt`, `pgs` | Forward branch |
| Additional methods | `cbas`, `mins`, `ddom`, `gabo`, `gtg`, `rgd`, `bonet`, `demo`, `root` | Main branch, with GABO's GP migrated as described below |
| Deferred | `spade` | Neither competing implementation selected for the formal workflow |

Sources are main `09991781f82b5f2f5ae1d249ac1e211c4e00c225` and forward branch
`05b35452c6325080705a16dd21009a8f9a536421`. Method-specific provenance and adaptation
disclosures remain in `MethodMetadata`; integration is not a claim that every
method reproduces its paper's original implementation or published numbers.

The four overlapping formal IDs (`best_logged`, `offline_mlp`, `coms`, `bdi`)
have exactly one registered implementation each, from the forward branch.
The old `Unified*` baseline classes, legacy factory and task-based optimizer
classes/runners are removed. There is no duplicate registration or parallel old
experiment workflow. Historical results require their recorded original code,
not a compatibility optimizer selected by the current runtime.

Main's SPADE file and raw registry entry remain available for Python research.
Formal plan freezing, loading, and execution reject `method_id="spade"`, including
when a different `run_id` is supplied. The forward branch's SPADE remains in its
original checkout/history. No SPADE result or frozen artifact is modified.

## One experiment protocol, method-specific internals

All 27 methods receive only the shared visible data through `OfflineProblem` and
return unevaluated candidates through `MethodResult`. They cannot access the
evaluator's oracle or hidden/reference utilities. The evaluator attaches the
target fidelity and saves candidates before evaluation.

The agreed protocol is unchanged: `utility=-loss`, within-scale utility 0–40%
split, target 1B/19500 steps, K=128, pilot seed 0, formal seeds 38–45, and fixed-1B
as a subset of the same visible manifest. See [Formal Workflow](INTEGRATION.md).
`bdi`, `ga_on_gp`, and `bo_qei` use float64; all other methods, including GABO,
use float32. Mixed precision is not enabled.

Shared data does not require identical model preprocessing. The nine additional
methods retain main's `PreparedFitThenProposeMethod`: their validation partition
is taken only from visible rows, with transformations fitted on its training
partition. No hidden high-utility data enters it. The forward methods retain
their own visible-only preprocessing rather than acquiring a new holdout split.
The complete constructor configuration, including inherited defaults, is frozen
and saved with every run; learned model state is not treated as configuration.

## GPyTorch backend

Runtime pins are `gpytorch==1.15.2` and `linear_operator==0.6.1` in both package
dependencies and the lockfile.

- GA on GP and BO-qEI retain the forward branch's learned exact RBF GP.
- GABO shares the same exact model/settings but keeps its own **fixed** GP:
  zero mean on standardized targets, unit RBF output variance, median-distance
  scalar lengthscale, ridge 1e-3, no input standardization, and no hyperparameter
  optimization. It retains latent-function uncertainty (not observation noise),
  the 1e-8 variance floor, and analytic EI.
- BDI remains a finite RBF kernel-ridge adaptation, not a GP posterior model.

GABO tests compare float32/float64 posterior means, uncertainty, query gradients,
and EI against the previous closed-form implementation, including repeated inputs
and constant targets. Numerical agreement does not promise bitwise identical
complete stochastic trajectories across libraries/devices.

## Before starting new formal experiments

Code integration does not choose the methods' final training budgets. **After
merge**, each method owner provides/approves explicit settings; `{}` in a methods file
explicitly freezes current defaults, it is not an implicit shared-budget policy.
Do not reuse the historical two-epoch/K=8 integration preset as a formal plan.

After the release code and budgets are fixed: prepare/reuse a verified shared
data manifest, create a **new** plan, run full-budget seed-0 pilots, then run
seeds 38–45. The frozen source identity prevents attaching old pilots to new code.
Results from different plans/splits must not be concatenated into one ranking.
Old archives remain unchanged and identifiable by their original source/plan.

The unified entry point and Docker services are code-integration changes, not a
new formal training or real-oracle evaluation campaign. Budget approval, new-plan
freezing, full-budget pilots and deferred SPADE selection are not prerequisites
for the code merge.
