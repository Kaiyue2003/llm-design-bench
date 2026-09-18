# Method Notes

This page describes the current registered implementations. The formal LLM-DM
roster is the 27 non-SPADE methods in [Integrated Methods](INTEGRATED_METHODS.md).
Use `llm-design-bench methods` to list them. Historical task-based optimizer
classes and their experiment runners are not the current runtime.

## Shared contract, different method internals

Every method receives `OfflineProblem` and `RunContext`, then returns exactly K
unevaluated physical-coordinate candidates in `MethodResult`. Only the evaluator
receives the task oracle, hidden/reference utilities and target-fidelity
evaluation machinery. Utility is always maximized.

The integrated methods share this boundary, **not one compulsory preprocessing
pipeline**. Forward methods retain their own visible-only feature/utility
normalization and candidate search. The nine additional methods retain the
`PreparedFitThenProposeMethod` pipeline: a seeded partition of visible rows and
train-only fitted transforms, with simplex log-ratio or unit-box model features.
That validation partition is not the hidden high-utility benchmark region.

The four overlapping IDs (`best_logged`, `offline_mlp`, `coms`, `bdi`) select
the forward branch implementations. No parallel old baseline implementation is
installed as an alternative formal workflow.

## Reference methods

- `best_logged`: selects high-utility unique visible designs and repeats rows if
  needed to return K candidates. It does not query the oracle to select them.
- `random_search`: samples valid designs from the declared design space.
- `sobol`: constructs a seeded low-discrepancy candidate batch in that space.

The scalar evaluator reference `D(best)` and the registered `best_logged`
candidate-producing method are distinct concepts. Repeated deterministic runs
satisfy the protocol's coverage bookkeeping, not eight independent stochastic
discoveries.

## Offline MLP and gradient ascent

`offline_mlp` fits a two-hidden-layer **ReLU** MLP to standardized visible
features and utilities. Its gradient search uses the frozen surrogate and target
context; simplex and box constraints are handled through the design-space maps.
`standard_ga` is another gradient-ascent baseline with its own model/search
configuration. Here GA means **gradient ascent**, not a genetic algorithm.

Neither baseline uses the additional methods' automatic train/validation
preparation. Constructor defaults and adaptation metadata, rather than the
historical main-branch MLP description, define the selected implementation.

## COMs

`coms` fits a compact conservative MLP with adversarial high-prediction designs.
Its loss penalizes overestimation relative to logged designs, with a learned
non-negative multiplier. Final candidates are optimized against the trained
surrogate at fixed target context.

This is a **lightweight PyTorch adaptation**, not a byte-for-byte port of
[design-baselines](https://github.com/brandontrabucco/design-baselines). Metadata
records its TensorFlow origin, the PyTorch implementation, fidelity conditioning
and generic simplex/box support. Those disclosures do not establish paper-table
or checkpoint equivalence.

## BDI

`bdi` uses standardized visible design/context features and RBF kernel ridge
regression. Candidate optimization combines forward utility maximization and
backward distillation from optimistic candidate support points to logged labels.
The default lengthscale uses non-zero pairwise feature distances.

This is a **BDI adaptation** of [GGchen1997/BDI](https://github.com/GGchen1997/BDI).
Finite RBF regression replaces the original infinite-width NTK/JAX/Neural Tangents
runtime. It is not a Gaussian-process posterior and does not use the GPyTorch
backend. Its registered constructor is the forward implementation; historical
`max_support_points` arguments are not part of this API.

## GP and other forward/search methods

| Method ID | Selected procedure |
| --- | --- |
| `ga_on_gp` | Gradient ascent on a learned exact RBF GP |
| `bo_qei` | Offline candidate acquisition using the fitted GP and qEI |
| `cma_es` | Covariance-adaptation search on a learned surrogate |
| `reinforce` | Policy-gradient candidate search on a learned surrogate |
| `mc_dropout` | Dropout-based predictive uncertainty for candidate search |
| `roma` | Robust model adaptation during candidate optimization |
| `ict` | Importance-aware co-teaching with differentiable reweighting |
| `tri_mentoring` | Ensemble mentoring and candidate-local adaptation |
| `ltr` | Learning-to-rank surrogate and target-context search |
| `match_opt` | Gradient-matching surrogate training and candidate search |
| `pgs` | Visible-data transition construction and policy-guided search |

The GP methods fit only logged observations; qEI does not authorize online oracle
queries or sequential acquisition of real training results. `exact_gp.py` uses
GPyTorch for GA on GP and BO-qEI. GABO also uses its exact model/settings through
its latent-GP adapter, while preserving its own fixed-kernel and analytic-EI
semantics. See [Integrated Methods](INTEGRATED_METHODS.md) for backend details.

These methods retain their adaptation disclosures. A working PyTorch interface
and component tests are not a claim that all original papers are reproduced.

## Additional methods and deferred SPADE

CbAS, MINs, DDOM, GABO, GTG, RGD, BONET, DEMO and ROOT are the nine additional
formal integrations. [Additional Methods](ADDITIONAL_METHODS.md) documents
their algorithms, source revisions and material substitutions. The catalog uses
`implemented_adaptation`; upstream parity requires separate evidence.

The main-branch SPADE implementation remains callable through the generic Python
registry for research and tests. It is **not selected for the formal workflow**:
freezing, loading and running a formal SPADE plan are rejected. Resolving it is
separate from merging the 27-method code.

## Metrics and experimental budgets

For candidate statistic `u` and frozen full-logged reference utilities `D`:

```text
refnorm score = (u - min(D)) / (max(D) - min(D))
```

Best, median and mean utility are normalized independently. Values are not clipped;
above 1 means above logged best, not proof of a global optimum. Diversity is mean
pairwise distance; novelty is mean nearest-visible-design distance. Box coordinates
are normalized by the bounds for these diagnostics.

Final training/search budgets are decided by method owners **after the code
merge**, frozen with full constructor defaults, and checked with seed-0 pilots
before formal seeds 38–45. A common K=128 output budget does not imply identical
training budgets or permit selection using pilot oracle scores.
