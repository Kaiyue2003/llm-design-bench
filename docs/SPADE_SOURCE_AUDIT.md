# SPADE source audit and adapter gate

Audited 2026-09-09. This is a pre-integration audit, not a registered method,
training run, or reproduction of a published result. The registry still has
18 methods. Complete the agreed method roster before the full one-seed trials
and subsequent eight-seed experiments.

## Source and reuse boundary

- Paper: [arXiv:2605.11246v2](https://arxiv.org/html/2605.11246v2).
- Official repository: <https://github.com/HarryYoung2018/spade>.
- Pinned revision: `586151bbb56e246f93ca97ce33f79887a13161bd`.
- [MIT license at that revision](https://github.com/HarryYoung2018/spade/blob/586151bbb56e246f93ca97ce33f79887a13161bd/LICENSE),
  copyright 2026 SPADE Authors. Retain the complete notice with any copied or
  modified core, and include it in both wheel and source distributions.
- The release declares Python >=3.9 and dependencies on NumPy, PyTorch, and
  scikit-learn. The neural model is PyTorch; kNN is scikit-learn/CPU and the
  evolutionary search is NumPy. It is not an entirely PyTorch runtime.
- The pinned `scripts/reproducibility_status.md` explicitly excludes the full
  benchmark preprocessing/evaluation stack and final per-task configurations.
  A generic NPZ loader is not the paper's LLM-DM adapter.

The source checkout used for inspection is outside this package. No upstream
code is vendored or imported by the benchmark in this audit. At integration,
prefer a small, attributed internal core over an unpinned runtime dependency.
Record every change, especially generator/dtype plumbing, native kNN and search.
The intended label is **SPADE adaptation (official-core-derived)** with
`ImplementationKind.MULTI_FIDELITY_ADAPTATION`, not an unchanged official wrapper
or exact paper reproduction.

## Mechanisms to preserve

These details were inspected in the pinned Python files, not inferred from a
method name. Paths below are relative to the official source repository.

| Component | Source | Retained behavior |
| --- | --- | --- |
| Conditional surrogate | `spade/diffusion.py` | Diffuse scalar utility, condition on design/features; epsilon-prediction MSE, linear beta schedule, SiLU network with time embedding |
| Differentiable calibration | `spade/train.py`, `spade/regularizers.py` | Short DDIM samples retain parameter gradients; mean-to-label MSE plus softplus ranking on sampled strictly ordered pairs |
| Support training loss | `spade/regularizers.py`, `spade/knn.py` | Hinge penalties against neighbor mean plus distance margin, and against a distance-dependent standard-deviation floor |
| Acquisition | `spade/optimize.py` | Maximize MC mean minus beta times MC standard deviation; no task/oracle calls |
| Search | `spade/optimize.py` | High-visible-utility starts, elites, convex crossover, Gaussian mutation and decaying mutation scale |

For the support loss, the code uses `d = log(R_k + 1e-8)`, margin `a*d`,
and floor `a0 + a1*d`. The neighbor mean is the arithmetic mean of logged
standardized utilities. Training queries include the query row itself among
neighbors; this is not leave-one-out kNN. Distances below one give negative
log-distance. Do not silently clamp log-distance to zero or replace it with a
normalized radius: either change would alter the audited mechanism.

The training hinge is `relu(mu - neighbor_mean - a*d)`, not a hard cap on
every prediction. The separate inference support transform is optional and
defaults to **off** in `optimize_spade`, including the NPZ quickstart. Training
support regularization is enabled by the default positive support weight.
Keep these two switches distinct; enabling the inference transform is an
explicit variation, not a prerequisite to calling the training method SPADE.

SPADE is a forward model of utility conditioned on a design, even though its
surrogate uses diffusion. It does not replace the nine inverse-generative
methods being investigated separately.

## Public defaults are not frozen experiment settings

The following are values in `spade/config.py`, not recovered paper configs:

| Group | Public defaults |
| --- | --- |
| Surrogate | diffusion steps 100; hidden width 2048; time dimension 128; beta 0.0001 to 0.02 |
| Training | Adam learning rate 0.001; 100 epochs; batch 64; gradient clip 1 |
| Calibration | weight 1; 32 sampled pairs; temperature multiplier 1; 4 MC draws, 10 DDIM steps, eta 0 |
| Support | weight 1; k=10; a=0.02, a0=0.02, a1=0.005; cached training neighbors |
| Acquisition | LCB beta 0.1; 256 MC draws, 50 DDIM steps, eta 0; chunk size 4096 |
| Evolution | population 128; elite 64; 100 generations; mutation 0.12 down to 0.02; crossover 0.3 |

The source multiplies mutation sigma by 0.98 per generation, bounded by its
minimum; initial population jitter has standard deviation 0.05 in normalized
coordinates. MC chunks group at most 16 draws, but differentiable calibration
retains the computation graphs until backward. Small input dimension does not
make the default hidden width/MC workload cheap. These defaults have not been
timed on Colab or adopted as our formal training budget.

The paper's Appendix F describes task-dependent tuning of regularization
weights, not the final selected LLM-DM values. Its tables report standard error
across eight seeds; retain our per-seed data and distinguish standard deviation
from standard error when comparing reports. Equation 12 uses an M-1 variance
denominator; the public implementation uses `unbiased=False` (denominator M).
Preserve and disclose the source convention rather than asserting numerical
identity between text and implementation.

## Concrete adapter hazards and decisions

1. **Task domain versus observed range.** `Dataset.from_npz` derives box bounds
   from observed coordinate minima/maxima. Those are not a simplex and are not
   necessarily the benchmark's full box. Use only visible rows for statistical
   normalization, but use `problem.design_space` for known domain constraints.
   Do not derive search bounds from hidden examples or restrict them to the
   logged coordinate envelope.
2. **Projection timing.** `optimize_spade(project_fn=...)` projects only the
   historical best point after evolution. The surrogate scores unprojected
   populations, and the returned population is not projected by that callback.
   Our search must project initialization, crossover and mutation outputs
   **before every acquisition/support query**. Preserve valid simplex boundary
   points and inward-representable box endpoints; do not relax evaluator checks.
3. **Fidelity is conditioning, not a decision variable.** Source evolution
   perturbs every feature. Feed `[design, context]` to the model, but evolve only
   design coordinates and append the unchanged target context for every query.
   Multi-scale logs remain training data; final target is 1B / 19500 steps.
   Do not turn changing model size/steps into an optimization advantage.
4. **kNN geometry must be declared.** Initial adapter uses Euclidean distance
   on visible-standardized joint `[design, context]` features, with unit weight
   per standardized feature. This prevents identical mixtures at very different
   fidelities from automatically being zero-distance neighbors. It remains a
   benchmark adaptation, not a recovered paper choice. Constant columns use
   scale one; freeze and report normalization and this distance convention.
   The same geometry serves training support and any optional inference transform.
5. **Exact candidate budget.** `evolutionary_optimize` caps population by N.
   With few rows the default elite count can cover the entire population,
   leaving no children. Maintain population at least K independently of N;
   repeat visible starts with feasible jitter when necessary, and reserve
   offspring slots. Do not simply duplicate the best output to claim K searches.
   Report effective population/elite sizes and distinct-candidate counts.
6. **Final population is not yet scored.** Source evaluates a population and
   then breeds the next one, including in the last generation. Its returned
   population is therefore neither a scored top-K set nor guaranteed to contain
   its historical best. Score the final feasible population plus retained elites
   with the frozen surrogate, and select exactly K by a declared LCB rule.
   This extra scoring cost and fresh MC noise must be counted; oracle scores
   cannot select candidates or checkpoints.
7. **Randomness and precision.** `train_spade` seeds Python, NumPy and global
   Torch RNGs, and can change global deterministic/backend settings. Model
   construction, diffusion draws and ranking pairs use global Torch randomness.
   Source data/noise also hard-code float32. Use the run generator throughout a
   PyTorch adapter, isolate initialization, support the requested dtype/device,
   and test preservation of caller RNG/backend state. Do not silently downcast
   a float64 run or rely on save/restore as a concurrency-safe RNG implementation.
8. **Small data and ties.** The public API does not cap k by N; cached neighbor
   construction raises when k>N (the quickstart separately caps it). Require
   N>=2 and configured k>=2, record effective k=min(k,N), and retain self-neighbor
   semantics. A native Torch distance/top-k helper must be chunked, have stable
   tie handling, and match the source on non-tied examples. Duplicate rows and
   zero radii require finite-value tests; do not silently drop observations.
9. **Degenerate uncertainty transform.** The optional source transform rescales
   centered samples to impose a standard-deviation floor. Identical samples
   remain identical after rescaling, so the floor is not actually achieved.
   For optional transformed LCB, transform moments directly, using
   `mu_adj - beta * max(sigma, floor)`; do not invent variance by rescaling zeros.
   Keep the default inference transform off and record this optional-path fix.
10. **Configuration validation.** Validate finite weights and positive budgets,
    even time-embedding dimension, ordered betas in (0,1), and valid elite and
    crossover settings before training. Initially support eta=0 only: the source
    stochastic skipped-step update uses the single-step alpha. Reject nonzero
    eta explicitly until a separately audited stochastic sampler is available.
    Require at least two MC draws for uncertainty estimation and DDIM steps
    between 2 and the diffusion horizon; never silently run an invalid schedule.

## Checks performed in this audit

The source's three `tests/test_smoke.py` tests passed using the benchmark's
existing CPU environment. They check imports, config construction and NPZ
loading only; they do **not** train a surrogate or validate end-to-end search.
No additional dependency was installed and no source files were modified.

Additional assertion-based probes invoked the public search with a mocked
surrogate, not an oracle or a trained network. Dataset: three two-dimensional
simplex points `(0.2,0.8)`, `(0.5,0.5)`, `(0.8,0.2)`, utilities 0,1,2,
identity normalization, box [0,1]. Settings: population 128, elite 1, two
generations, default seed 0. Mock samples were two draws `x[:,0]` and
`x[:,0]+0.1`; the final projection callback divided coordinates by their sum.

- Returned population had 3 rows, despite requested population 128.
- Both population-scoring calls contained points whose coordinates did not sum
  to one; the projection callback ran once, for the final best point only.
- Returned offspring differed from the last population passed for scoring.
- Separately, four identical utility samples, log-distance zero and floor 0.02
  yielded actual standard deviation zero after `apply_support_transform`.
- `KnnStats` with three rows and k=10 raised `ValueError`.

These are contract/edge-case observations, not performance measurements. No
formal seed, real data-recipes evaluation, CUDA run or reference result was
produced. This audit does not establish end-to-end correctness of SPADE.

Benchmark regression validation after the catalog/documentation changes:
409 tests passed, six skipped (four CUDA checks and two requiring the missing
data-recipes checkout). Ruff passed for the changed catalog test file, and
the diff passed whitespace checks. These counts exclude the three upstream
public-API smoke tests described above.

## Next implementation step and acceptance gate

1. Add an attributed PyTorch scalar-diffusion/calibration/support core with
   source provenance and packaged MIT notice. Test reference formulas,
   calibration parameter gradients, kNN behavior, RNG isolation and dtype.
2. Add constrained design-only EA and the `OfflineBBOMethod` adapter. Return
   `MethodResult` with exactly K feasible candidates, loss histories, effective
   budgets, support geometry, inference-transform setting and source revision.
   Keep all target context fixed and all fitting statistics visible-only.
3. Register only after simplex/box, multi/fixed-fidelity, constant-label,
   duplicate/small-data, reproducibility and no-oracle tests pass. Add reduced
   CPU integration smoke runs through the evaluator/report writer, then update
   catalog status and method count. CUDA/data-recipes checks remain separate.

Do not freeze the formal configuration or run eight seeds as part of this
integration step. Coordinate the inverse-method roster before formal training.
