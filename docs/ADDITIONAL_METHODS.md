# Additional PyTorch methods

All ten methods are callable through the generic Python API
`make_method(id, **settings)`. Nine are selected for the integrated formal CLI;
SPADE remains deferred and is rejected by formal freeze/load/run. They operate
on `OfflineProblem`: logged designs, utility,
fidelity context, and domain constraints. The evaluator is never passed into
training or generation. All final candidates have the requested count, device,
dtype, and valid physical coordinates.

## Implementation status

These are independent, compact **continuous PyTorch adaptations**. Each
implements the components listed below; none is marked `faithful_pytorch_port`
or `parity_validated`. The upstream programs use different architectures,
normalization, dataset wrappers, and sometimes online evaluation protocols.
Passing local equation and execution checks does not establish the original
papers' scores, distributional equivalence, or checkpoint compatibility.

The registry metadata records a 40-character inspected upstream revision for
each method. Every benchmark exports that revision and the actual resolved
constructor settings, including defaults. These source pins identify the
reference used for implementation; the source repositories are not imported
at runtime. TensorFlow, JAX, Design-Bench, Lightning and BoTorch are not needed
by these integrations.

| ID | Implemented procedure | Material adaptation |
| --- | --- | --- |
| `cbas` | Gaussian conditional VAE, bootstrapped probabilistic ensemble, nondecreasing quantile threshold, prior/adaptive decoder density ratio, survival-probability weighted VAE fitting | Continuous Gaussian decoder; density ratio uses the same latent draw as the authors' sampling estimator, with log-ratio clipping at ±20 |
| `mins` | Utility/context conditional GAN, weighted discriminator and generator losses, mismatched-pair negatives, target-utility search | Compact MLPs; finite target grid ranked by ensemble lower confidence bound and discriminator plausibility |
| `ddom` | Reweighted conditional denoising, label dropout with an explicit presence bit, classifier-free guidance | Cosine discrete VP schedule and complete DDIM sampler replace continuous-time SDE sampling; exponential utility weights replace smoothed histogram weights |
| `gabo` | Conditional VAE latent representation, Wasserstein source critic, adaptive dual coefficient, latent GP and expected-improvement search | Compact VAE; finite-grid dual approximation; exact scalar RBF GP with median lengthscale and sequential analytic EI replaces BoTorch batch qEI |
| `gtg` | Nearby improving trajectories, mean-return conditioned diffusion, classifier-free guidance, anchored starting design | Within-fidelity neighbor walks; flattened trajectory MLP replaces temporal U-Net; final trajectory design is returned |
| `rgd` | Conditional diffusion, probabilistic proxy, probability-flow density estimates, clipped reverse-KL proxy refinement, robust gradient guidance | Seeded Hutchinson trace and fixed midpoint integration stopping at t=.98; Gaussian ensemble; normalized LCB gradients on denoised samples |
| `bonet` | Sorted trajectories, causal transformer, Gaussian next-design likelihood, regret-to-go conditioning and autoregressive rollout | Within-fidelity bootstrap-sorted trajectories; trained proxy updates regret so rollout stays strictly offline |
| `demo` | Conditional source diffusion, surrogate-ascent pseudo-targets, target diffusion fine-tuning, partial noising and target-directed denoising | Pseudo-targets are generated in-process; cosine/DDIM editing replaces the upstream VP-SDE/Heun solver |
| `root` | Low/high utility strata, within-fidelity pairing, noisy Brownian bridge, displacement regression, analytic stochastic reverse transitions | Compact time-conditioned MLP; linear bridge schedule and quantile pairing for continuous tasks |
| `spade` | Scalar conditional diffusion for p(y given x,context), moment/rank calibration, leave-one-out kNN support regularization, support-adjusted LCB and evolutionary search | Cosine schedule, compact denoiser, exact Torch kNN and seeded population search; not the unreleased paper-specific preprocessing |

GABO's public implementation uses a VAE representation plus an adversarial
source critic; the catalog now reflects that boundary. SPADE's public release
provides a generic implementation but explicitly does not include the complete
paper-specific benchmark preprocessing and configurations.

## Shared numerical choices

- Train/validation splitting uses `split_seed`. All fitted scalers use training
  rows only. Utility remains a maximization score.
- Box designs use unit-box coordinates; simplex designs use additive log-ratio
  coordinates. Additional methods standardize these encoded designs using
  training means and a minimum scale of 0.05.
- Physical candidates are decoded before evaluation. Box values are clipped to
  bounds and simplex values use softmax decoding. No oracle score ranks
  candidates inside a method.
- New diffusion models use a cosine variance-preserving schedule (100 steps by
  default), epsilon prediction, and a DDIM solver that finishes at clean data.
  The clean prediction is bounded to ±8 standardized units before decoding.
- Model initializers and all random draws are seeded. CPU float32 and float64
  replay are tested. GPU execution is exposed through `--device cuda`; the
  supplied Docker reproduction services and current validation use CPU.
- Full resolved configurations, training diagnostics and adaptations are stored
  per run, rather than only storing user overrides.

## Running selected methods

Use the single formal workflow documented in [Reproducing Results](REPRODUCING.md):
`llm-design-bench methods`, then `prepare`, `freeze`, and `run`. The explicit
`llm-design-bench-llmdm` alias calls the same application. There is no parallel
publication CLI or `--all-methods` experiment preset.

After merge, each owner approves method-specific budgets and provides entries
in the methods file: `method_id`, optional `run_id`, and explicit `kwargs`.
Freeze exports complete constructor settings, including inherited defaults;
it is not a global `--epochs` override applied indiscriminately to all methods.
BONET's rollout length, diffusion sampling, MINs' target grid and surrogate
training have separate method-specific controls. An empty `kwargs` mapping
explicitly chooses current defaults, not an approved experiment-wide budget.

For development, methods can still be run directly on a Python `OfflineProblem`.
Such calls, including raw SPADE research calls, do not create a valid frozen
formal experiment or waive its pilot/coverage requirements.

## Validation and reproduction

`tests/test_additional_methods.py` checks every method on simplex and box data,
with fidelity context and without it, in float32 and float64. It checks exact
same-seed replay, preservation of the input tensors and ambient RNG, valid
candidate counts, and finite JSON diagnostics. Component checks cover CbAS
density ratios, diffusion forward/reverse identities, causal masking,
within-fidelity trajectories, bridge moments, Gaussian probability-flow
likelihood, reverse-KL gradients, GP acquisition and support conservatism.

The container `smoke` service uses invented data and small budgets to exercise
the 27-method workflow. Its short training does not establish optimization
quality, real-data pilot readiness, or paper-table parity. See
[Docker](DOCKER.md) for its precise scope.

The historical 14-method integration and publication snapshots remain unchanged
under `reference_results/`; their old commands require their original recorded
commit/environment. Do not combine those numbers with a new 27-method ranking.
