# GP and Uncertainty Baseline Source Audit

This audit covers the `bo_qei`, `ga_on_gp`, and `mc_dropout` method IDs. It
records the implementation boundary needed to compare them without claiming
an unavailable reproduction of the SPADE LLM-DM table.

## Sources and reproducibility boundary

- Paper: [SPADE, arXiv:2605.11246](https://arxiv.org/abs/2605.11246).
- qEI formulation: [BoTorch acquisition documentation](https://botorch.org/docs/v0.17.0/acquisition/),
  version 0.17.0.
- GP reference: [ROOT Gaussian Process code](https://github.com/cuong-dm/ROOT/tree/main/gaussian_process),
  audited at commit `d23f14fe30d53f1fc4423ce006056672d0353906`.
- Dropout uncertainty: [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142).
- SPADE code: [HarryYoung2018/spade](https://github.com/HarryYoung2018/spade).

The SPADE paper names these baseline mechanisms and evaluates 128 returned
candidates over eight seeds. Its public repository explicitly omits final
task-specific configurations, preprocessing, and full evaluation scripts.
The public ROOT repository contains an RBF Gaussian Process and posterior-mean
code, but not the exact BO-qEI, GA-on-GP, or MC-Dropout LLM-DM baseline
configuration. These method IDs must therefore retain the **adaptation** label.

## Shared exact GP

`bo_qei` and `ga_on_gp` share a GPyTorch exact Gaussian Process (GPyTorch
1.15.2 and linear_operator 0.6.1). The immutable `llmdm_forward_v1` release
used a handwritten PyTorch GP; its pinned checkout remains historical and must
not be mixed with the new `llmdm_forward_gpytorch_v2` experiment.

- design coordinates and optional fidelity context are standardized together;
- maximization utility is standardized once;
- an ARD RBF kernel is used with a zero prior mean;
- length scales, output scale, and observation noise are learned by exact
  marginal likelihood;
- Cholesky solves are used for posterior mean and covariance;
- posterior factorization increases diagonal jitter only when required.

The backend uses GPyTorch `ExactGP`, `ZeroMean`, an ARD `RBFKernel` wrapped in
`ScaleKernel`, `GaussianLikelihood`, and `ExactMarginalLogLikelihood`.
See the [official regression tutorial](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html).
The optimizer retains the original log-space hyperparameters, clamps and total
negative marginal log likelihood. Training covariance includes observation
noise plus explicit numerical jitter; candidate acquisition uses the latent
function posterior, not noisy observation predictions. Dense exact solves and
joint covariance are retained, without fast predictive variance approximations.

qEI candidate optimization remains our PyTorch adaptation, not a BoTorch
implementation. Fixed-hyperparameter numerical checks do not imply identical
optimization trajectories or paper-exact replication. BDI's separate RBF
kernel-ridge adaptation is not a Gaussian Process and is unchanged.

## BO-qEI adaptation

The acquisition is the joint Monte Carlo q-Expected Improvement

```text
E[max(max(f(x_1), ..., f(x_K)) - best_logged_utility, 0)].
```

It uses reparameterized samples from the joint GP posterior. One seeded set of
base normal samples is held fixed during candidate optimization, so the
objective remains reproducible and its gradients do not change solely because
of resampling. The full candidate batch is optimized together through the
common simplex or box mapping.

Differences from a possible paper implementation include the GP configuration,
gradient optimizer, initialization mix, stopping rule, and shared defaults.
The public paper artifacts do not pin these choices for LLM-DM.

## GA-on-GP adaptation

This method maximizes the exact GP posterior mean using plain gradient ascent.
Candidate starts mix top unique logged designs with seeded random feasible
designs. Optimization occurs in unconstrained coordinates, followed by the
same simplex or box mapping used by all registered methods.

The retained mechanism is posterior-mean ascent on an RBF GP. Context
conditioning, constrained coordinates, initialization, and shared defaults
are benchmark adaptations.

## MC-Dropout adaptation

This method trains a standardized MSE surrogate with dropout in every hidden
layer. Dropout remains active for repeated inference passes. Candidate search
maximizes

```text
MC mean - uncertainty_weight * MC standard deviation.
```

This is a lower-confidence-bound objective for maximization. Dropout masks,
minibatch orders, model initialization, and candidate starts all derive from
the method seed without mutating the process-wide Torch random state.

The uncertainty weight, architecture, optimizer, and stopping rule are shared
benchmark defaults because final LLM-DM baseline settings are not public.

## Promotion criteria

An adaptation label can be removed only after the exact source/configuration
is published and pinned, preprocessing and candidate generation match it, and
candidate-level or aggregate parity is demonstrated. Until then, publication
tables must include the adaptation display names and recorded source metadata.
