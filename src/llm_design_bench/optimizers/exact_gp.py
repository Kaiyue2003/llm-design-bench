from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import gpytorch
import linear_operator
import torch

from llm_design_bench.problem import OfflineProblem, RunContext


class _ExactRBFModel(gpytorch.models.ExactGP):
    """Zero-mean, ARD RBF model with explicit log-space parameters."""

    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=train_inputs.shape[1],
                lengthscale_constraint=gpytorch.constraints.Positive(
                    transform=torch.exp,
                    inv_transform=torch.log,
                ),
            ),
            outputscale_constraint=gpytorch.constraints.Positive(
                transform=torch.exp,
                inv_transform=torch.log,
            ),
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(inputs),
            self.covar_module(inputs),
        )


@contextmanager
def _exact_settings() -> Iterator[None]:
    """Keep this small-data backend exact, regardless of ambient settings.

    The likelihood already includes the configured training jitter. Disable
    automatic extra jitter so a failed factorization is reported rather than
    silently changing the fitted covariance. Candidate posterior sampling has
    its own separately reported jitter in ``stable_posterior_cholesky``.
    """

    with (
        gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ),
        gpytorch.settings.fast_pred_var(False),
        gpytorch.settings.fast_pred_samples(False),
        gpytorch.settings.prior_mode(False),
        gpytorch.settings.skip_posterior_variances(False),
        gpytorch.settings.detach_test_caches(True),
        gpytorch.settings.max_cholesky_size(2**31 - 1),
        gpytorch.settings.cholesky_jitter(
            float_value=0.0,
            double_value=0.0,
            half_value=0.0,
        ),
        gpytorch.settings.cholesky_max_tries(1),
    ):
        yield


@dataclass(frozen=True)
class ExactRBFGaussianProcess:
    """Fitted GPyTorch GP with frozen parameters and differentiable candidates.

    ``noise`` is the learned observation variance, excluding ``jitter``.
    The likelihood includes both terms to preserve the training covariance;
    prediction methods return the latent function posterior without either
    diagonal term added to candidate uncertainty.
    """

    model: _ExactRBFModel
    likelihood: gpytorch.likelihoods.GaussianLikelihood
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    utility_mean: torch.Tensor
    utility_std: torch.Tensor
    lengthscale: torch.Tensor
    output_scale: torch.Tensor
    noise: torch.Tensor
    final_negative_log_likelihood: float
    jitter: float

    def posterior_mean_standardized(
        self,
        problem: OfflineProblem,
        designs: torch.Tensor,
    ) -> torch.Tensor:
        test_inputs = self._normalized_target_inputs(problem, designs)
        with _exact_settings(), gpytorch.settings.skip_posterior_variances(True):
            return self.model(test_inputs).mean

    def posterior_standardized(
        self,
        problem: OfflineProblem,
        designs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        test_inputs = self._normalized_target_inputs(problem, designs)
        with _exact_settings():
            posterior = self.model(test_inputs)
            covariance = posterior.covariance_matrix
            covariance = 0.5 * (covariance + covariance.transpose(0, 1))
            return posterior.mean, covariance

    def training_summary(self) -> dict[str, float | int | list[float] | str]:
        return {
            "surrogate": "exact_rbf_gaussian_process",
            "backend": "gpytorch",
            "gpytorch_version": gpytorch.__version__,
            "linear_operator_version": linear_operator.__version__,
            "parameterization": "log_exp_with_clamped_bounds",
            "inference": "dense_exact_cholesky",
            "posterior": "latent_function",
            "training_jitter": self.jitter,
            "train_samples": len(self.train_inputs),
            "feature_dim": int(self.train_inputs.shape[1]),
            "lengthscale": self.lengthscale.detach().cpu().tolist(),
            "output_scale": float(self.output_scale.detach().cpu()),
            "noise": float(self.noise.detach().cpu()),
            "final_negative_log_likelihood": self.final_negative_log_likelihood,
            "utility_mean": float(self.utility_mean.detach().cpu()),
            "utility_std": float(self.utility_std.detach().cpu()),
        }

    def _normalized_target_inputs(
        self,
        problem: OfflineProblem,
        designs: torch.Tensor,
    ) -> torch.Tensor:
        features = problem.features_at_target(designs)
        return (features - self.feature_mean) / self.feature_std


def fit_exact_rbf_gp(
    problem: OfflineProblem,
    context: RunContext,
    *,
    training_steps: int,
    learning_rate: float,
    initial_lengthscale: float,
    initial_output_scale: float,
    initial_noise: float,
    minimum_std: float,
    jitter: float,
) -> ExactRBFGaussianProcess:
    """Fit a GPyTorch ExactGP using the original total-NLL Adam budget.

    Standardization uses only the visible offline data. Exp/log constraints
    preserve the previous log-parameter optimization coordinates and bounds;
    multiplying GPyTorch's per-observation MLL by the sample count preserves
    the total negative marginal log likelihood objective. Zero training steps
    are supported here for fixed-hyperparameter numerical validation.
    """

    features = problem.train_features
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0, unbiased=False).clamp_min(minimum_std)
    train_inputs = ((features - feature_mean) / feature_std).detach()
    train_targets, utility_mean, utility_std = problem.standardized_utility(minimum_std)
    train_targets = train_targets.detach()

    # GaussianLikelihood adds this variance at training observations. Using
    # exp(raw_noise) + jitter keeps learned noise separate from stabilization.
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(
            torch.tensor(jitter, device=context.device, dtype=context.dtype),
            transform=torch.exp,
            inv_transform=torch.log,
        ),
    ).to(device=context.device, dtype=context.dtype)
    model = _ExactRBFModel(train_inputs, train_targets, likelihood).to(
        device=context.device,
        dtype=context.dtype,
    )
    log_lengthscale = model.covar_module.base_kernel.raw_lengthscale
    log_output_scale = model.covar_module.raw_outputscale
    log_noise = likelihood.noise_covar.raw_noise
    with torch.no_grad():
        # GPyTorch creates constraint bounds in the default dtype before .to();
        # restore the configured value after conversion for float64 runs.
        likelihood.noise_covar.raw_noise_constraint.lower_bound.fill_(jitter)
        log_lengthscale.fill_(math.log(initial_lengthscale))
        log_output_scale.fill_(math.log(initial_output_scale))
        log_noise.fill_(math.log(initial_noise))

    optimizer = torch.optim.Adam(
        [log_lengthscale, log_output_scale, log_noise],
        lr=learning_rate,
    )
    marginal_log_likelihood = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood,
        model,
    )
    model.train()
    likelihood.train()
    with _exact_settings():
        for _ in range(training_steps):
            optimizer.zero_grad(set_to_none=True)
            negative_log_likelihood = (
                -marginal_log_likelihood(
                    model(train_inputs),
                    train_targets,
                )
                * problem.sample_count
            )
            negative_log_likelihood.backward()
            optimizer.step()
            with torch.no_grad():
                log_lengthscale.clamp_(math.log(1e-3), math.log(1e3))
                log_output_scale.clamp_(math.log(1e-4), math.log(1e4))
                log_noise.clamp_(math.log(1e-8), math.log(1.0))

        with torch.no_grad():
            final_nll = (
                -marginal_log_likelihood(
                    model(train_inputs),
                    train_targets,
                )
                * problem.sample_count
            )
            lengthscale = log_lengthscale.exp().reshape(-1).detach()
            output_scale = log_output_scale.exp().detach()
            noise = log_noise.exp().squeeze().detach()

    model.eval()
    likelihood.eval()
    model.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)

    return ExactRBFGaussianProcess(
        model=model,
        likelihood=likelihood,
        train_inputs=train_inputs.detach(),
        train_targets=train_targets.detach(),
        feature_mean=feature_mean.detach(),
        feature_std=feature_std.detach(),
        utility_mean=utility_mean.detach(),
        utility_std=utility_std.detach(),
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise=noise,
        final_negative_log_likelihood=float(final_nll.cpu()),
        jitter=jitter,
    )


def stable_posterior_cholesky(
    covariance: torch.Tensor,
    *,
    jitter: float,
    maximum_attempts: int = 6,
) -> tuple[torch.Tensor, float]:
    """Factor a posterior covariance, increasing diagonal jitter if needed."""

    identity = torch.eye(
        len(covariance),
        device=covariance.device,
        dtype=covariance.dtype,
    )
    current_jitter = jitter
    for _ in range(maximum_attempts):
        factor, info = torch.linalg.cholesky_ex(covariance + current_jitter * identity)
        if int(info.max().detach().cpu()) == 0:
            return factor, current_jitter
        current_jitter *= 10.0
    raise RuntimeError("GP posterior covariance is not positive definite")
