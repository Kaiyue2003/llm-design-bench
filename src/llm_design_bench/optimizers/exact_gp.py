from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from llm_design_bench.problem import OfflineProblem, RunContext


@dataclass(frozen=True)
class ExactRBFGaussianProcess:
    """Small exact RBF GP with standardized inputs and utilities."""

    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    utility_mean: torch.Tensor
    utility_std: torch.Tensor
    lengthscale: torch.Tensor
    output_scale: torch.Tensor
    noise: torch.Tensor
    cholesky: torch.Tensor
    alpha: torch.Tensor
    final_negative_log_likelihood: float
    jitter: float

    def posterior_mean_standardized(
        self,
        problem: OfflineProblem,
        designs: torch.Tensor,
    ) -> torch.Tensor:
        test_inputs = self._normalized_target_inputs(problem, designs)
        cross_covariance = _rbf_kernel(
            self.train_inputs,
            test_inputs,
            self.lengthscale,
            self.output_scale,
        )
        return cross_covariance.transpose(0, 1) @ self.alpha

    def posterior_standardized(
        self,
        problem: OfflineProblem,
        designs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        test_inputs = self._normalized_target_inputs(problem, designs)
        cross_covariance = _rbf_kernel(
            self.train_inputs,
            test_inputs,
            self.lengthscale,
            self.output_scale,
        )
        mean = cross_covariance.transpose(0, 1) @ self.alpha
        projected = torch.linalg.solve_triangular(
            self.cholesky,
            cross_covariance,
            upper=False,
        )
        covariance = _rbf_kernel(
            test_inputs,
            test_inputs,
            self.lengthscale,
            self.output_scale,
        ) - projected.transpose(0, 1) @ projected
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        return mean, covariance

    def training_summary(self) -> dict[str, float | int | list[float] | str]:
        return {
            "surrogate": "exact_rbf_gaussian_process",
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
    """Fit exact GP hyperparameters by marginal likelihood in PyTorch."""

    features = problem.train_features
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0, unbiased=False).clamp_min(minimum_std)
    train_inputs = (features - feature_mean) / feature_std
    train_targets, utility_mean, utility_std = problem.standardized_utility(
        minimum_std
    )
    feature_dim = train_inputs.shape[1]

    log_lengthscale = torch.nn.Parameter(
        torch.full(
            (feature_dim,),
            math.log(initial_lengthscale),
            device=context.device,
            dtype=context.dtype,
        )
    )
    log_output_scale = torch.nn.Parameter(
        torch.tensor(
            math.log(initial_output_scale),
            device=context.device,
            dtype=context.dtype,
        )
    )
    log_noise = torch.nn.Parameter(
        torch.tensor(
            math.log(initial_noise),
            device=context.device,
            dtype=context.dtype,
        )
    )
    optimizer = torch.optim.Adam(
        [log_lengthscale, log_output_scale, log_noise],
        lr=learning_rate,
    )
    identity = torch.eye(
        problem.sample_count,
        device=context.device,
        dtype=context.dtype,
    )

    for _ in range(training_steps):
        lengthscale = log_lengthscale.exp()
        output_scale = log_output_scale.exp()
        noise = log_noise.exp()
        covariance = _rbf_kernel(
            train_inputs,
            train_inputs,
            lengthscale,
            output_scale,
        ) + (noise + jitter) * identity
        cholesky = torch.linalg.cholesky(covariance)
        alpha = torch.cholesky_solve(
            train_targets.unsqueeze(1),
            cholesky,
        ).squeeze(1)
        negative_log_likelihood = (
            0.5 * train_targets.dot(alpha)
            + torch.log(torch.diagonal(cholesky)).sum()
            + 0.5 * problem.sample_count * math.log(2.0 * math.pi)
        )
        optimizer.zero_grad(set_to_none=True)
        negative_log_likelihood.backward()
        optimizer.step()
        with torch.no_grad():
            log_lengthscale.clamp_(math.log(1e-3), math.log(1e3))
            log_output_scale.clamp_(math.log(1e-4), math.log(1e4))
            log_noise.clamp_(math.log(1e-8), math.log(1.0))

    with torch.no_grad():
        lengthscale = log_lengthscale.exp().detach()
        output_scale = log_output_scale.exp().detach()
        noise = log_noise.exp().detach()
        covariance = _rbf_kernel(
            train_inputs,
            train_inputs,
            lengthscale,
            output_scale,
        ) + (noise + jitter) * identity
        cholesky = torch.linalg.cholesky(covariance).detach()
        alpha = torch.cholesky_solve(
            train_targets.unsqueeze(1),
            cholesky,
        ).squeeze(1).detach()
        final_nll = (
            0.5 * train_targets.dot(alpha)
            + torch.log(torch.diagonal(cholesky)).sum()
            + 0.5 * problem.sample_count * math.log(2.0 * math.pi)
        )

    return ExactRBFGaussianProcess(
        train_inputs=train_inputs.detach(),
        train_targets=train_targets.detach(),
        feature_mean=feature_mean.detach(),
        feature_std=feature_std.detach(),
        utility_mean=utility_mean.detach(),
        utility_std=utility_std.detach(),
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise=noise,
        cholesky=cholesky,
        alpha=alpha,
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
        factor, info = torch.linalg.cholesky_ex(
            covariance + current_jitter * identity
        )
        if int(info.max().detach().cpu()) == 0:
            return factor, current_jitter
        current_jitter *= 10.0
    raise RuntimeError("GP posterior covariance is not positive definite")


def _rbf_kernel(
    first: torch.Tensor,
    second: torch.Tensor,
    lengthscale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    scaled_difference = (
        first[:, None, :] - second[None, :, :]
    ) / lengthscale
    squared_distance = scaled_difference.square().sum(dim=-1)
    return output_scale * torch.exp(-0.5 * squared_distance)
