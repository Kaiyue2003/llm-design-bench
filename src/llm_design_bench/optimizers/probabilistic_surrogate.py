from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from llm_design_bench.problem import OfflineProblem, RunContext


class GaussianMLP(nn.Module):
    """MLP with a bounded Gaussian output, following Design-Baselines."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_size: int,
        num_layers: int,
        initial_min_std: float,
        initial_max_std: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(width, hidden_size), nn.LeakyReLU()])
            width = hidden_size
        layers.append(nn.Linear(width, 2))
        self.layers = nn.Sequential(*layers)
        self.min_logstd = nn.Parameter(torch.tensor(math.log(initial_min_std)))
        self.max_logstd = nn.Parameter(torch.tensor(math.log(initial_max_std)))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_logstd = self.layers(inputs).unbind(dim=-1)
        max_logstd = self.max_logstd.to(dtype=inputs.dtype)
        min_logstd = self.min_logstd.to(dtype=inputs.dtype)
        logstd = max_logstd - torch.nn.functional.softplus(
            max_logstd - raw_logstd
        )
        logstd = min_logstd + torch.nn.functional.softplus(logstd - min_logstd)
        return mean, logstd


@dataclass(frozen=True)
class GaussianEnsemble:
    models: tuple[GaussianMLP, ...]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    utility_mean: torch.Tensor
    utility_std: torch.Tensor
    final_mse: float
    final_nll: float

    def predict_standardized(
        self,
        problem: OfflineProblem,
        designs: torch.Tensor,
    ) -> torch.Tensor:
        features = problem.features_at_target(designs)
        normalized = (features - self.feature_mean) / self.feature_std
        means = torch.stack([model(normalized)[0] for model in self.models])
        return means.mean(dim=0)

    def training_summary(self) -> dict[str, float | int | str]:
        return {
            "surrogate": "gaussian_mlp_ensemble",
            "ensemble_size": len(self.models),
            "final_standardized_mse": self.final_mse,
            "final_standardized_nll": self.final_nll,
            "utility_mean": float(self.utility_mean.cpu()),
            "utility_std": float(self.utility_std.cpu()),
        }


def fit_gaussian_ensemble(
    problem: OfflineProblem,
    context: RunContext,
    generator: torch.Generator,
    *,
    ensemble_size: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    initial_min_std: float,
    initial_max_std: float,
    minimum_std: float,
) -> GaussianEnsemble:
    features = problem.train_features
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0, unbiased=False).clamp_min(minimum_std)
    normalized = (features - feature_mean) / feature_std
    utility, utility_mean, utility_std = problem.standardized_utility(minimum_std)

    models: list[GaussianMLP] = []
    optimizers: list[torch.optim.Optimizer] = []
    for index in range(ensemble_size):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                context.method_seed + 10_007 * index
            )
            model = GaussianMLP(
                normalized.shape[1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                initial_min_std=initial_min_std,
                initial_max_std=initial_max_std,
            ).to(device=context.device, dtype=context.dtype)
        models.append(model)
        optimizers.append(torch.optim.Adam(model.parameters(), lr=learning_rate))

    bootstraps = [
        torch.randint(
            problem.sample_count,
            (problem.sample_count,),
            generator=generator,
            device=context.device,
        )
        for _ in models
    ]
    for _ in range(epochs):
        for model, optimizer, bootstrap in zip(
            models,
            optimizers,
            bootstraps,
            strict=True,
        ):
            order = torch.randperm(
                problem.sample_count,
                generator=generator,
                device=context.device,
            )
            shuffled_bootstrap = bootstrap[order]
            for start in range(0, problem.sample_count, batch_size):
                indices = shuffled_bootstrap[start : start + batch_size]
                mean, logstd = model(normalized[indices])
                inverse_variance = torch.exp(-2.0 * logstd)
                nll = 0.5 * (
                    (utility[indices] - mean).square() * inverse_variance
                    + 2.0 * logstd
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                nll.backward()
                optimizer.step()

    with torch.no_grad():
        means = []
        nlls = []
        for model in models:
            model.eval()
            mean, logstd = model(normalized)
            means.append(mean)
            nlls.append(
                0.5
                * (
                    (utility - mean).square() * torch.exp(-2.0 * logstd)
                    + 2.0 * logstd
                ).mean()
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        ensemble_mean = torch.stack(means).mean(dim=0)
        final_mse = torch.nn.functional.mse_loss(ensemble_mean, utility)
        final_nll = torch.stack(nlls).mean()

    return GaussianEnsemble(
        models=tuple(models),
        feature_mean=feature_mean.detach(),
        feature_std=feature_std.detach(),
        utility_mean=utility_mean.detach(),
        utility_std=utility_std.detach(),
        final_mse=float(final_mse.cpu()),
        final_nll=float(final_nll.cpu()),
    )
