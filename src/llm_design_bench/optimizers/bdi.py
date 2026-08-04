import torch
from torch import nn

from llm_design_bench.optimizers.offline_utils import (
    design_parameters,
    finalize_offline_trace,
    offline_data,
    parameters_to_designs,
    set_seed,
    target_features,
    unique_top_mixtures,
)


class BackwardDistillationOptimizer:
    """A simplex-aware BDI adaptation for the low-dimensional mixture space.

    The public BDI implementation uses an infinite-width NTK and Design-Bench
    task transforms. Here an RBF kernel gives the same differentiable
    distillation structure without importing the legacy runtime.
    """

    def __init__(
        self,
        recommendations: int = 128,
        seed: int = 0,
        steps: int = 100,
        learning_rate: float = 5e-2,
        lengthscale: float | None = None,
        ridge: float = 1e-3,
        label: float = 2.0,
        gamma: float = 0.0,
        forward_weight: float = 1.0,
        distillation_weight: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.recommendations = recommendations
        self.seed = seed
        self.steps = steps
        self.learning_rate = learning_rate
        self.lengthscale = lengthscale
        self.ridge = ridge
        self.label = label
        self.gamma = gamma
        self.forward_weight = forward_weight
        self.distillation_weight = distillation_weight
        self.device = device

    def optimize(self, task):
        set_seed(self.seed)
        data = offline_data(task, device=self.device)
        lengthscale = self.lengthscale or _median_lengthscale(data.features)
        kernel = lambda left, right: _rbf_kernel(left, right, lengthscale)
        identity = torch.eye(len(data.features), device=self.device)
        coefficients = torch.linalg.solve(
            kernel(data.features, data.features) + self.ridge * identity,
            data.utility,
        ).detach()
        weights = torch.softmax(self.gamma * data.utility, dim=0).detach()

        parameters = nn.Parameter(
            design_parameters(
                unique_top_mixtures(task, self.recommendations),
                task,
                device=self.device,
            )
        )
        optimizer = torch.optim.Adam([parameters], lr=self.learning_rate)
        for _ in range(self.steps):
            mixtures = parameters_to_designs(parameters, task)
            support = target_features(mixtures, task)
            forward_score = kernel(support, data.features) @ coefficients
            support_labels = torch.full_like(forward_score, self.label)
            support_identity = torch.eye(len(support), device=self.device)
            distilled_coefficients = torch.linalg.solve(
                kernel(support, support) + self.ridge * support_identity,
                support_labels,
            )
            reconstructed = kernel(data.features, support) @ distilled_coefficients
            distillation_loss = torch.sum(weights * (reconstructed - data.utility).square())
            loss = -self.forward_weight * forward_score.mean()
            loss = loss + self.distillation_weight * distillation_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return finalize_offline_trace("bdi", task, parameters_to_designs(parameters, task))


def _median_lengthscale(features: torch.Tensor) -> float:
    distances = torch.pdist(features.detach())
    nonzero = distances[distances > 0]
    if len(nonzero) == 0:
        return 1.0
    return float(nonzero.median().clamp_min(1e-3))


def _rbf_kernel(left: torch.Tensor, right: torch.Tensor, lengthscale: float) -> torch.Tensor:
    squared_distance = torch.cdist(left, right).square()
    return torch.exp(-0.5 * squared_distance / (lengthscale**2))
