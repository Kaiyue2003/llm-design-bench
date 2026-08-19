from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from llm_design_bench.optimizers.base import EvaluationTrace
from llm_design_bench.types import CandidateBatch


@dataclass(frozen=True)
class OfflineData:
    features: torch.Tensor
    utility: torch.Tensor
    utility_mean: torch.Tensor
    utility_std: torch.Tensor


class MLPSurrogate(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def offline_data(task, device: str = "cpu") -> OfflineData:
    features = batch_features(task.logged_x, task, device=device)
    utility = torch.as_tensor(task.logged_y, dtype=torch.float32, device=device)
    utility_mean = utility.mean()
    utility_std = utility.std().clamp_min(1e-6)
    return OfflineData(
        features=features,
        utility=(utility - utility_mean) / utility_std,
        utility_mean=utility_mean,
        utility_std=utility_std,
    )


def batch_features(batch: CandidateBatch, task, device: str = "cpu") -> torch.Tensor:
    mixtures = torch.as_tensor(batch.mixtures, dtype=torch.float32, device=device)
    model_scales = torch.as_tensor(batch.model_scales, dtype=torch.float32, device=device)
    training_steps = torch.as_tensor(batch.training_steps, dtype=torch.float32, device=device)
    target_model_scale = float(getattr(task, "target_model_scale", 1.0))
    max_training_steps = float(getattr(task, "max_training_steps", 1.0))
    scale_feature = torch.log1p(model_scales) / np.log1p(target_model_scale)
    step_feature = training_steps / max_training_steps
    return torch.cat(
        [mixtures, scale_feature[:, None], step_feature[:, None]],
        dim=1,
    )


def target_features(mixtures: torch.Tensor, task) -> torch.Tensor:
    count = len(mixtures)
    target_model_scale = float(getattr(task, "target_model_scale", 1.0))
    target_training_steps = float(getattr(task, "target_training_steps", 1.0))
    max_training_steps = float(getattr(task, "max_training_steps", 1.0))
    scale = torch.full(
        (count, 1),
        np.log1p(target_model_scale) / np.log1p(target_model_scale),
        dtype=mixtures.dtype,
        device=mixtures.device,
    )
    steps = torch.full(
        (count, 1),
        target_training_steps / max_training_steps,
        dtype=mixtures.dtype,
        device=mixtures.device,
    )
    return torch.cat([mixtures, scale, steps], dim=1)


def fit_surrogate(
    model: nn.Module,
    data: OfflineData,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        order = torch.randperm(len(data.features), device=data.features.device)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            loss = nn.functional.mse_loss(model(data.features[indices]), data.utility[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def unique_top_mixtures(task, count: int) -> np.ndarray:
    selected: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for index in np.argsort(task.logged_y)[::-1]:
        mixture = task.logged_x.mixtures[index]
        key = tuple(np.round(mixture, 12))
        if key in seen:
            continue
        seen.add(key)
        selected.append(mixture)
        if len(selected) == count:
            break
    if not selected:
        raise ValueError("logged dataset does not contain any mixtures")
    while len(selected) < count:
        selected.extend(selected[: count - len(selected)])
    return np.vstack(selected[:count])


def design_parameters(
    designs: np.ndarray | torch.Tensor,
    task,
    device: str = "cpu",
) -> torch.Tensor:
    tensor = torch.as_tensor(designs, dtype=torch.float32, device=device)
    bounds = _box_bounds(task, device=device)
    if bounds is None:
        return torch.log(tensor.clamp_min(1e-8))
    lower, upper = bounds
    unit = ((tensor - lower) / (upper - lower).clamp_min(1e-8)).clamp(1e-6, 1 - 1e-6)
    return torch.logit(unit)


def parameters_to_designs(parameters: torch.Tensor, task) -> torch.Tensor:
    bounds = _box_bounds(task, device=str(parameters.device))
    if bounds is None:
        return torch.softmax(parameters, dim=1)
    lower, upper = bounds
    return lower + (upper - lower) * torch.sigmoid(parameters)


def mixture_logits(mixtures: np.ndarray | torch.Tensor, device: str = "cpu") -> torch.Tensor:
    tensor = torch.as_tensor(mixtures, dtype=torch.float32, device=device)
    return torch.log(tensor.clamp_min(1e-8))


def finalize_offline_trace(name: str, task, mixtures: torch.Tensor) -> EvaluationTrace:
    recommendations = task.at_target_fidelity(mixtures.detach().cpu().numpy())
    recommendation_utility = task.predict(recommendations)
    mixture_dim = int(getattr(task, "mixture_dim", recommendations.mixtures.shape[1]))
    empty = CandidateBatch(
        mixtures=np.empty((0, mixture_dim)),
        model_scales=np.empty(0),
        training_steps=np.empty(0),
    )
    return EvaluationTrace(
        name=name,
        recommendations=recommendations,
        recommendation_utility=recommendation_utility,
        queried=empty,
        query_utility=np.empty(0),
        query_cost=np.empty(0),
    )


def _box_bounds(task, device: str):
    bounds = getattr(task, "design_bounds", None)
    if bounds is None:
        return None
    tensor = torch.as_tensor(bounds, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError("design_bounds must have shape (dimension, 2)")
    return tensor[:, 0], tensor[:, 1]
