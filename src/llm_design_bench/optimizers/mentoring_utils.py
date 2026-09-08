"""Torch components for offline proxy mentoring; no task or oracle access."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from llm_design_bench.problem import OfflineProblem, RunContext


class MentoringMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class MentoringData:
    features: torch.Tensor
    utility: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    @classmethod
    def from_problem(
        cls,
        problem: OfflineProblem,
        minimum_std: float,
    ) -> "MentoringData":
        features = problem.train_features.detach()
        mean = features.mean(0)
        std = features.std(0, unbiased=False).clamp_min(minimum_std)
        utility, _, _ = problem.standardized_utility(minimum_std)
        return cls((features - mean) / std, utility.detach(), mean, std)

    def at_target(
        self,
        problem: OfflineProblem,
        designs: torch.Tensor,
    ) -> torch.Tensor:
        return (problem.features_at_target(designs) - self.feature_mean) / (
            self.feature_std
        )


def fit_proxy_ensemble(
    data: MentoringData,
    context: RunContext,
    generator: torch.Generator,
    *,
    ensemble_size: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
) -> tuple[list[nn.Module], dict]:
    """Fit independent MSE proxies; select checkpoints using offline data only.

    Each proxy has an independent initialization and shuffled split. With too
    few rows for Pearson correlation, select by validation MSE; with fewer
    than three rows (or validation disabled), retain the final checkpoint.
    """
    models, selected_epochs = [], []
    count = len(data.utility)
    validation_count = (
        min(count - 1, max(1, int(count * validation_fraction)))
        if count >= 3 and validation_fraction > 0
        else 0
    )
    for _ in range(ensemble_size):
        seed = int(
            torch.randint(
                0, 2**31 - 1, (), generator=generator, device=context.device
            ).item()
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = MentoringMLP(data.features.shape[1], hidden_size).to(
                device=context.device, dtype=context.dtype
            )
        order = torch.randperm(count, generator=generator, device=context.device)
        val_ids, train_ids = order[:validation_count], order[validation_count:]
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_score, best_state, selected_epoch = -math.inf, None, epochs
        for epoch in range(epochs):
            rate = learning_rate * (1 + math.cos(math.pi * epoch / epochs)) / 2
            for group in optimizer.param_groups:
                group["lr"] = rate
            shuffled = train_ids[
                torch.randperm(
                    len(train_ids), generator=generator, device=context.device
                )
            ]
            for indices in shuffled.split(batch_size):
                loss = F.mse_loss(model(data.features[indices]), data.utility[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite mentoring pretraining loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            if validation_count:
                with torch.no_grad():
                    prediction, labels = (
                        model(data.features[val_ids]),
                        data.utility[val_ids],
                    )
                    centered_p = prediction - prediction.mean()
                    centered_y = labels - labels.mean()
                    denominator = centered_p.norm() * centered_y.norm()
                    score = (
                        centered_p.dot(centered_y) / denominator.clamp_min(1e-12)
                        if validation_count >= 2 and centered_y.norm() > 1e-12
                        else -F.mse_loss(prediction, labels)
                    )
                    if float(score) > best_score:
                        best_score = float(score)
                        best_state = copy.deepcopy(model.state_dict())
                        selected_epoch = epoch + 1
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        selected_epochs.append(selected_epoch)
    return models, {
        "ensemble_size": ensemble_size,
        "epochs": epochs,
        "train_samples": count,
        "validation_samples_per_proxy": validation_count,
        "selected_epochs": selected_epochs,
        "feature_dim": int(data.features.shape[1]),
        "validation_selection": "pearson_or_negative_mse_for_small_constant_labels",
    }


def sample_local_designs(
    problem: OfflineProblem,
    parameters: torch.Tensor,
    neighbor_samples: int,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Perturb only design parameters; the caller appends fixed target context."""
    noise = torch.randn(
        (neighbor_samples, problem.design_dim),
        generator=generator,
        device=parameters.device,
        dtype=parameters.dtype,
    )
    return problem.design_space.from_unconstrained(
        parameters.detach() + noise_std * noise
    ).detach()


def pairwise_consensus(
    predictions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unique pairs, majority labels and proxy disagreement masks.

    predictions has shape (3, neighbors). A vote is 1 only for a strictly
    greater prediction, as in the paper's indicator; ties vote 0.
    """
    if predictions.ndim != 2 or predictions.shape[0] != 3:
        raise ValueError("predictions must have shape (3, neighbors)")
    if predictions.shape[1] < 2 or not torch.isfinite(predictions).all():
        raise ValueError("at least two finite neighborhood predictions required")
    pairs = torch.triu_indices(
        predictions.shape[1],
        predictions.shape[1],
        offset=1,
        device=predictions.device,
    )
    votes = predictions[:, pairs[0]] > predictions[:, pairs[1]]
    consensus = votes.sum(0) >= 2
    return pairs, consensus.to(predictions.dtype), votes != consensus.unsqueeze(0)


def differentiable_sgd_step(
    model: nn.Module,
    loss: torch.Tensor,
    learning_rate: float,
) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, tuple(parameters.values()), create_graph=True)
    return {
        name: parameter - learning_rate * gradient
        for (name, parameter), gradient in zip(parameters.items(), gradients)
    }


def pairwise_loss(
    model: nn.Module,
    features: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    predictions = model(features)
    logits = predictions[pairs[0]] - predictions[pairs[1]]
    return F.binary_cross_entropy_with_logits(logits, labels)


def soft_label_meta_loss(
    model: nn.Module,
    features: torch.Tensor,
    pairs: torch.Tensor,
    labels: torch.Tensor,
    data: MentoringData,
    learning_rate: float,
) -> torch.Tensor:
    """Logged MSE after a differentiable virtual pairwise SGD update."""
    updated = differentiable_sgd_step(
        model, pairwise_loss(model, features, pairs, labels), learning_rate
    )
    prediction = functional_call(model, updated, (data.features,))
    return F.mse_loss(prediction, data.utility)


def mentor_proxy(
    model: nn.Module,
    features: torch.Tensor,
    pairs: torch.Tensor,
    consensus: torch.Tensor,
    data: MentoringData,
    *,
    learning_rate: float,
    label_learning_rate: float,
    soft_labels: bool,
) -> float:
    """Refine labels through logged MSE, then make one real SGD proxy update."""
    if pairs.shape[1] == 0:
        return 0.0
    labels = consensus.detach().clone()
    change = 0.0
    if soft_labels:
        labels.requires_grad_(True)
        meta_loss = soft_label_meta_loss(
            model, features, pairs, labels, data, learning_rate
        )
        gradient = torch.autograd.grad(meta_loss, labels)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError("non-finite soft-label meta-gradient")
        refined = (labels - label_learning_rate * gradient).detach().clamp(0, 1)
        change = float((refined - labels.detach()).abs().sum())
        labels = refined
    loss = pairwise_loss(model, features, pairs, labels)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    with torch.no_grad():
        for parameter, gradient in zip(model.parameters(), gradients):
            parameter.add_(gradient, alpha=-learning_rate)
    return change
