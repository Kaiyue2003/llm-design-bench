"""Importance-aware co-teaching, implemented from the ICT paper equations."""

from __future__ import annotations

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from llm_design_bench.optimizers.mentoring_utils import (
    MentoringData,
    differentiable_sgd_step,
)


def exchanged_small_loss_indices(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    remember_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return each student's training indices, chosen by the OTHER student."""
    if predictions.ndim != 2 or predictions.shape[0] != 2:
        raise ValueError("predictions must have shape (2, neighbors)")
    if labels.shape != predictions.shape[1:]:
        raise ValueError("labels must have shape (neighbors,)")
    if (
        isinstance(remember_count, bool)
        or not isinstance(remember_count, int)
        or not 1 <= remember_count <= len(labels)
    ):
        raise ValueError("remember_count must be between one and neighbor count")
    if not torch.isfinite(predictions).all() or not torch.isfinite(labels).all():
        raise RuntimeError("non-finite co-teaching predictions")
    order = (
        (predictions.detach() - labels.detach()).square().argsort(dim=1, stable=True)
    )
    return order[1, :remember_count], order[0, :remember_count]


def weighted_pseudo_loss(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    prediction = model(features)
    # Enforce row-wise weights: (K, 1) * (K,) would silently create a K x K loss.
    if (
        prediction.ndim != 1
        or prediction.shape != labels.shape
        or labels.shape != weights.shape
    ):
        raise ValueError(
            "prediction, labels and weights must have matching (K,) shapes"
        )
    if prediction.numel() == 0:
        raise ValueError("weighted pseudo loss requires at least one sample")
    return (weights * (prediction - labels.detach()).square()).mean()


def sample_weight_meta_loss(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    data: MentoringData,
    learning_rate: float,
) -> torch.Tensor:
    """Logged MSE after the virtual weighted SGD update (ICT equations 6–8)."""
    updated = differentiable_sgd_step(
        model, weighted_pseudo_loss(model, features, labels, weights), learning_rate
    )
    return F.mse_loss(functional_call(model, updated, (data.features,)), data.utility)


def reweight_and_update_proxy(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    data: MentoringData,
    *,
    learning_rate: float,
    weight_learning_rate: float,
    reweighting: bool,
) -> torch.Tensor:
    """Refine weights from ones, then update the ORIGINAL proxy exactly once."""
    features, labels = features.detach(), labels.detach()
    weights = torch.ones_like(labels, requires_grad=reweighting)
    if reweighting:
        meta_loss = sample_weight_meta_loss(
            model, features, labels, weights, data, learning_rate
        )
        gradient = torch.autograd.grad(meta_loss, weights)[0]
        if not torch.isfinite(meta_loss) or not torch.isfinite(gradient).all():
            raise RuntimeError("non-finite ICT sample-weight meta-gradient")
        # Use the paper's raw gradient, with the reference runtime's [0, 2] guard.
        weights = (weights - weight_learning_rate * gradient).detach().clamp(0, 2)
    loss = weighted_pseudo_loss(model, features, labels, weights)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("non-finite ICT proxy update")
    with torch.no_grad():
        for parameter, gradient in zip(model.parameters(), gradients):
            parameter.add_(gradient, alpha=-learning_rate)
    return weights.detach()


def co_teach_round(
    models: list[nn.Module],
    teacher_index: int,
    features: torch.Tensor,
    data: MentoringData,
    *,
    remember_count: int,
    learning_rate: float,
    weight_learning_rate: float,
    reweighting: bool,
) -> float:
    """One teacher labels; two students exchange selections made before updates."""
    if len(models) != 3 or teacher_index not in range(3):
        raise ValueError("co-teaching requires three models and a teacher index 0–2")
    students = [model for i, model in enumerate(models) if i != teacher_index]
    with torch.no_grad():
        labels = models[teacher_index](features).detach()
        predictions = torch.stack([model(features) for model in students])
        selections = exchanged_small_loss_indices(predictions, labels, remember_count)
    change = 0.0
    for student, selected in zip(students, selections):
        weights = reweight_and_update_proxy(
            student,
            features[selected],
            labels[selected],
            data,
            learning_rate=learning_rate,
            weight_learning_rate=weight_learning_rate,
            reweighting=reweighting,
        )
        change += float((weights - 1).abs().sum())
    return change
