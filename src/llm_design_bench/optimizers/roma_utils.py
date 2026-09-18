"""Native functional weight perturbation and candidate-local RoMA adaptation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.func import functional_call

from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.problem import OfflineProblem

Weights = Mapping[str, torch.Tensor]


def gaussian_nll(
    mean: torch.Tensor, logstd: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Gaussian NLL up to an additive constant, with strict row-wise shapes."""
    if mean.ndim != 1 or mean.shape != logstd.shape or mean.shape != labels.shape:
        raise ValueError("mean, logstd and labels must have matching (N,) shapes")
    return (0.5 * (mean - labels).square() * (-2 * logstd).exp() + logstd).mean()


def relative_projected_step(
    weights: Weights,
    reference: Weights,
    gradients: tuple[torch.Tensor, ...],
    *,
    step_size: float,
    radius: float,
    ascent: bool,
) -> dict[str, torch.Tensor]:
    """Normalized per-tensor PGD inside radius * ||reference tensor||.

    Zero-norm reference tensors remain fixed. Returned tensors are detached
    leaves, preventing differentiation through the PGD optimization trajectory.
    """
    result = {}
    with torch.no_grad():
        for (name, weight), gradient in zip(weights.items(), gradients, strict=True):
            if not torch.isfinite(gradient).all():
                raise RuntimeError("non-finite RoMA weight gradient")
            origin = reference[name].detach()
            origin_norm, grad_norm = origin.norm(), gradient.norm()
            if not torch.isfinite(origin_norm) or not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite RoMA weight norm")
            tiny = torch.finfo(weight.dtype).tiny
            direction = gradient / grad_norm.clamp_min(tiny)
            delta = (
                weight
                - origin
                + (1 if ascent else -1) * step_size * origin_norm * direction
            )
            delta_norm = delta.norm()
            if not torch.isfinite(delta_norm):
                raise RuntimeError("non-finite RoMA weight perturbation")
            factor = (radius * origin_norm / delta_norm.clamp_min(tiny)).clamp(max=1)
            projected = origin + delta * factor
            result[name] = projected.detach().requires_grad_(True)
    return result


def adversarial_weights(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    steps: int,
    radius: float,
) -> dict[str, torch.Tensor]:
    """Maximize noisy logged NLL in a bounded weight neighborhood; no mutation."""
    reference = {name: p.detach() for name, p in model.named_parameters()}
    weights = {name: p.clone().requires_grad_(True) for name, p in reference.items()}
    if radius == 0:
        return weights
    for _ in range(steps):
        loss = gaussian_nll(*functional_call(model, weights, (features,)), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite RoMA adversarial loss")
        gradients = torch.autograd.grad(loss, tuple(weights.values()))
        weights = relative_projected_step(
            weights,
            reference,
            gradients,
            step_size=radius / steps,
            radius=radius,
            ascent=True,
        )
    return weights


def noisy_design_features(
    features: torch.Tensor,
    design_dim: int,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Gaussian training augmentation in normalized design columns, not fidelity."""
    noisy = features.detach().clone()
    noise = torch.randn(
        (len(features), design_dim),
        generator=generator,
        device=features.device,
        dtype=features.dtype,
    )
    noisy[:, :design_dim] += noise_std * noise
    return noisy


def adaptation_loss(
    model: nn.Module,
    weights: Weights,
    coordinates: torch.Tensor,
    previous_mean: torch.Tensor,
    problem: OfflineProblem,
    data: MentoringData,
    *,
    consistency_weight: float,
    uncertainty_weight: float,
) -> torch.Tensor:
    """Design-gradient norm plus consistency NLL against a detached old mean.

    Differentiate the norm through model weights (second-order autograd), but
    never differentiate fidelity or backpropagate to the previous prediction.
    """
    design = problem.design_space.from_unconstrained(coordinates)
    mean, logstd = functional_call(model, weights, (data.at_target(problem, design),))
    score = mean - uncertainty_weight * logstd
    input_gradient = torch.autograd.grad(score.sum(), coordinates, create_graph=True)[0]
    smoothness = torch.linalg.vector_norm(input_gradient, dim=1).mean()
    return smoothness + consistency_weight * gaussian_nll(
        mean, logstd, previous_mean.detach()
    )


def adapt_candidate_weights(
    model: nn.Module,
    coordinates: torch.Tensor,
    previous_mean: torch.Tensor,
    problem: OfflineProblem,
    data: MentoringData,
    *,
    steps: int,
    radius: float,
    consistency_weight: float,
    uncertainty_weight: float,
) -> dict[str, torch.Tensor]:
    """Reset to pretrained weights for each update, then minimize local roughness."""
    reference = {name: p.detach() for name, p in model.named_parameters()}
    weights = {name: p.clone().requires_grad_(True) for name, p in reference.items()}
    point = coordinates.detach().clone().requires_grad_(True)
    if steps and radius:
        for _ in range(steps):
            loss = adaptation_loss(
                model,
                weights,
                point,
                previous_mean,
                problem,
                data,
                consistency_weight=consistency_weight,
                uncertainty_weight=uncertainty_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite RoMA adaptation loss")
            gradients = torch.autograd.grad(loss, tuple(weights.values()))
            weights = relative_projected_step(
                weights,
                reference,
                gradients,
                step_size=radius / steps,
                radius=radius,
                ascent=False,
            )
    return {name: p.detach() for name, p in weights.items()}


def trust_region_score(
    mean: torch.Tensor,
    logstd: torch.Tensor,
    initial_score: torch.Tensor,
    *,
    uncertainty_weight: float,
    region: float,
) -> torch.Tensor:
    """Source-style score-space penalty, NOT a distance penalty on designs."""
    score = mean - uncertainty_weight * logstd
    return score - (score - initial_score.detach()).square() / (2 * region)
