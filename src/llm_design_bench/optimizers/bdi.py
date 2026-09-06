from __future__ import annotations

import math

import torch
from torch import nn

from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.offline_utils import (
    design_parameters,
    finalize_offline_trace,
    offline_data,
    parameters_to_designs,
    set_seed,
    target_features,
    unique_top_mixtures,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import initialize_candidate_designs
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


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


@register_method()
class BackwardDistillationMethod(OfflineBBOMethod):
    """RBF-based PyTorch BDI adaptation behind the strict offline API."""

    metadata = MethodMetadata(
        method_id="bdi",
        display_name="BDI adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.LIGHTWEIGHT_ADAPTATION,
        source_url="https://github.com/GGchen1997/BDI",
        description=(
            "Forward RBF kernel regression plus backward distillation from "
            "optimistic candidate support points."
        ),
        adaptations=(
            "pytorch",
            "rbf_kernel_replaces_infinite_width_ntk",
            "model_scale_and_training_step_context",
            "generic_simplex_and_box_spaces",
        ),
    )
    capabilities = MethodCapabilities(
        supports_simplex=True,
        supports_box=True,
        supports_context=True,
        stochastic=True,
    )

    def __init__(
        self,
        *,
        steps: int = 100,
        learning_rate: float = 5e-2,
        lengthscale: float | None = None,
        ridge: float = 1e-3,
        label: float = 2.0,
        gamma: float = 0.0,
        forward_weight: float = 1.0,
        distillation_weight: float = 1.0,
        minimum_std: float = 1e-6,
    ) -> None:
        _validate_positive_integer("steps", steps)
        _validate_positive_float("learning_rate", learning_rate)
        if lengthscale is not None:
            _validate_positive_float("lengthscale", lengthscale)
        _validate_positive_float("ridge", ridge)
        _validate_finite_float("label", label)
        _validate_finite_float("gamma", gamma)
        _validate_non_negative_float("forward_weight", forward_weight)
        _validate_non_negative_float(
            "distillation_weight",
            distillation_weight,
        )
        if forward_weight == 0 and distillation_weight == 0:
            raise ValueError("at least one BDI loss weight must be positive")
        _validate_positive_float("minimum_std", minimum_std)

        self.steps = steps
        self.learning_rate = learning_rate
        self.lengthscale = lengthscale
        self.ridge = ridge
        self.label = label
        self.gamma = gamma
        self.forward_weight = forward_weight
        self.distillation_weight = distillation_weight
        self.minimum_std = minimum_std

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        raw_features = problem.train_features
        feature_mean = raw_features.mean(dim=0)
        feature_std = raw_features.std(dim=0, unbiased=False).clamp_min(
            self.minimum_std
        )
        features = (raw_features - feature_mean) / feature_std
        utility, utility_mean, utility_std = problem.standardized_utility(
            self.minimum_std
        )
        lengthscale = self.lengthscale or _median_lengthscale(features)
        identity = torch.eye(
            problem.sample_count,
            device=context.device,
            dtype=context.dtype,
        )
        gram = _rbf_kernel(features, features, lengthscale)
        coefficients = torch.linalg.solve(
            gram + self.ridge * identity,
            utility,
        ).detach()
        weights = torch.softmax(self.gamma * utility, dim=0).detach()

        initial, logged_count, random_count = initialize_candidate_designs(
            problem,
            candidate_budget=context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        parameters = nn.Parameter(
            problem.design_space.to_unconstrained(initial).detach()
        )
        optimizer = torch.optim.Adam([parameters], lr=self.learning_rate)
        for _ in range(self.steps):
            candidates = problem.design_space.from_unconstrained(parameters)
            support = (
                problem.features_at_target(candidates) - feature_mean
            ) / feature_std
            forward_score = _rbf_kernel(
                support,
                features,
                lengthscale,
            ) @ coefficients
            support_labels = torch.full_like(forward_score, self.label)
            support_identity = torch.eye(
                len(support),
                device=context.device,
                dtype=context.dtype,
            )
            distilled_coefficients = torch.linalg.solve(
                _rbf_kernel(support, support, lengthscale)
                + self.ridge * support_identity,
                support_labels,
            )
            reconstructed = _rbf_kernel(
                features,
                support,
                lengthscale,
            ) @ distilled_coefficients
            distillation_loss = torch.sum(
                weights * (reconstructed - utility).square()
            )
            loss = -self.forward_weight * forward_score.mean()
            loss = loss + self.distillation_weight * distillation_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        candidates = problem.design_space.from_unconstrained(parameters).detach()
        with torch.no_grad():
            fitted = gram @ coefficients
            fitted_mse = torch.nn.functional.mse_loss(fitted, utility)
            final_support = (
                problem.features_at_target(candidates) - feature_mean
            ) / feature_std
            final_forward = _rbf_kernel(
                final_support,
                features,
                lengthscale,
            ) @ coefficients
            final_labels = torch.full_like(final_forward, self.label)
            final_identity = torch.eye(
                len(final_support),
                device=context.device,
                dtype=context.dtype,
            )
            final_coefficients = torch.linalg.solve(
                _rbf_kernel(final_support, final_support, lengthscale)
                + self.ridge * final_identity,
                final_labels,
            )
            final_reconstructed = _rbf_kernel(
                features,
                final_support,
                lengthscale,
            ) @ final_coefficients
            final_distillation = torch.sum(
                weights * (final_reconstructed - utility).square()
            )
        return MethodResult(
            candidates=candidates,
            training_summary={
                "train_samples": problem.sample_count,
                "feature_dim": int(features.shape[1]),
                "kernel": "rbf",
                "lengthscale": lengthscale,
                "ridge": self.ridge,
                "final_standardized_mse": float(fitted_mse.cpu()),
                "utility_mean": float(utility_mean.cpu()),
                "utility_std": float(utility_std.cpu()),
            },
            diagnostics={
                "search": "forward_backward_distillation",
                "steps": self.steps,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "final_forward_score_mean": float(final_forward.mean().cpu()),
                "final_distillation_loss": float(final_distillation.cpu()),
            },
        )


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_positive_float(name: str, value: float) -> None:
    _validate_finite_float(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _validate_non_negative_float(name: str, value: float) -> None:
    _validate_finite_float(name, value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _validate_finite_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
