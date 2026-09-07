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
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import (
    initialize_mixed_candidate_designs,
)
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


class DropoutMLP(nn.Module):
    """MLP with generator-controlled dropout for isolated reproducibility."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_size: int,
        num_layers: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()
        layers = []
        width = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(width, hidden_size))
            width = hidden_size
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(width, 1)
        self.dropout_probability = dropout_probability

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        generator: torch.Generator,
        stochastic: bool,
    ) -> torch.Tensor:
        hidden = inputs
        for layer in self.hidden_layers:
            hidden = torch.relu(layer(hidden))
            if stochastic:
                keep = torch.rand(
                    hidden.shape,
                    generator=generator,
                    device=hidden.device,
                    dtype=hidden.dtype,
                ) >= self.dropout_probability
                hidden = hidden * keep / (1.0 - self.dropout_probability)
        return self.output_layer(hidden).squeeze(-1)


@register_method()
class MCDropoutMethod(OfflineBBOMethod):
    """Risk-aware candidate search using inference-time dropout uncertainty."""

    metadata = MethodMetadata(
        method_id="mc_dropout",
        display_name="MC-Dropout adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url="https://arxiv.org/abs/1506.02142",
        description=(
            "Inference-time dropout estimates epistemic uncertainty for "
            "lower-confidence-bound candidate optimization."
        ),
        adaptations=(
            "generator_controlled_pytorch_dropout",
            "risk_aware_lower_confidence_bound",
            "model_scale_and_training_step_context",
            "generic_simplex_and_box_spaces",
            "shared_config_due_unpublished_llm_dm_hyperparameters",
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
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout_probability: float = 0.1,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        mc_samples: int = 32,
        uncertainty_weight: float = 1.0,
        particle_steps: int = 100,
        particle_learning_rate: float = 5e-2,
        random_start_fraction: float = 0.25,
        minimum_std: float = 1e-6,
    ) -> None:
        for name, value in (
            ("hidden_size", hidden_size),
            ("num_layers", num_layers),
            ("epochs", epochs),
            ("batch_size", batch_size),
            ("mc_samples", mc_samples),
            ("particle_steps", particle_steps),
        ):
            _positive_integer(name, value)
        for name, value in (
            ("learning_rate", learning_rate),
            ("particle_learning_rate", particle_learning_rate),
            ("minimum_std", minimum_std),
        ):
            _positive_float(name, value)
        _open_fraction("dropout_probability", dropout_probability)
        _nonnegative_float("uncertainty_weight", uncertainty_weight)
        _fraction("random_start_fraction", random_start_fraction)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_probability = float(dropout_probability)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.mc_samples = mc_samples
        self.uncertainty_weight = float(uncertainty_weight)
        self.particle_steps = particle_steps
        self.particle_learning_rate = float(particle_learning_rate)
        self.random_start_fraction = float(random_start_fraction)
        self.minimum_std = float(minimum_std)

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        model, feature_mean, feature_std, training_summary = self._fit(
            problem,
            context,
            generator,
        )
        initial_designs, logged_count, random_count = (
            initialize_mixed_candidate_designs(
                problem,
                candidate_budget=context.candidate_budget,
                random_fraction=self.random_start_fraction,
                generator=generator,
                device=context.device,
                dtype=context.dtype,
            )
        )
        parameters = torch.nn.Parameter(
            problem.design_space.to_unconstrained(initial_designs).detach()
        )
        optimizer = torch.optim.Adam(
            [parameters],
            lr=self.particle_learning_rate,
        )
        for _ in range(self.particle_steps):
            candidates = problem.design_space.from_unconstrained(parameters)
            predictions = _mc_predictions(
                model,
                _normalized_target_features(
                    problem,
                    candidates,
                    feature_mean,
                    feature_std,
                ),
                generator,
                self.mc_samples,
            )
            predicted_mean = predictions.mean(dim=0)
            predicted_std = predictions.std(dim=0, unbiased=False)
            acquisition = predicted_mean - self.uncertainty_weight * predicted_std
            optimizer.zero_grad(set_to_none=True)
            (-acquisition.mean()).backward()
            optimizer.step()
            with torch.no_grad():
                parameters.clamp_(-20.0, 20.0)

        candidates = problem.design_space.from_unconstrained(parameters).detach()
        with torch.no_grad():
            predictions = _mc_predictions(
                model,
                _normalized_target_features(
                    problem,
                    candidates,
                    feature_mean,
                    feature_std,
                ),
                generator,
                self.mc_samples,
            )
            predicted_mean = predictions.mean(dim=0)
            predicted_std = predictions.std(dim=0, unbiased=False)
            acquisition = predicted_mean - self.uncertainty_weight * predicted_std

        return MethodResult(
            candidates=candidates,
            training_summary=training_summary,
            diagnostics={
                "search": "mc_dropout_lower_confidence_bound",
                "particle_steps": self.particle_steps,
                "mc_samples": self.mc_samples,
                "uncertainty_weight": self.uncertainty_weight,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "predicted_standardized_utility_mean": float(
                    predicted_mean.mean().cpu()
                ),
                "predicted_standardized_utility_max": float(
                    predicted_mean.max().cpu()
                ),
                "predicted_standardized_std_mean": float(
                    predicted_std.mean().cpu()
                ),
                "acquisition_mean": float(acquisition.mean().cpu()),
            },
        )

    def _fit(
        self,
        problem: OfflineProblem,
        context: RunContext,
        generator: torch.Generator,
    ) -> tuple[DropoutMLP, torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
        features = problem.train_features
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0, unbiased=False).clamp_min(
            self.minimum_std
        )
        normalized_features = (features - feature_mean) / feature_std
        utility, utility_mean, utility_std = problem.standardized_utility(
            self.minimum_std
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(context.method_seed)
            model = DropoutMLP(
                normalized_features.shape[1],
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout_probability=self.dropout_probability,
            ).to(device=context.device, dtype=context.dtype)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        for _ in range(self.epochs):
            order = torch.randperm(
                problem.sample_count,
                generator=generator,
                device=context.device,
            )
            for start in range(0, problem.sample_count, self.batch_size):
                indices = order[start : start + self.batch_size]
                prediction = model(
                    normalized_features[indices],
                    generator=generator,
                    stochastic=True,
                )
                loss = torch.nn.functional.mse_loss(prediction, utility[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            predictions = _mc_predictions(
                model,
                normalized_features,
                generator,
                self.mc_samples,
            )
            final_mse = torch.nn.functional.mse_loss(
                predictions.mean(dim=0),
                utility,
            )
        return model, feature_mean.detach(), feature_std.detach(), {
            "surrogate": "mc_dropout_mlp",
            "train_samples": problem.sample_count,
            "feature_dim": int(normalized_features.shape[1]),
            "epochs": self.epochs,
            "dropout_probability": self.dropout_probability,
            "final_standardized_mse": float(final_mse.cpu()),
            "utility_mean": float(utility_mean.cpu()),
            "utility_std": float(utility_std.cpu()),
        }


def _normalized_target_features(
    problem: OfflineProblem,
    candidates: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
    return (problem.features_at_target(candidates) - feature_mean) / feature_std


def _mc_predictions(
    model: DropoutMLP,
    features: torch.Tensor,
    generator: torch.Generator,
    samples: int,
) -> torch.Tensor:
    return torch.stack(
        [
            model(features, generator=generator, stochastic=True)
            for _ in range(samples)
        ]
    )


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _nonnegative_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number")


def _fraction(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be in [0, 1)")
    if not math.isfinite(float(value)) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")


def _open_fraction(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be in (0, 1)")
    if not math.isfinite(float(value)) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
