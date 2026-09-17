import math

import torch

from llm_design_bench.optimizers.base import (
    FitThenProposeMethod,
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
)
from llm_design_bench.optimizers.mlp_model import MLPSurrogate
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import initialize_candidate_designs
from llm_design_bench.problem import OfflineProblem, RunContext


@register_method()
class OfflineMLPMethod(FitThenProposeMethod):
    """Torch-native forward surrogate followed by gradient candidate search."""

    metadata = MethodMetadata(
        method_id="offline_mlp",
        display_name="Offline MLP",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        description=(
            "Standardized MLP utility surrogate with gradient ascent in the "
            "task's constrained design space."
        ),
        adaptations=("pytorch", "model_scale_and_training_step_context"),
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
        hidden_size: int = 128,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        particle_steps: int = 100,
        particle_learning_rate: float = 5e-2,
        minimum_std: float = 1e-6,
    ) -> None:
        _validate_positive_integer("hidden_size", hidden_size)
        _validate_positive_integer("epochs", epochs)
        _validate_positive_integer("batch_size", batch_size)
        _validate_positive_integer("particle_steps", particle_steps)
        _validate_positive_float("learning_rate", learning_rate)
        _validate_positive_float("particle_learning_rate", particle_learning_rate)
        _validate_positive_float("minimum_std", minimum_std)

        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.particle_steps = particle_steps
        self.particle_learning_rate = particle_learning_rate
        self.minimum_std = minimum_std

        self._model: MLPSurrogate | None = None
        self._feature_mean: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None
        self._diagnostics: dict[str, float | int | str] = {}

    def fit(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> dict[str, float | int]:
        features = problem.train_features
        feature_mean = features.mean(dim=0)
        feature_std = features.std(dim=0, unbiased=False).clamp_min(self.minimum_std)
        normalized_features = (features - feature_mean) / feature_std
        standardized_utility, utility_mean, utility_std = problem.standardized_utility(
            self.minimum_std
        )

        # Linear layers initialize from Torch's global CPU generator. Fork it so
        # direct ``method.run`` calls are reproducible without leaking RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(context.method_seed)
            model = MLPSurrogate(
                normalized_features.shape[1],
                hidden_size=self.hidden_size,
            ).to(device=context.device, dtype=context.dtype)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train()
        for _ in range(self.epochs):
            order = torch.randperm(
                problem.sample_count,
                generator=generator,
                device=context.device,
            )
            for start in range(0, problem.sample_count, self.batch_size):
                indices = order[start : start + self.batch_size]
                prediction = model(normalized_features[indices])
                loss = torch.nn.functional.mse_loss(
                    prediction,
                    standardized_utility[indices],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            final_mse = torch.nn.functional.mse_loss(
                model(normalized_features),
                standardized_utility,
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self._model = model
        self._feature_mean = feature_mean.detach()
        self._feature_std = feature_std.detach()
        self._diagnostics = {}
        return {
            "train_samples": problem.sample_count,
            "feature_dim": int(normalized_features.shape[1]),
            "epochs": self.epochs,
            "final_standardized_mse": float(final_mse.detach().cpu()),
            "utility_mean": float(utility_mean.detach().cpu()),
            "utility_std": float(utility_std.detach().cpu()),
        }

    def propose(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> torch.Tensor:
        model, feature_mean, feature_std = self._fitted_state()
        initial_designs, logged_count, random_count = initialize_candidate_designs(
            problem,
            candidate_budget=context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
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
            features = problem.features_at_target(candidates)
            normalized_features = (features - feature_mean) / feature_std
            predicted_utility = model(normalized_features)
            loss = -predicted_utility.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        candidates = problem.design_space.from_unconstrained(parameters).detach()
        with torch.no_grad():
            final_features = problem.features_at_target(candidates)
            final_prediction = model((final_features - feature_mean) / feature_std)
        self._diagnostics = {
            "search": "gradient_ascent",
            "particle_steps": self.particle_steps,
            "logged_initializations": logged_count,
            "random_initializations": random_count,
            "predicted_standardized_utility_mean": float(final_prediction.mean().cpu()),
            "predicted_standardized_utility_max": float(final_prediction.max().cpu()),
        }
        return candidates

    def diagnostics(self) -> dict[str, float | int | str]:
        return dict(self._diagnostics)

    def _fitted_state(
        self,
    ) -> tuple[MLPSurrogate, torch.Tensor, torch.Tensor]:
        if (
            self._model is None
            or self._feature_mean is None
            or self._feature_std is None
        ):
            raise RuntimeError("fit must complete before propose")
        return self._model, self._feature_mean, self._feature_std


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
