from __future__ import annotations

import math

import torch
from torch import nn

from llm_design_bench.optimizers.base import (
    FitThenProposeMethod,
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
)
from llm_design_bench.optimizers.offline_utils import (
    MLPSurrogate,
    design_parameters,
    finalize_offline_trace,
    offline_data,
    parameters_to_designs,
    set_seed,
    target_features,
    unique_top_mixtures,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import (
    initialize_candidate_designs,
    move_designs_to_interior,
)
from llm_design_bench.problem import OfflineProblem, RunContext


class ConservativeObjectiveModelOptimizer:
    def __init__(
        self,
        recommendations: int = 128,
        seed: int = 0,
        hidden_size: int = 128,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        alpha: float = 0.1,
        alpha_learning_rate: float = 1e-2,
        overestimation_limit: float = 0.5,
        adversarial_steps: int = 20,
        particle_steps: int = 100,
        particle_learning_rate: float = 5e-2,
        device: str = "cpu",
    ) -> None:
        self.recommendations = recommendations
        self.seed = seed
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.alpha_learning_rate = alpha_learning_rate
        self.overestimation_limit = overestimation_limit
        self.adversarial_steps = adversarial_steps
        self.particle_steps = particle_steps
        self.particle_learning_rate = particle_learning_rate
        self.device = device

    def optimize(self, task):
        set_seed(self.seed)
        data = offline_data(task, device=self.device)
        model = MLPSurrogate(data.features.shape[1], hidden_size=self.hidden_size).to(self.device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        log_alpha = nn.Parameter(
            torch.tensor(self.alpha, dtype=torch.float32, device=self.device).log()
        )
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=self.alpha_learning_rate)

        for _ in range(self.epochs):
            order = torch.randperm(len(data.features), device=self.device)
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                positive = data.features[indices]
                labels = data.utility[indices]
                negative = self._adversarial_features(model, positive, task)
                positive_score = model(positive)
                negative_score = model(negative)
                overestimation = negative_score - positive_score
                alpha = log_alpha.exp().clamp(max=1e6)

                alpha_loss = alpha * (self.overestimation_limit - overestimation.detach().mean())
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                model_loss = nn.functional.mse_loss(positive_score, labels)
                model_loss = model_loss + alpha.detach() * overestimation.mean()
                model_optimizer.zero_grad()
                model_loss.backward()
                model_optimizer.step()

        parameters = nn.Parameter(
            design_parameters(
                unique_top_mixtures(task, self.recommendations),
                task,
                device=self.device,
            )
        )
        particle_optimizer = torch.optim.Adam([parameters], lr=self.particle_learning_rate)
        for _ in range(self.particle_steps):
            mixtures = parameters_to_designs(parameters, task)
            loss = -model(target_features(mixtures, task)).mean()
            particle_optimizer.zero_grad()
            loss.backward()
            particle_optimizer.step()

        return finalize_offline_trace("coms", task, parameters_to_designs(parameters, task))

    def _adversarial_features(self, model, features: torch.Tensor, task) -> torch.Tensor:
        parameters = design_parameters(features[:, : task.mixture_dim], task, device=self.device)
        parameters.requires_grad_(True)
        fidelity = features[:, task.mixture_dim :].detach()
        for _ in range(self.adversarial_steps):
            mixtures = parameters_to_designs(parameters, task)
            score = model(torch.cat([mixtures, fidelity], dim=1)).sum()
            (gradient,) = torch.autograd.grad(score, parameters)
            parameters = (parameters + self.particle_learning_rate * gradient).detach()
            parameters.requires_grad_(True)
        return torch.cat([parameters_to_designs(parameters, task), fidelity], dim=1).detach()


@register_method()
class ConservativeObjectiveModelMethod(FitThenProposeMethod):
    """Compact PyTorch COMs adaptation behind the strict offline API."""

    metadata = MethodMetadata(
        method_id="coms",
        display_name="COMs adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.LIGHTWEIGHT_ADAPTATION,
        source_url="https://github.com/brandontrabucco/design-baselines",
        description=(
            "Conservative objective surrogate with adversarial negative designs "
            "and target-fidelity gradient search."
        ),
        adaptations=(
            "compact_pytorch_reimplementation",
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
        hidden_size: int = 128,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        alpha: float = 0.1,
        alpha_learning_rate: float = 1e-2,
        overestimation_limit: float = 0.5,
        adversarial_steps: int = 20,
        particle_steps: int = 100,
        particle_learning_rate: float = 5e-2,
        minimum_std: float = 1e-6,
    ) -> None:
        _validate_positive_integer("hidden_size", hidden_size)
        _validate_positive_integer("epochs", epochs)
        _validate_positive_integer("batch_size", batch_size)
        _validate_positive_integer("adversarial_steps", adversarial_steps)
        _validate_positive_integer("particle_steps", particle_steps)
        _validate_positive_float("learning_rate", learning_rate)
        _validate_positive_float("alpha", alpha)
        _validate_positive_float("alpha_learning_rate", alpha_learning_rate)
        _validate_finite_float("overestimation_limit", overestimation_limit)
        _validate_positive_float("particle_learning_rate", particle_learning_rate)
        _validate_positive_float("minimum_std", minimum_std)

        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.alpha_learning_rate = alpha_learning_rate
        self.overestimation_limit = overestimation_limit
        self.adversarial_steps = adversarial_steps
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
        feature_std = features.std(dim=0, unbiased=False).clamp_min(
            self.minimum_std
        )
        normalized_features = (features - feature_mean) / feature_std
        standardized_utility, utility_mean, utility_std = (
            problem.standardized_utility(self.minimum_std)
        )

        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(context.method_seed)
            model = MLPSurrogate(
                normalized_features.shape[1],
                hidden_size=self.hidden_size,
            ).to(device=context.device, dtype=context.dtype)

        model_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
        )
        log_alpha = nn.Parameter(
            torch.tensor(
                math.log(self.alpha),
                device=context.device,
                dtype=context.dtype,
            )
        )
        alpha_optimizer = torch.optim.Adam(
            [log_alpha],
            lr=self.alpha_learning_rate,
        )

        model.train()
        last_overestimation = torch.tensor(
            float("nan"),
            device=context.device,
            dtype=context.dtype,
        )
        for _ in range(self.epochs):
            order = torch.randperm(
                problem.sample_count,
                generator=generator,
                device=context.device,
            )
            for start in range(0, problem.sample_count, self.batch_size):
                indices = order[start : start + self.batch_size]
                positive = normalized_features[indices]
                labels = standardized_utility[indices]
                negative = self._adversarial_features(
                    model,
                    problem,
                    features[indices],
                    feature_mean,
                    feature_std,
                )
                positive_score = model(positive)
                negative_score = model(negative)
                overestimation = negative_score - positive_score
                multiplier = log_alpha.exp().clamp(max=1e6)

                alpha_loss = multiplier * (
                    self.overestimation_limit - overestimation.detach().mean()
                )
                alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_optimizer.step()

                model_loss = torch.nn.functional.mse_loss(
                    positive_score,
                    labels,
                )
                model_loss = model_loss + (
                    multiplier.detach() * overestimation.mean()
                )
                model_optimizer.zero_grad(set_to_none=True)
                model_loss.backward()
                model_optimizer.step()
                last_overestimation = overestimation.detach().mean()

        model.eval()
        with torch.no_grad():
            final_mse = torch.nn.functional.mse_loss(
                model(normalized_features),
                standardized_utility,
            )
            final_alpha = log_alpha.exp().clamp(max=1e6)
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
            "final_standardized_mse": float(final_mse.cpu()),
            "final_alpha": float(final_alpha.cpu()),
            "last_batch_overestimation": float(last_overestimation.cpu()),
            "utility_mean": float(utility_mean.cpu()),
            "utility_std": float(utility_std.cpu()),
        }

    def propose(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> torch.Tensor:
        model, feature_mean, feature_std = self._fitted_state()
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
        optimizer = torch.optim.Adam(
            [parameters],
            lr=self.particle_learning_rate,
        )
        for _ in range(self.particle_steps):
            candidates = problem.design_space.from_unconstrained(parameters)
            normalized = (
                problem.features_at_target(candidates) - feature_mean
            ) / feature_std
            loss = -model(normalized).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        candidates = problem.design_space.from_unconstrained(parameters).detach()
        with torch.no_grad():
            prediction = model(
                (problem.features_at_target(candidates) - feature_mean)
                / feature_std
            )
        self._diagnostics = {
            "search": "conservative_surrogate_gradient_ascent",
            "particle_steps": self.particle_steps,
            "logged_initializations": logged_count,
            "random_initializations": random_count,
            "predicted_standardized_utility_mean": float(prediction.mean().cpu()),
            "predicted_standardized_utility_max": float(prediction.max().cpu()),
        }
        return candidates

    def diagnostics(self) -> dict[str, float | int | str]:
        return dict(self._diagnostics)

    def _adversarial_features(
        self,
        model: MLPSurrogate,
        problem: OfflineProblem,
        raw_features: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> torch.Tensor:
        designs = move_designs_to_interior(
            raw_features[:, : problem.design_dim],
            problem.design_space,
        )
        fidelity = raw_features[:, problem.design_dim :].detach()
        parameters = problem.design_space.to_unconstrained(designs).detach()
        for _ in range(self.adversarial_steps):
            parameters.requires_grad_(True)
            adversarial = problem.design_space.from_unconstrained(parameters)
            normalized = (
                torch.cat([adversarial, fidelity], dim=1) - feature_mean
            ) / feature_std
            (gradient,) = torch.autograd.grad(model(normalized).sum(), parameters)
            parameters = (
                parameters + self.particle_learning_rate * gradient
            ).detach()
        adversarial = problem.design_space.from_unconstrained(parameters)
        return (
            (torch.cat([adversarial, fidelity], dim=1) - feature_mean)
            / feature_std
        ).detach()

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
    _validate_finite_float(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _validate_finite_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
