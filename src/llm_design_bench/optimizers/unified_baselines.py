from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
    PreparedFitThenProposeMethod,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext
from llm_design_bench.transforms import PreparedOfflineProblem


_CONTINUOUS_CONTEXT_CAPABILITIES = MethodCapabilities(
    supports_simplex=True,
    supports_box=True,
    supports_discrete=False,
    supports_context=True,
    stochastic=True,
)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@register_method("best_logged")
class UnifiedBestLogged(OfflineBBOMethod):
    metadata = MethodMetadata(
        method_id="best_logged",
        display_name="Best Logged",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
        description="Returns the highest-utility distinct designs in visible logged data.",
        original_framework="PyTorch",
    )
    capabilities = MethodCapabilities(
        supports_simplex=True,
        supports_box=True,
        supports_discrete=False,
        supports_context=True,
        stochastic=False,
    )

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        candidates = _repeat_top_designs(
            problem.train_designs,
            problem.train_utility,
            context.candidate_budget,
        )
        return MethodResult(
            candidates=candidates,
            training_summary={"visible_logged_samples": problem.sample_count},
        )


class _GradientSurrogateMethod(PreparedFitThenProposeMethod):
    def __init__(
        self,
        *,
        hidden_size: int = 128,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        particle_steps: int = 100,
        particle_learning_rate: float = 5e-2,
        validation_fraction: float = 0.2,
    ) -> None:
        if hidden_size < 1 or epochs < 1 or batch_size < 1:
            raise ValueError("hidden_size, epochs, and batch_size must be positive")
        if learning_rate <= 0 or particle_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if particle_steps < 0:
            raise ValueError("particle_steps must be non-negative")
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.particle_steps = int(particle_steps)
        self.particle_learning_rate = float(particle_learning_rate)
        self.validation_fraction = float(validation_fraction)
        self._model: _MLP | None = None
        self._validation_mse = float("nan")

    def _new_model(
        self,
        input_dim: int,
        context: RunContext,
    ) -> _MLP:
        torch.manual_seed(context.method_seed)
        if context.device.type == "cuda":
            torch.cuda.manual_seed_all(context.method_seed)
        return _MLP(input_dim, self.hidden_size).to(
            device=context.device,
            dtype=context.dtype,
        )

    def _fit_mse(
        self,
        prepared: PreparedOfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> Mapping[str, Any]:
        features = prepared.train_features
        labels = prepared.train_utility
        model = self._new_model(features.shape[1], context)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        final_loss = float("nan")
        for _ in range(self.epochs):
            order = torch.randperm(
                len(features),
                generator=generator,
                device=features.device,
            )
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                loss = nn.functional.mse_loss(
                    model(features[indices]),
                    labels[indices],
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach())
        self._model = model
        self._validation_mse = _validation_mse(model, prepared)
        return {
            "final_train_mse": final_loss,
            "validation_mse": self._validation_mse,
            "epochs": self.epochs,
        }

    def propose_prepared(
        self,
        problem: PreparedOfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("fit_prepared must run before propose_prepared")
        seeds = _repeat_top_designs(
            problem.split.train.designs,
            problem.split.train.utility,
            context.candidate_budget,
        )
        parameters = nn.Parameter(problem.problem.design_space.to_unconstrained(seeds))
        optimizer = torch.optim.Adam(
            [parameters],
            lr=self.particle_learning_rate,
        )
        self._model.eval()
        for _ in range(self.particle_steps):
            designs = problem.problem.design_space.from_unconstrained(parameters)
            loss = -self._model(problem.features_at_target(designs)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return problem.problem.design_space.from_unconstrained(parameters).detach()

    def diagnostics(self) -> Mapping[str, Any]:
        return {"validation_mse": self._validation_mse}


@register_method("offline_mlp")
class UnifiedOfflineMLP(_GradientSurrogateMethod):
    metadata = MethodMetadata(
        method_id="offline_mlp",
        display_name="Offline MLP",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
        description="PyTorch MLP surrogate followed by gradient search in design space.",
        original_framework="PyTorch",
        adaptations=("explicit multi-fidelity context conditioning",),
    )
    capabilities = _CONTINUOUS_CONTEXT_CAPABILITIES

    def fit_prepared(
        self,
        problem: PreparedOfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> Mapping[str, Any]:
        return self._fit_mse(problem, context=context, generator=generator)


@register_method("coms")
class UnifiedCOM(_GradientSurrogateMethod):
    metadata = MethodMetadata(
        method_id="coms",
        display_name="COM",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.LIGHTWEIGHT_ADAPTATION,
        source_url="https://github.com/rail-berkeley/design-baselines",
        paper_url="https://proceedings.mlr.press/v139/trabucco21a.html",
        original_framework="TensorFlow",
        description="PyTorch conservative objective-model adaptation.",
        adaptations=(
            "explicit multi-fidelity context conditioning",
            "continuous simplex and box search spaces",
        ),
    )
    capabilities = _CONTINUOUS_CONTEXT_CAPABILITIES

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        alpha_learning_rate: float = 1e-2,
        overestimation_limit: float = 0.5,
        adversarial_steps: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if alpha <= 0 or alpha_learning_rate <= 0:
            raise ValueError("alpha and alpha_learning_rate must be positive")
        if adversarial_steps < 0:
            raise ValueError("adversarial_steps must be non-negative")
        self.alpha = float(alpha)
        self.alpha_learning_rate = float(alpha_learning_rate)
        self.overestimation_limit = float(overestimation_limit)
        self.adversarial_steps = int(adversarial_steps)
        self._final_alpha = float("nan")

    def fit_prepared(
        self,
        problem: PreparedOfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> Mapping[str, Any]:
        model = self._new_model(problem.train_features.shape[1], context)
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
        raw = problem.split.train
        final_mse = float("nan")
        final_overestimation = float("nan")
        for _ in range(self.epochs):
            order = torch.randperm(
                len(raw),
                generator=generator,
                device=raw.designs.device,
            )
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                designs = raw.designs[indices]
                contexts = raw.context[indices]
                labels = problem.transforms.transform_utility(raw.utility[indices])
                positive = problem.transforms.features(designs, contexts)
                negative = self._adversarial_features(
                    model,
                    designs,
                    contexts,
                    problem,
                )
                positive_score = model(positive)
                negative_score = model(negative)
                overestimation = negative_score - positive_score
                alpha = log_alpha.exp().clamp(max=1e6)

                alpha_loss = alpha * (
                    self.overestimation_limit - overestimation.detach().mean()
                )
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                mse = nn.functional.mse_loss(positive_score, labels)
                model_loss = mse + alpha.detach() * overestimation.mean()
                model_optimizer.zero_grad()
                model_loss.backward()
                model_optimizer.step()
                final_mse = float(mse.detach())
                final_overestimation = float(overestimation.detach().mean())

        self._model = model
        self._final_alpha = float(log_alpha.exp().detach())
        self._validation_mse = _validation_mse(model, problem)
        return {
            "final_train_mse": final_mse,
            "validation_mse": self._validation_mse,
            "final_overestimation": final_overestimation,
            "final_alpha": self._final_alpha,
            "epochs": self.epochs,
        }

    def _adversarial_features(
        self,
        model: _MLP,
        designs: torch.Tensor,
        contexts: torch.Tensor,
        problem: PreparedOfflineProblem,
    ) -> torch.Tensor:
        space = problem.problem.design_space
        parameters = space.to_unconstrained(designs).detach().requires_grad_(True)
        for _ in range(self.adversarial_steps):
            adversarial_designs = space.from_unconstrained(parameters)
            score = model(
                problem.transforms.features(adversarial_designs, contexts)
            ).sum()
            (gradient,) = torch.autograd.grad(score, parameters)
            parameters = (
                parameters + self.particle_learning_rate * gradient
            ).detach().requires_grad_(True)
        adversarial_designs = space.from_unconstrained(parameters).detach()
        return problem.transforms.features(adversarial_designs, contexts).detach()

    def diagnostics(self) -> Mapping[str, Any]:
        diagnostics = dict(super().diagnostics())
        diagnostics["final_alpha"] = self._final_alpha
        return diagnostics


@register_method("bdi")
class UnifiedBDI(PreparedFitThenProposeMethod):
    metadata = MethodMetadata(
        method_id="bdi",
        display_name="BDI",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.LIGHTWEIGHT_ADAPTATION,
        source_url="https://github.com/GGchen1997/BDI",
        paper_url=(
            "https://proceedings.neurips.cc/paper_files/paper/2022/hash/"
            "bd391cf5bdc4b63674d6da3edc1bde0d-Abstract.html"
        ),
        original_framework="JAX",
        description="PyTorch RBF-kernel adaptation of bidirectional distillation.",
        adaptations=(
            "finite RBF kernel replaces the official infinite-width NTK",
            "explicit multi-fidelity context conditioning",
        ),
    )
    capabilities = _CONTINUOUS_CONTEXT_CAPABILITIES

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
        max_support_points: int = 512,
        validation_fraction: float = 0.2,
    ) -> None:
        if steps < 0 or learning_rate <= 0 or ridge <= 0:
            raise ValueError("steps must be non-negative and rates must be positive")
        if lengthscale is not None and lengthscale <= 0:
            raise ValueError("lengthscale must be positive")
        if max_support_points < 2:
            raise ValueError("max_support_points must be at least two")
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.lengthscale = lengthscale
        self.ridge = float(ridge)
        self.label = float(label)
        self.gamma = float(gamma)
        self.forward_weight = float(forward_weight)
        self.distillation_weight = float(distillation_weight)
        self.max_support_points = int(max_support_points)
        self.validation_fraction = float(validation_fraction)
        self._features: torch.Tensor | None = None
        self._utility: torch.Tensor | None = None
        self._coefficients: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None
        self._lengthscale = float("nan")

    def fit_prepared(
        self,
        problem: PreparedOfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> Mapping[str, Any]:
        features = problem.train_features
        utility = problem.train_utility
        if len(features) > self.max_support_points:
            indices = torch.randperm(
                len(features),
                generator=generator,
                device=features.device,
            )[: self.max_support_points]
            features = features[indices]
            utility = utility[indices]
        self._lengthscale = self.lengthscale or _median_lengthscale(features)
        identity = torch.eye(
            len(features),
            device=features.device,
            dtype=features.dtype,
        )
        kernel = _rbf_kernel(features, features, self._lengthscale)
        coefficients = torch.linalg.solve(
            kernel + self.ridge * identity,
            utility,
        ).detach()
        prediction = kernel @ coefficients
        self._features = features.detach()
        self._utility = utility.detach()
        self._coefficients = coefficients
        self._weights = torch.softmax(self.gamma * utility, dim=0).detach()
        return {
            "support_points": len(features),
            "lengthscale": self._lengthscale,
            "support_train_mse": float(
                nn.functional.mse_loss(prediction, utility).detach()
            ),
        }

    def propose_prepared(
        self,
        problem: PreparedOfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if any(
            value is None
            for value in (
                self._features,
                self._utility,
                self._coefficients,
                self._weights,
            )
        ):
            raise RuntimeError("fit_prepared must run before propose_prepared")
        features = self._features
        utility = self._utility
        coefficients = self._coefficients
        weights = self._weights
        assert features is not None
        assert utility is not None
        assert coefficients is not None
        assert weights is not None

        seeds = _repeat_top_designs(
            problem.split.train.designs,
            problem.split.train.utility,
            context.candidate_budget,
        )
        space = problem.problem.design_space
        parameters = nn.Parameter(space.to_unconstrained(seeds))
        optimizer = torch.optim.Adam([parameters], lr=self.learning_rate)
        for _ in range(self.steps):
            designs = space.from_unconstrained(parameters)
            support = problem.features_at_target(designs)
            forward_score = (
                _rbf_kernel(support, features, self._lengthscale) @ coefficients
            )
            support_labels = torch.full_like(forward_score, self.label)
            identity = torch.eye(
                len(support),
                device=support.device,
                dtype=support.dtype,
            )
            distilled_coefficients = torch.linalg.solve(
                _rbf_kernel(support, support, self._lengthscale)
                + self.ridge * identity,
                support_labels,
            )
            reconstructed = (
                _rbf_kernel(features, support, self._lengthscale)
                @ distilled_coefficients
            )
            distillation_loss = torch.sum(
                weights * (reconstructed - utility).square()
            )
            loss = -self.forward_weight * forward_score.mean()
            loss = loss + self.distillation_weight * distillation_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return space.from_unconstrained(parameters).detach()


def _repeat_top_designs(
    designs: torch.Tensor,
    utility: torch.Tensor,
    count: int,
) -> torch.Tensor:
    order = torch.argsort(utility, descending=True, stable=True)
    selected: list[torch.Tensor] = []
    seen: set[tuple[float, ...]] = set()
    for index in order.tolist():
        design = designs[index]
        key = tuple(round(float(value), 12) for value in design.detach().cpu())
        if key not in seen:
            seen.add(key)
            selected.append(design)
        if len(selected) == count:
            break
    if not selected:
        raise ValueError("the logged dataset does not contain any designs")
    stacked = torch.stack(selected)
    repeats = math.ceil(count / len(stacked))
    return stacked.repeat((repeats, 1))[:count].clone()


def _validation_mse(
    model: nn.Module,
    problem: PreparedOfflineProblem,
) -> float:
    if len(problem.split.validation) == 0:
        return float("nan")
    model.eval()
    with torch.no_grad():
        return float(
            nn.functional.mse_loss(
                model(problem.validation_features),
                problem.validation_utility,
            )
        )


def _median_lengthscale(features: torch.Tensor) -> float:
    distances = torch.pdist(features.detach())
    nonzero = distances[distances > 0]
    if len(nonzero) == 0:
        return 1.0
    return float(nonzero.median().clamp_min(1e-3))


def _rbf_kernel(
    left: torch.Tensor,
    right: torch.Tensor,
    lengthscale: float,
) -> torch.Tensor:
    squared_distance = torch.cdist(left, right).square()
    return torch.exp(-0.5 * squared_distance / (lengthscale**2))
