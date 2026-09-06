from __future__ import annotations

import math

import torch

from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.probabilistic_surrogate import (
    fit_gaussian_ensemble,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import initialize_candidate_designs
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


DESIGN_BASELINES_COMMIT = "785dbcfa58107bfcc426257a1c2e69d7f71c3c27"


@register_method()
class StandardGradientAscentMethod(OfflineBBOMethod):
    """Optimize a probabilistic forward model's mean by plain gradient ascent."""

    metadata = MethodMetadata(
        method_id="standard_ga",
        display_name="Standard GA adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url=(
            "https://github.com/brandontrabucco/design-baselines/"
            "tree/master/design_baselines/gradient_ascent"
        ),
        source_commit=DESIGN_BASELINES_COMMIT,
        description=(
            "A probabilistic neural forward model whose mean prediction is "
            "optimized from high-utility logged designs by plain gradient ascent."
        ),
        adaptations=(
            "tensorflow_to_pytorch",
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
        hidden_size: int = 2048,
        num_layers: int = 2,
        surrogate_epochs: int = 100,
        batch_size: int = 128,
        surrogate_learning_rate: float = 3e-4,
        solver_steps: int = 200,
        solver_learning_rate: float = 1e-2,
        initial_min_std: float = 0.1,
        initial_max_std: float = 0.2,
        minimum_std: float = 1e-6,
    ) -> None:
        for name, value in (
            ("hidden_size", hidden_size),
            ("num_layers", num_layers),
            ("surrogate_epochs", surrogate_epochs),
            ("batch_size", batch_size),
            ("solver_steps", solver_steps),
        ):
            _positive_integer(name, value)
        for name, value in (
            ("surrogate_learning_rate", surrogate_learning_rate),
            ("solver_learning_rate", solver_learning_rate),
            ("initial_min_std", initial_min_std),
            ("initial_max_std", initial_max_std),
            ("minimum_std", minimum_std),
        ):
            _positive_float(name, value)
        if initial_max_std <= initial_min_std:
            raise ValueError("initial_max_std must be greater than initial_min_std")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.surrogate_epochs = surrogate_epochs
        self.batch_size = batch_size
        self.surrogate_learning_rate = float(surrogate_learning_rate)
        self.solver_steps = solver_steps
        self.solver_learning_rate = float(solver_learning_rate)
        self.initial_min_std = float(initial_min_std)
        self.initial_max_std = float(initial_max_std)
        self.minimum_std = float(minimum_std)

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        ensemble = fit_gaussian_ensemble(
            problem,
            context,
            generator,
            ensemble_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            epochs=self.surrogate_epochs,
            batch_size=self.batch_size,
            learning_rate=self.surrogate_learning_rate,
            initial_min_std=self.initial_min_std,
            initial_max_std=self.initial_max_std,
            minimum_std=self.minimum_std,
        )
        initial_designs, logged_count, random_count = initialize_candidate_designs(
            problem,
            candidate_budget=context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        parameters = (
            problem.design_space.to_unconstrained(initial_designs)
            .detach()
            .requires_grad_(True)
        )
        step_size = self.solver_learning_rate * math.sqrt(problem.design_dim)
        for _ in range(self.solver_steps):
            candidates = problem.design_space.from_unconstrained(parameters)
            prediction = ensemble.predict_standardized(problem, candidates)
            gradient = torch.autograd.grad(prediction.sum(), parameters)[0]
            parameters = (parameters + step_size * gradient).detach().requires_grad_(
                True
            )

        candidates = problem.design_space.from_unconstrained(parameters).detach()
        with torch.no_grad():
            prediction = ensemble.predict_standardized(problem, candidates)
        training_summary = dict(ensemble.training_summary())
        training_summary.update(
            {
                "train_samples": problem.sample_count,
                "surrogate_epochs": self.surrogate_epochs,
            }
        )
        return MethodResult(
            candidates=candidates,
            training_summary=training_summary,
            diagnostics={
                "search": "plain_gradient_ascent",
                "solver_steps": self.solver_steps,
                "effective_step_size": step_size,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "predicted_standardized_utility_mean": float(
                    prediction.mean().cpu()
                ),
                "predicted_standardized_utility_max": float(
                    prediction.max().cpu()
                ),
            },
        )


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
