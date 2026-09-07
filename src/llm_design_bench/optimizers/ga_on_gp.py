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
from llm_design_bench.optimizers.exact_gp import fit_exact_rbf_gp
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import (
    initialize_mixed_candidate_designs,
)
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


ROOT_GP_COMMIT = "d23f14fe30d53f1fc4423ce006056672d0353906"


@register_method()
class GradientAscentOnGPMethod(OfflineBBOMethod):
    """Maximize an exact GP posterior mean by constrained gradient ascent."""

    metadata = MethodMetadata(
        method_id="ga_on_gp",
        display_name="GA on GP adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url=(
            "https://github.com/cuong-dm/ROOT/tree/main/gaussian_process"
        ),
        source_commit=ROOT_GP_COMMIT,
        description=(
            "Exact RBF Gaussian Process posterior mean optimized by plain "
            "gradient ascent from mixed logged and random starts."
        ),
        adaptations=(
            "native_pytorch_exact_gp",
            "paper_baseline_code_not_public",
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
        gp_training_steps: int = 100,
        gp_learning_rate: float = 5e-2,
        solver_steps: int = 100,
        solver_learning_rate: float = 1e-2,
        random_start_fraction: float = 0.25,
        initial_lengthscale: float = 1.0,
        initial_output_scale: float = 1.0,
        initial_noise: float = 1e-2,
        minimum_std: float = 1e-6,
        jitter: float = 1e-5,
    ) -> None:
        for name, value in (
            ("gp_training_steps", gp_training_steps),
            ("solver_steps", solver_steps),
        ):
            _positive_integer(name, value)
        for name, value in (
            ("gp_learning_rate", gp_learning_rate),
            ("solver_learning_rate", solver_learning_rate),
            ("initial_lengthscale", initial_lengthscale),
            ("initial_output_scale", initial_output_scale),
            ("initial_noise", initial_noise),
            ("minimum_std", minimum_std),
            ("jitter", jitter),
        ):
            _positive_float(name, value)
        _fraction("random_start_fraction", random_start_fraction)

        self.gp_training_steps = gp_training_steps
        self.gp_learning_rate = float(gp_learning_rate)
        self.solver_steps = solver_steps
        self.solver_learning_rate = float(solver_learning_rate)
        self.random_start_fraction = float(random_start_fraction)
        self.initial_lengthscale = float(initial_lengthscale)
        self.initial_output_scale = float(initial_output_scale)
        self.initial_noise = float(initial_noise)
        self.minimum_std = float(minimum_std)
        self.jitter = float(jitter)

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        gp = fit_exact_rbf_gp(
            problem,
            context,
            training_steps=self.gp_training_steps,
            learning_rate=self.gp_learning_rate,
            initial_lengthscale=self.initial_lengthscale,
            initial_output_scale=self.initial_output_scale,
            initial_noise=self.initial_noise,
            minimum_std=self.minimum_std,
            jitter=self.jitter,
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
        parameters = (
            problem.design_space.to_unconstrained(initial_designs)
            .detach()
            .requires_grad_(True)
        )
        step_size = self.solver_learning_rate * math.sqrt(problem.design_dim)
        with torch.no_grad():
            initial_prediction = gp.posterior_mean_standardized(
                problem,
                initial_designs,
            )
        for _ in range(self.solver_steps):
            candidates = problem.design_space.from_unconstrained(parameters)
            prediction = gp.posterior_mean_standardized(problem, candidates)
            gradient = torch.autograd.grad(prediction.sum(), parameters)[0]
            parameters = (parameters + step_size * gradient).detach().clamp(
                -20.0,
                20.0,
            ).requires_grad_(True)

        candidates = problem.design_space.from_unconstrained(parameters).detach()
        with torch.no_grad():
            final_prediction = gp.posterior_mean_standardized(problem, candidates)
        training_summary = gp.training_summary()
        training_summary["gp_training_steps"] = self.gp_training_steps
        return MethodResult(
            candidates=candidates,
            training_summary=training_summary,
            diagnostics={
                "search": "gradient_ascent_on_gp_mean",
                "solver_steps": self.solver_steps,
                "effective_step_size": step_size,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "initial_predicted_standardized_utility_mean": float(
                    initial_prediction.mean().cpu()
                ),
                "predicted_standardized_utility_mean": float(
                    final_prediction.mean().cpu()
                ),
                "predicted_standardized_utility_max": float(
                    final_prediction.max().cpu()
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


def _fraction(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be in [0, 1)")
    if not math.isfinite(float(value)) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
