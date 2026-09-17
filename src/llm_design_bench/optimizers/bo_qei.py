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
from llm_design_bench.optimizers.exact_gp import (
    ExactRBFGaussianProcess,
    fit_exact_rbf_gp,
    stable_posterior_cholesky,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import (
    initialize_mixed_candidate_designs,
)
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


@register_method()
class BayesianOptimizationQEiMethod(OfflineBBOMethod):
    """Joint Monte Carlo qEI optimized through a GPyTorch exact GP."""

    metadata = MethodMetadata(
        method_id="bo_qei",
        display_name="BO-qEI adaptation",
        family=MethodFamily.STANDARD,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url="https://botorch.org/docs/v0.17.0/acquisition/",
        source_commit="botorch-v0.17.0-formulation",
        description=(
            "Joint Monte Carlo q-Expected Improvement over a learned exact "
            "RBF Gaussian Process implemented with GPyTorch."
        ),
        adaptations=(
            "gpytorch_exact_gp",
            "native_pytorch_joint_qei",
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
        acquisition_steps: int = 100,
        acquisition_learning_rate: float = 5e-2,
        mc_samples: int = 128,
        random_start_fraction: float = 0.25,
        initial_lengthscale: float = 1.0,
        initial_output_scale: float = 1.0,
        initial_noise: float = 1e-2,
        minimum_std: float = 1e-6,
        jitter: float = 1e-5,
    ) -> None:
        for name, value in (
            ("gp_training_steps", gp_training_steps),
            ("acquisition_steps", acquisition_steps),
            ("mc_samples", mc_samples),
        ):
            _positive_integer(name, value)
        for name, value in (
            ("gp_learning_rate", gp_learning_rate),
            ("acquisition_learning_rate", acquisition_learning_rate),
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
        self.acquisition_steps = acquisition_steps
        self.acquisition_learning_rate = float(acquisition_learning_rate)
        self.mc_samples = mc_samples
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
        gp = self._fit_gp(problem, context)
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
            lr=self.acquisition_learning_rate,
        )
        base_samples = torch.randn(
            (self.mc_samples, context.candidate_budget),
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        best_f = gp.train_targets.max()
        initial_qei, _ = _joint_qei(
            gp,
            problem,
            initial_designs,
            base_samples,
            best_f,
            jitter=self.jitter,
        )
        posterior_jitter = self.jitter
        for _ in range(self.acquisition_steps):
            candidates = problem.design_space.from_unconstrained(parameters)
            qei, posterior_jitter = _joint_qei(
                gp,
                problem,
                candidates,
                base_samples,
                best_f,
                jitter=self.jitter,
            )
            optimizer.zero_grad(set_to_none=True)
            (-qei).backward()
            optimizer.step()
            with torch.no_grad():
                parameters.clamp_(-20.0, 20.0)

        candidates = problem.design_space.from_unconstrained(parameters).detach()
        with torch.no_grad():
            final_qei, posterior_jitter = _joint_qei(
                gp,
                problem,
                candidates,
                base_samples,
                best_f,
                jitter=self.jitter,
            )
            posterior_mean, posterior_covariance = gp.posterior_standardized(
                problem,
                candidates,
            )
            posterior_std = torch.diagonal(posterior_covariance).clamp_min(0).sqrt()

        training_summary = gp.training_summary()
        training_summary["gp_training_steps"] = self.gp_training_steps
        return MethodResult(
            candidates=candidates,
            training_summary=training_summary,
            diagnostics={
                "search": "joint_monte_carlo_qei",
                "acquisition_steps": self.acquisition_steps,
                "mc_samples": self.mc_samples,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "initial_qei": float(initial_qei.detach().cpu()),
                "final_qei": float(final_qei.detach().cpu()),
                "posterior_jitter": posterior_jitter,
                "predicted_standardized_utility_mean": float(
                    posterior_mean.mean().cpu()
                ),
                "predicted_standardized_utility_max": float(posterior_mean.max().cpu()),
                "posterior_standardized_std_mean": float(posterior_std.mean().cpu()),
            },
        )

    def _fit_gp(
        self,
        problem: OfflineProblem,
        context: RunContext,
    ) -> ExactRBFGaussianProcess:
        return fit_exact_rbf_gp(
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


def _joint_qei(
    gp: ExactRBFGaussianProcess,
    problem: OfflineProblem,
    candidates: torch.Tensor,
    base_samples: torch.Tensor,
    best_f: torch.Tensor,
    *,
    jitter: float,
) -> tuple[torch.Tensor, float]:
    posterior_mean, posterior_covariance = gp.posterior_standardized(
        problem,
        candidates,
    )
    factor, used_jitter = stable_posterior_cholesky(
        posterior_covariance,
        jitter=jitter,
    )
    posterior_samples = posterior_mean.unsqueeze(0) + base_samples @ factor.T
    improvement = (posterior_samples.max(dim=1).values - best_f).clamp_min(0.0)
    return improvement.mean(), used_jitter


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
