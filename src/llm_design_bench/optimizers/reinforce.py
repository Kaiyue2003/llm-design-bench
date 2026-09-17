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
from llm_design_bench.optimizers.standard_ga import DESIGN_BASELINES_COMMIT
from llm_design_bench.optimizers.torch_utils import initialize_candidate_designs
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


@register_method()
class ReinforceMethod(OfflineBBOMethod):
    """Optimize a fixed-variance Gaussian design policy with REINFORCE."""

    metadata = MethodMetadata(
        method_id="reinforce",
        display_name="REINFORCE adaptation",
        family=MethodFamily.STANDARD,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url=(
            "https://github.com/brandontrabucco/design-baselines/"
            "tree/master/design_baselines/reinforce"
        ),
        source_commit=DESIGN_BASELINES_COMMIT,
        description=(
            "A fixed-variance Gaussian design policy trained with normalized "
            "surrogate rewards and the score-function estimator."
        ),
        adaptations=(
            "tensorflow_to_pytorch",
            "model_scale_and_training_step_context",
            "unconstrained_policy_mapped_to_simplex_or_box",
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
        iterations: int = 200,
        reinforce_batch_size: int = 2048,
        exploration_std: float = 0.1,
        reinforce_learning_rate: float = 1e-2,
        ensemble_size: int = 5,
        hidden_size: int = 256,
        num_layers: int = 1,
        surrogate_epochs: int = 100,
        batch_size: int = 100,
        surrogate_learning_rate: float = 1e-3,
        initial_min_std: float = 0.1,
        initial_max_std: float = 0.2,
        minimum_std: float = 1e-6,
    ) -> None:
        for name, value in (
            ("iterations", iterations),
            ("reinforce_batch_size", reinforce_batch_size),
            ("ensemble_size", ensemble_size),
            ("hidden_size", hidden_size),
            ("num_layers", num_layers),
            ("surrogate_epochs", surrogate_epochs),
            ("batch_size", batch_size),
        ):
            _positive_integer(name, value)
        for name, value in (
            ("exploration_std", exploration_std),
            ("reinforce_learning_rate", reinforce_learning_rate),
            ("surrogate_learning_rate", surrogate_learning_rate),
            ("initial_min_std", initial_min_std),
            ("initial_max_std", initial_max_std),
            ("minimum_std", minimum_std),
        ):
            _positive_float(name, value)
        if initial_max_std <= initial_min_std:
            raise ValueError("initial_max_std must be greater than initial_min_std")

        self.iterations = iterations
        self.reinforce_batch_size = reinforce_batch_size
        self.exploration_std = float(exploration_std)
        self.reinforce_learning_rate = float(reinforce_learning_rate)
        self.ensemble_size = ensemble_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.surrogate_epochs = surrogate_epochs
        self.batch_size = batch_size
        self.surrogate_learning_rate = float(surrogate_learning_rate)
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
            ensemble_size=self.ensemble_size,
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
        initial_parameters = problem.design_space.to_unconstrained(initial_designs)
        policy_mean = torch.nn.Parameter(initial_parameters.mean(dim=0).detach())
        optimizer = torch.optim.Adam(
            [policy_mean],
            lr=self.reinforce_learning_rate,
        )
        scale = torch.full_like(policy_mean, self.exploration_std)
        final_reward_mean = float("nan")
        final_reward_std = float("nan")

        for _ in range(self.iterations):
            noise = torch.randn(
                (self.reinforce_batch_size, problem.design_dim),
                generator=generator,
                device=context.device,
                dtype=context.dtype,
            )
            sampled_parameters = policy_mean.detach() + scale * noise
            sampled_designs = problem.design_space.from_unconstrained(
                sampled_parameters
            )
            with torch.no_grad():
                reward = ensemble.predict_standardized(problem, sampled_designs)
                reward_mean = reward.mean()
                reward_std = reward.std(unbiased=False).clamp_min(self.minimum_std)
                advantage = (reward - reward_mean) / reward_std

            distribution = torch.distributions.Independent(
                torch.distributions.Normal(policy_mean, scale),
                1,
            )
            loss = -(distribution.log_prob(sampled_parameters) * advantage).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_reward_mean = float(reward_mean.cpu())
            final_reward_std = float(reward_std.cpu())

        final_noise = torch.randn(
            (context.candidate_budget, problem.design_dim),
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        final_parameters = policy_mean.detach() + scale * final_noise
        candidates = problem.design_space.from_unconstrained(final_parameters).detach()
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
                "search": "reinforce_gaussian_policy",
                "iterations": self.iterations,
                "reinforce_batch_size": self.reinforce_batch_size,
                "exploration_std": self.exploration_std,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "final_training_reward_mean": final_reward_mean,
                "final_training_reward_std": final_reward_std,
                "predicted_standardized_utility_mean": float(prediction.mean().cpu()),
                "predicted_standardized_utility_max": float(prediction.max().cpu()),
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
