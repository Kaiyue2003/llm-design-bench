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
class CMAEvolutionStrategyMethod(OfflineBBOMethod):
    """Independent full-covariance CMA-ES searches over a frozen surrogate."""

    metadata = MethodMetadata(
        method_id="cma_es",
        display_name="CMA-ES adaptation",
        family=MethodFamily.STANDARD,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url=(
            "https://github.com/brandontrabucco/design-baselines/"
            "tree/master/design_baselines/cma_es"
        ),
        source_commit=DESIGN_BASELINES_COMMIT,
        description=(
            "Full-covariance CMA-ES searches on the mean prediction of a "
            "bootstrapped probabilistic neural ensemble."
        ),
        adaptations=(
            "tensorflow_and_pycma_to_pytorch",
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
        generations: int = 100,
        sigma: float = 0.5,
        population_size: int | None = None,
        ensemble_size: int = 5,
        hidden_size: int = 256,
        num_layers: int = 1,
        surrogate_epochs: int = 100,
        batch_size: int = 100,
        surrogate_learning_rate: float = 1e-3,
        initial_min_std: float = 0.1,
        initial_max_std: float = 0.2,
        minimum_std: float = 1e-6,
        covariance_epsilon: float = 1e-6,
    ) -> None:
        _positive_integer("generations", generations)
        if population_size is not None:
            _positive_integer("population_size", population_size)
            if population_size < 2:
                raise ValueError("population_size must be at least 2")
        for name, value in (
            ("ensemble_size", ensemble_size),
            ("hidden_size", hidden_size),
            ("num_layers", num_layers),
            ("surrogate_epochs", surrogate_epochs),
            ("batch_size", batch_size),
        ):
            _positive_integer(name, value)
        for name, value in (
            ("sigma", sigma),
            ("surrogate_learning_rate", surrogate_learning_rate),
            ("initial_min_std", initial_min_std),
            ("initial_max_std", initial_max_std),
            ("minimum_std", minimum_std),
            ("covariance_epsilon", covariance_epsilon),
        ):
            _positive_float(name, value)
        if initial_max_std <= initial_min_std:
            raise ValueError("initial_max_std must be greater than initial_min_std")

        self.generations = generations
        self.sigma = float(sigma)
        self.population_size = population_size
        self.ensemble_size = ensemble_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.surrogate_epochs = surrogate_epochs
        self.batch_size = batch_size
        self.surrogate_learning_rate = float(surrogate_learning_rate)
        self.initial_min_std = float(initial_min_std)
        self.initial_max_std = float(initial_max_std)
        self.minimum_std = float(minimum_std)
        self.covariance_epsilon = float(covariance_epsilon)

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
        means = problem.design_space.to_unconstrained(initial_designs).detach()
        strategy_count, dimension = means.shape
        population_size = self.population_size or 4 + int(3 * math.log(dimension))
        parent_count = population_size // 2

        ranks = torch.arange(
            1,
            parent_count + 1,
            device=context.device,
            dtype=context.dtype,
        )
        weights = torch.log(
            torch.tensor(
                parent_count + 0.5,
                device=context.device,
                dtype=context.dtype,
            )
        ) - torch.log(ranks)
        weights = weights / weights.sum()
        effective_parents = float(1.0 / weights.square().sum())

        covariance_path_rate = (4.0 + effective_parents / dimension) / (
            dimension + 4.0 + 2.0 * effective_parents / dimension
        )
        sigma_path_rate = (effective_parents + 2.0) / (
            dimension + effective_parents + 5.0
        )
        rank_one_rate = 2.0 / ((dimension + 1.3) ** 2 + effective_parents)
        rank_mu_rate = min(
            1.0 - rank_one_rate,
            2.0
            * (effective_parents - 2.0 + 1.0 / effective_parents)
            / ((dimension + 2.0) ** 2 + effective_parents),
        )
        damping = (
            1.0
            + 2.0
            * max(0.0, math.sqrt((effective_parents - 1.0) / (dimension + 1.0)) - 1.0)
            + sigma_path_rate
        )
        expected_norm = math.sqrt(dimension) * (
            1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension**2)
        )

        covariance = (
            torch.eye(
                dimension,
                device=context.device,
                dtype=context.dtype,
            )
            .expand(strategy_count, -1, -1)
            .clone()
        )
        covariance_path = torch.zeros_like(means)
        sigma_path = torch.zeros_like(means)
        sigmas = torch.full(
            (strategy_count,),
            self.sigma,
            device=context.device,
            dtype=context.dtype,
        )

        with torch.no_grad():
            best_designs = initial_designs.clone()
            best_scores = ensemble.predict_standardized(problem, best_designs)
            for generation in range(self.generations):
                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                eigenvalues = eigenvalues.clamp(
                    min=self.covariance_epsilon,
                    max=1.0 / self.covariance_epsilon,
                )
                transform = eigenvectors @ torch.diag_embed(eigenvalues.sqrt())
                inverse_sqrt = (
                    eigenvectors
                    @ torch.diag_embed(eigenvalues.rsqrt())
                    @ eigenvectors.transpose(-1, -2)
                )
                noise = torch.randn(
                    (strategy_count, population_size, dimension),
                    generator=generator,
                    device=context.device,
                    dtype=context.dtype,
                )
                steps = torch.einsum("kpd,ked->kpe", noise, transform)
                offspring_parameters = means[:, None, :] + (
                    sigmas[:, None, None] * steps
                )
                flat_parameters = offspring_parameters.reshape(-1, dimension)
                offspring_designs = problem.design_space.from_unconstrained(
                    flat_parameters
                ).reshape(strategy_count, population_size, dimension)
                scores = ensemble.predict_standardized(
                    problem,
                    offspring_designs.reshape(-1, dimension),
                ).reshape(strategy_count, population_size)

                generation_scores, generation_indices = scores.max(dim=1)
                improved = generation_scores > best_scores
                best_scores = torch.where(improved, generation_scores, best_scores)
                best_designs[improved] = offspring_designs[
                    torch.arange(strategy_count, device=context.device),
                    generation_indices,
                ][improved]

                parent_indices = scores.topk(parent_count, dim=1).indices
                selected_steps = steps.gather(
                    1,
                    parent_indices.unsqueeze(-1).expand(-1, -1, dimension),
                )
                weighted_step = torch.einsum("m,kmd->kd", weights, selected_steps)
                means = means + sigmas[:, None] * weighted_step

                whitened_step = torch.einsum(
                    "kde,ke->kd",
                    inverse_sqrt,
                    weighted_step,
                )
                sigma_path = (1.0 - sigma_path_rate) * sigma_path + math.sqrt(
                    sigma_path_rate * (2.0 - sigma_path_rate) * effective_parents
                ) * whitened_step
                sigma_path_norm = torch.linalg.vector_norm(sigma_path, dim=1)
                path_scale = math.sqrt(
                    1.0 - (1.0 - sigma_path_rate) ** (2 * (generation + 1))
                )
                path_active = (
                    sigma_path_norm / path_scale / expected_norm
                    < 1.4 + 2.0 / (dimension + 1.0)
                ).to(context.dtype)
                covariance_path = (
                    1.0 - covariance_path_rate
                ) * covariance_path + path_active[:, None] * math.sqrt(
                    covariance_path_rate
                    * (2.0 - covariance_path_rate)
                    * effective_parents
                ) * weighted_step

                rank_one = covariance_path.unsqueeze(2) * covariance_path.unsqueeze(1)
                rank_mu = torch.einsum(
                    "m,kmi,kmj->kij",
                    weights,
                    selected_steps,
                    selected_steps,
                )
                inactive_correction = (
                    (1.0 - path_active)
                    * covariance_path_rate
                    * (2.0 - covariance_path_rate)
                )[:, None, None] * covariance
                covariance = (
                    (1.0 - rank_one_rate - rank_mu_rate) * covariance
                    + rank_one_rate * (rank_one + inactive_correction)
                    + rank_mu_rate * rank_mu
                )
                covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
                sigmas = (
                    sigmas
                    * torch.exp(
                        sigma_path_rate
                        / damping
                        * (sigma_path_norm / expected_norm - 1.0)
                    )
                ).clamp(1e-4, 10.0)

        training_summary = dict(ensemble.training_summary())
        training_summary.update(
            {
                "train_samples": problem.sample_count,
                "surrogate_epochs": self.surrogate_epochs,
            }
        )
        return MethodResult(
            candidates=best_designs.detach(),
            training_summary=training_summary,
            diagnostics={
                "search": "full_covariance_cma_es",
                "generations": self.generations,
                "population_size": population_size,
                "initial_sigma": self.sigma,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "predicted_standardized_utility_mean": float(best_scores.mean().cpu()),
                "predicted_standardized_utility_max": float(best_scores.max().cpu()),
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
