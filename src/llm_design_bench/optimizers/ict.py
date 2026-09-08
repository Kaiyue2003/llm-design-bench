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
from llm_design_bench.optimizers.ict_utils import co_teach_round
from llm_design_bench.optimizers.mentoring_utils import (
    MentoringData,
    fit_proxy_ensemble,
    sample_local_designs,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import initialize_candidate_designs
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


def _advance_designs(
    parameters: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    models: list[nn.Module],
    problem: OfflineProblem,
    data: MentoringData,
) -> None:
    features = data.at_target(
        problem, problem.design_space.from_unconstrained(parameters)
    )
    # Sum across candidates so each independent trajectory has the same step size.
    objective = torch.stack([model(features) for model in models]).mean(0).sum()
    gradient = torch.autograd.grad(-objective, parameters)[0]
    if not torch.isfinite(objective) or not torch.isfinite(gradient).all():
        raise RuntimeError("non-finite ICT design update")
    optimizer.zero_grad(set_to_none=True)
    parameters.grad = gradient
    optimizer.step()
    with torch.no_grad():
        parameters.clamp_(-20, 20)


@register_method()
class ImportanceAwareCoTeachingMethod(OfflineBBOMethod):
    """Three-proxy co-teaching, then frozen-ensemble candidate optimization."""

    metadata = MethodMetadata(
        method_id="ict",
        display_name="ICT adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url="https://github.com/mila-iqia/Importance-aware-Co-teaching",
        source_commit="6d51fcc04e7a0a60c7ad578ae4c4f743bc678ae4",
        description="Rotating teachers with small-loss exchange and meta-reweighting.",
        adaptations=(
            "independent_pytorch_implementation_from_paper",
            "functional_sgd_replaces_higher_adam",
            "raw_meta_gradient_with_weights_clipped_to_0_2",
            "fresh_teacher_labels_each_rotation",
            "best_logged_moving_anchor_then_frozen_ensemble_search",
            "model_scale_and_training_step_context",
            "constrained_coordinate_neighborhoods",
            "small_offline_dataset_validation_fallback",
            "unit_scale_for_constant_features",
            "unique_logged_and_random_initializations",
        ),
    )
    capabilities = MethodCapabilities(supports_box=True)

    def __init__(
        self,
        *,
        hidden_size: int = 2048,
        surrogate_epochs: int = 200,
        batch_size: int = 128,
        surrogate_learning_rate: float = 0.1,
        validation_fraction: float = 0.1,
        adaptation_steps: int = 100,
        solver_steps: int = 100,
        solver_learning_rate: float = 1e-3,
        neighbor_samples: int = 128,
        remember_count: int = 8,
        neighbor_noise_std: float = 0.1,
        mentoring_learning_rate: float = 1e-3,
        weight_learning_rate: float = 0.1,
        reweighting: bool = True,
        minimum_std: float = 1e-6,
    ) -> None:
        for name, value in (
            ("hidden_size", hidden_size),
            ("surrogate_epochs", surrogate_epochs),
            ("batch_size", batch_size),
            ("adaptation_steps", adaptation_steps),
            ("solver_steps", solver_steps),
            ("neighbor_samples", neighbor_samples),
            ("remember_count", remember_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if remember_count > neighbor_samples:
            raise ValueError("remember_count must not exceed neighbor_samples")
        for name, value in (
            ("surrogate_learning_rate", surrogate_learning_rate),
            ("solver_learning_rate", solver_learning_rate),
            ("neighbor_noise_std", neighbor_noise_std),
            ("mentoring_learning_rate", mentoring_learning_rate),
            ("weight_learning_rate", weight_learning_rate),
            ("minimum_std", minimum_std),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            isinstance(validation_fraction, bool)
            or not isinstance(validation_fraction, (int, float))
            or not 0 <= validation_fraction < 1
        ):
            raise ValueError("validation_fraction must be in [0, 1)")
        if not isinstance(reweighting, bool):
            raise ValueError("reweighting must be boolean")
        self.hidden_size = hidden_size
        self.surrogate_epochs = surrogate_epochs
        self.batch_size = batch_size
        self.surrogate_learning_rate = surrogate_learning_rate
        self.validation_fraction = validation_fraction
        self.adaptation_steps = adaptation_steps
        self.solver_steps = solver_steps
        self.solver_learning_rate = solver_learning_rate
        self.neighbor_samples = neighbor_samples
        self.remember_count = remember_count
        self.neighbor_noise_std = neighbor_noise_std
        self.mentoring_learning_rate = mentoring_learning_rate
        self.weight_learning_rate = weight_learning_rate
        self.reweighting = reweighting
        self.minimum_std = minimum_std

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        data = MentoringData.from_problem(problem, self.minimum_std)
        models, summary = fit_proxy_ensemble(
            data,
            context,
            generator,
            ensemble_size=3,
            hidden_size=self.hidden_size,
            epochs=self.surrogate_epochs,
            batch_size=self.batch_size,
            learning_rate=self.surrogate_learning_rate,
            validation_fraction=self.validation_fraction,
        )
        # Adapt one shared ensemble at the best logged design, regardless of K.
        anchor, _, _ = initialize_candidate_designs(
            problem,
            candidate_budget=1,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        parameters = nn.Parameter(
            problem.design_space.to_unconstrained(anchor).detach()
        )
        optimizer = torch.optim.Adam([parameters], lr=self.solver_learning_rate)
        weight_change = 0.0
        for _ in range(self.adaptation_steps):
            _advance_designs(parameters, optimizer, models, problem, data)
            neighbors = sample_local_designs(
                problem,
                parameters,
                self.neighbor_samples,
                self.neighbor_noise_std,
                generator,
            )
            features = data.at_target(problem, neighbors).detach()
            for teacher in range(3):
                weight_change += co_teach_round(
                    models,
                    teacher,
                    features,
                    data,
                    remember_count=self.remember_count,
                    learning_rate=self.mentoring_learning_rate,
                    weight_learning_rate=self.weight_learning_rate,
                    reweighting=self.reweighting,
                )
        # Final search starts afresh; no further pseudo-label or proxy updates.
        for model in models:
            model.requires_grad_(False)
        starts, logged_count, random_count = initialize_candidate_designs(
            problem,
            candidate_budget=context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        parameters = nn.Parameter(
            problem.design_space.to_unconstrained(starts).detach()
        )
        optimizer = torch.optim.Adam([parameters], lr=self.solver_learning_rate)
        for _ in range(self.solver_steps):
            _advance_designs(parameters, optimizer, models, problem, data)
        return MethodResult(
            candidates=problem.design_space.from_unconstrained(parameters).detach(),
            training_summary=summary,
            diagnostics={
                "search": "ict",
                "adaptation_steps": self.adaptation_steps,
                "solver_steps": self.solver_steps,
                "neighbor_samples": self.neighbor_samples,
                "remember_count": self.remember_count,
                "reweighting": self.reweighting,
                "teacher_rounds": self.adaptation_steps * 3,
                "mentoring_updates": self.adaptation_steps * 6,
                "sample_weight_total_absolute_change": weight_change,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "frozen_ensemble_final_search": True,
            },
        )
