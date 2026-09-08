from __future__ import annotations

import copy
import math

import torch

from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.mentoring_utils import (
    MentoringData,
    fit_proxy_ensemble,
    mentor_proxy,
    pairwise_consensus,
    sample_local_designs,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import initialize_candidate_designs
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


@register_method()
class TriMentoringMethod(OfflineBBOMethod):
    """Three-proxy pairwise mentoring with differentiable adaptive soft labels."""

    metadata = MethodMetadata(
        method_id="tri_mentoring",
        display_name="Tri-Mentoring adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url="https://github.com/GGchen1997/parallel_mentoring",
        source_commit="db1a055073cc0c22afc12c5eace55f8f030caa52",
        description="Three-proxy majority voting and bilevel soft-label mentoring.",
        adaptations=(
            "independent_pytorch_implementation_from_paper",
            "functional_sgd_replaces_higher",
            "model_scale_and_training_step_context",
            "constrained_coordinate_neighborhoods",
            "small_offline_dataset_validation_fallback",
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
        solver_steps: int = 200,
        solver_learning_rate: float = 1e-3,
        neighbor_samples: int = 10,
        neighbor_noise_std: float = 0.1,
        mentoring_learning_rate: float = 1e-3,
        label_learning_rate: float = 0.1,
        soft_labels: bool = True,
        minimum_std: float = 1e-6,
    ) -> None:
        for name, value in (
            ("hidden_size", hidden_size),
            ("surrogate_epochs", surrogate_epochs),
            ("batch_size", batch_size),
            ("solver_steps", solver_steps),
            ("neighbor_samples", neighbor_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if neighbor_samples < 2:
            raise ValueError("neighbor_samples must be at least two")
        for name, value in (
            ("surrogate_learning_rate", surrogate_learning_rate),
            ("solver_learning_rate", solver_learning_rate),
            ("neighbor_noise_std", neighbor_noise_std),
            ("mentoring_learning_rate", mentoring_learning_rate),
            ("label_learning_rate", label_learning_rate),
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
        if not isinstance(soft_labels, bool):
            raise ValueError("soft_labels must be boolean")
        self.hidden_size = hidden_size
        self.surrogate_epochs = surrogate_epochs
        self.batch_size = batch_size
        self.surrogate_learning_rate = surrogate_learning_rate
        self.validation_fraction = validation_fraction
        self.solver_steps = solver_steps
        self.solver_learning_rate = solver_learning_rate
        self.neighbor_samples = neighbor_samples
        self.neighbor_noise_std = neighbor_noise_std
        self.mentoring_learning_rate = mentoring_learning_rate
        self.label_learning_rate = label_learning_rate
        self.soft_labels = soft_labels
        self.minimum_std = minimum_std

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        data = MentoringData.from_problem(problem, self.minimum_std)
        pretrained, summary = fit_proxy_ensemble(
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
        starts, logged_count, random_count = initialize_candidate_designs(
            problem,
            candidate_budget=context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        candidates = []
        disagreements, updates, label_change = 0, 0, 0.0
        # Each candidate owns its adapted ensemble. Retain only one at a time.
        for start in starts.split(1):
            models = copy.deepcopy(pretrained)
            parameters = torch.nn.Parameter(
                problem.design_space.to_unconstrained(start).detach()
            )
            optimizer = torch.optim.Adam([parameters], lr=self.solver_learning_rate)
            for _ in range(self.solver_steps):
                neighbors = sample_local_designs(
                    problem,
                    parameters,
                    self.neighbor_samples,
                    self.neighbor_noise_std,
                    generator,
                )
                features = data.at_target(problem, neighbors).detach()
                with torch.no_grad():
                    predictions = torch.stack([model(features) for model in models])
                    pairs, consensus, masks = pairwise_consensus(predictions)
                for model, mask in zip(models, masks):
                    count = int(mask.sum())
                    if count:
                        label_change += mentor_proxy(
                            model,
                            features,
                            pairs[:, mask],
                            consensus[mask],
                            data,
                            learning_rate=self.mentoring_learning_rate,
                            label_learning_rate=self.label_learning_rate,
                            soft_labels=self.soft_labels,
                        )
                        disagreements += count
                        updates += 1
                design = problem.design_space.from_unconstrained(parameters)
                features = data.at_target(problem, design)
                objective = torch.stack([model(features) for model in models]).mean()
                # Compute only the design gradient; proxy .grad buffers stay clear.
                optimizer.zero_grad(set_to_none=True)
                parameters.grad = torch.autograd.grad(-objective, parameters)[0]
                optimizer.step()
                with torch.no_grad():
                    parameters.clamp_(-20, 20)
            candidates.append(
                problem.design_space.from_unconstrained(parameters).detach()
            )
        return MethodResult(
            candidates=torch.cat(candidates),
            training_summary=summary,
            diagnostics={
                "search": "tri_mentoring",
                "solver_steps": self.solver_steps,
                "neighbor_samples": self.neighbor_samples,
                "soft_labels": self.soft_labels,
                "mentoring_updates": updates,
                "disagreeing_pairs": disagreements,
                "soft_label_total_absolute_change": label_change,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "independent_candidate_ensembles": True,
            },
        )
