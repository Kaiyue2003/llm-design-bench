from __future__ import annotations

import math

import torch
from torch import nn
from torch.func import functional_call

from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.optimizers.probabilistic_surrogate import GaussianMLP
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.roma_utils import (
    adapt_candidate_weights,
    adversarial_weights,
    gaussian_nll,
    noisy_design_features,
    trust_region_score,
)
from llm_design_bench.optimizers.torch_utils import initialize_candidate_designs
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


@register_method()
class RobustModelAdaptationMethod(OfflineBBOMethod):
    """Robust Gaussian pretraining and candidate-specific model adaptation."""

    metadata = MethodMetadata(
        method_id="roma",
        display_name="RoMA adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url="https://github.com/sihyun-yu/RoMA",
        source_commit="08b1ab54ce3daf383e3ca3a1227d75a471afe023",
        description="Adversarial weight training and local smoothness adaptation.",
        adaptations=(
            "independent_pytorch_implementation_from_paper",
            "gaussian_nll_training_and_consistency",
            "explicit_per_tensor_relative_weight_projection",
            "candidate_specific_functional_weights_reset_each_step",
            "consistency_from_previous_target_prediction_not_logged_fidelity",
            "normalized_design_noise_without_fidelity_noise",
            "smoothness_in_constrained_design_coordinates",
            "source_style_score_space_trust_penalty",
            "all_visible_rows_final_epoch_no_validation_selection",
            "unit_scale_for_constant_features",
            "unique_logged_and_random_initializations",
        ),
    )
    capabilities = MethodCapabilities(supports_box=True)

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        surrogate_epochs: int = 50,
        batch_size: int = 128,
        surrogate_learning_rate: float = 1e-3,
        weight_perturbation_steps: int = 20,
        weight_radius: float = 5e-4,
        input_noise_std: float = 0.2,
        adaptation_steps: int = 100,
        solver_steps: int = 500,
        solver_learning_rate: float = 3e-3,
        consistency_weight: float = 1.0,
        uncertainty_weight: float = 0.0,
        region: float = 4.0,
        gradient_clip: float = 1.0,
        initial_min_std: float = 0.1,
        initial_max_std: float = 0.2,
        minimum_std: float = 1e-6,
    ) -> None:
        for name, value in (
            ("hidden_size", hidden_size),
            ("surrogate_epochs", surrogate_epochs),
            ("batch_size", batch_size),
            ("weight_perturbation_steps", weight_perturbation_steps),
            ("solver_steps", solver_steps),
            ("adaptation_steps", adaptation_steps),
        ):
            minimum = 0 if name == "adaptation_steps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("surrogate_learning_rate", surrogate_learning_rate),
            ("weight_radius", weight_radius),
            ("input_noise_std", input_noise_std),
            ("solver_learning_rate", solver_learning_rate),
            ("consistency_weight", consistency_weight),
            ("uncertainty_weight", uncertainty_weight),
            ("region", region),
            ("gradient_clip", gradient_clip),
            ("initial_min_std", initial_min_std),
            ("initial_max_std", initial_max_std),
            ("minimum_std", minimum_std),
        ):
            allow_zero = name in {
                "weight_radius",
                "input_noise_std",
                "consistency_weight",
                "uncertainty_weight",
            }
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and not allow_zero)
            ):
                requirement = "nonnegative" if allow_zero else "positive"
                raise ValueError(f"invalid {name}: must be finite and {requirement}")
        if initial_min_std > initial_max_std:
            raise ValueError("initial_min_std must not exceed initial_max_std")
        self.hidden_size = hidden_size
        self.surrogate_epochs = surrogate_epochs
        self.batch_size = batch_size
        self.surrogate_learning_rate = surrogate_learning_rate
        self.weight_perturbation_steps = weight_perturbation_steps
        self.weight_radius = weight_radius
        self.input_noise_std = input_noise_std
        self.adaptation_steps = adaptation_steps
        self.solver_steps = solver_steps
        self.solver_learning_rate = solver_learning_rate
        self.consistency_weight = consistency_weight
        self.uncertainty_weight = uncertainty_weight
        self.region = region
        self.gradient_clip = gradient_clip
        self.initial_min_std = initial_min_std
        self.initial_max_std = initial_max_std
        self.minimum_std = minimum_std

    def _fit(
        self,
        problem: OfflineProblem,
        data: MentoringData,
        context: RunContext,
        generator: torch.Generator,
    ) -> tuple[GaussianMLP, dict]:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(context.method_seed)
            model = GaussianMLP(
                data.features.shape[1],
                hidden_size=self.hidden_size,
                num_layers=2,
                initial_min_std=self.initial_min_std,
                initial_max_std=self.initial_max_std,
                activation="softplus",
            ).to(device=context.device, dtype=context.dtype)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.surrogate_learning_rate
        )
        batches, total_perturbation = 0, 0.0
        for _ in range(self.surrogate_epochs):
            order = torch.randperm(
                problem.sample_count, generator=generator, device=context.device
            )
            for indices in order.split(self.batch_size):
                features = noisy_design_features(
                    data.features[indices],
                    problem.design_dim,
                    self.input_noise_std,
                    generator,
                )
                labels = data.utility[indices]
                perturbed = adversarial_weights(
                    model,
                    features,
                    labels,
                    steps=self.weight_perturbation_steps,
                    radius=self.weight_radius,
                )
                # Hold the perturbation fixed: outer gradients flow to base weights,
                # not through adversarial optimization. Base parameters never move
                # to the adversarial point, so restoration cannot be forgotten.
                shifted = {}
                for name, p in model.named_parameters():
                    delta = (perturbed[name] - p).detach()
                    total_perturbation += float(delta.norm())
                    shifted[name] = p + delta
                loss = gaussian_nll(
                    *functional_call(model, shifted, (features,)), labels
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite RoMA pretraining loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), self.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                batches += 1
        model.eval()
        model.requires_grad_(False)
        with torch.no_grad():
            mean, logstd = model(data.features)
            nll = gaussian_nll(mean, logstd, data.utility)
            mse = (mean - data.utility).square().mean()
            if not torch.isfinite(nll) or not torch.isfinite(mse):
                raise RuntimeError("non-finite RoMA final pretraining metrics")
        return model, {
            "surrogate": "softplus_gaussian_mlp",
            "epochs": self.surrogate_epochs,
            "train_samples": problem.sample_count,
            "feature_dim": data.features.shape[1],
            "pretraining_batches": batches,
            "weight_perturbation_updates": batches * self.weight_perturbation_steps
            if self.weight_radius
            else 0,
            "total_adversarial_perturbation_norm": total_perturbation,
            "final_standardized_nll": float(nll),
            "final_standardized_mse": float(mse),
            "checkpoint_selection": "final_epoch_all_visible_rows",
        }

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        data = MentoringData.from_problem(problem, self.minimum_std)
        model, summary = self._fit(problem, data, context, generator)
        starts, logged_count, random_count = initialize_candidate_designs(
            problem,
            candidate_budget=context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        candidates, total_adaptation = [], 0.0
        base = {name: p.detach() for name, p in model.named_parameters()}
        for start in starts.split(1):
            coordinates = nn.Parameter(
                problem.design_space.to_unconstrained(start).detach()
            )
            optimizer = torch.optim.Adam([coordinates], lr=self.solver_learning_rate)
            previous = base
            with torch.no_grad():
                initial_mean, initial_logstd = model(data.at_target(problem, start))
                initial_score = initial_mean - self.uncertainty_weight * initial_logstd
            for _ in range(self.solver_steps):
                with torch.no_grad():
                    design = problem.design_space.from_unconstrained(coordinates)
                    previous_mean, _ = functional_call(
                        model, previous, (data.at_target(problem, design),)
                    )
                adapted = adapt_candidate_weights(
                    model,
                    coordinates,
                    previous_mean,
                    problem,
                    data,
                    steps=self.adaptation_steps,
                    radius=self.weight_radius,
                    consistency_weight=self.consistency_weight,
                    uncertainty_weight=self.uncertainty_weight,
                )
                total_adaptation += sum(
                    float((adapted[name] - p).norm()) for name, p in base.items()
                )
                design = problem.design_space.from_unconstrained(coordinates)
                mean, logstd = functional_call(
                    model, adapted, (data.at_target(problem, design),)
                )
                objective = trust_region_score(
                    mean,
                    logstd,
                    initial_score,
                    uncertainty_weight=self.uncertainty_weight,
                    region=self.region,
                ).sum()
                gradient = torch.autograd.grad(-objective, coordinates)[0]
                if not torch.isfinite(objective) or not torch.isfinite(gradient).all():
                    raise RuntimeError("non-finite RoMA candidate gradient")
                optimizer.zero_grad(set_to_none=True)
                coordinates.grad = gradient
                optimizer.step()
                with torch.no_grad():
                    coordinates.clamp_(-20, 20)
                previous = adapted
            candidates.append(
                problem.design_space.from_unconstrained(coordinates).detach()
            )
        return MethodResult(
            candidates=torch.cat(candidates),
            training_summary=summary,
            diagnostics={
                "search": "roma",
                "solver_steps": self.solver_steps,
                "adaptation_steps": self.adaptation_steps,
                "adaptation_updates": context.candidate_budget
                * self.solver_steps
                * self.adaptation_steps
                if self.weight_radius
                else 0,
                "total_adapted_weight_delta_norm": total_adaptation,
                "weight_radius": self.weight_radius,
                "independent_candidate_adaptation": True,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
            },
        )
