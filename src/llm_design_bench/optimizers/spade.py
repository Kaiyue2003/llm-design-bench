"""Offline-only SPADE adaptation with constrained target-fidelity evolution."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import ClassVar

import torch

from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.optimizers.pgs_transitions import project_designs
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.spade_core import (
    ScalarDiffusion,
    SupportNeighbors,
    calibration_loss,
    lower_confidence_bound,
    proximity_loss,
)
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


def evolve_designs(
    problem: OfflineProblem,
    data: MentoringData,
    score: Callable[[torch.Tensor], torch.Tensor],
    config: SpadeMethod,
    budget: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict]:
    """Keep elites, evolve only designs, and score the final population too."""
    size = max(config.ea_pop, budget, 2)
    elite_size = min(config.ea_elite, size - 1)
    ordered = problem.train_utility.argsort(descending=True, stable=True)
    indices = ordered[torch.arange(size, device=ordered.device) % len(ordered)]
    scale = data.feature_std[: problem.design_dim]
    population = problem.train_designs[indices].clone()

    def noise(shape):
        return torch.randn(
            shape, device=population.device, dtype=population.dtype, generator=generator
        )

    population = project_designs(
        population + 0.05 * scale * noise(population.shape), problem.design_space
    )
    sigma = config.ea_mut_sigma_init
    for _ in range(config.ea_gens):
        scores = score(population)
        elites = population[scores.argsort(descending=True, stable=True)[:elite_size]]
        count = size - elite_size
        parents = torch.randint(
            elite_size, (count,), device=population.device, generator=generator
        )
        children = elites[parents].clone()
        crossovers = int(count * config.ea_crossover)
        if crossovers:
            other = torch.randint(
                elite_size, (crossovers,), device=population.device, generator=generator
            )
            alpha = torch.rand(
                (crossovers, 1),
                device=population.device,
                dtype=population.dtype,
                generator=generator,
            )
            children[:crossovers] = (
                alpha * children[:crossovers] + (1 - alpha) * elites[other]
            )
        children[crossovers:] += sigma * scale * noise(children[crossovers:].shape)
        population = project_designs(
            torch.cat([elites, children]), problem.design_space
        )
        sigma = max(sigma * 0.98, config.ea_mut_sigma_min)
    final_scores = score(population)
    candidates = population[final_scores.argsort(descending=True, stable=True)[:budget]]
    return candidates.detach(), {
        "effective_population": size,
        "effective_elites": elite_size,
        "acquisition_rows": size * (config.ea_gens + 1),
        "final_selection": "fresh_lcb_top_k_from_final_population_including_retained_elites",
        "unique_candidate_count": len(torch.unique(candidates, dim=0)),
    }


@register_method()
@dataclass(frozen=True)
class SpadeMethod(OfflineBBOMethod):
    """Compact configurable defaults, not the unpublished paper configuration."""

    diff_hidden: int = 64
    diff_t_dim: int = 32
    diff_steps: int = 100
    diff_beta_start: float = 1e-4
    diff_beta_end: float = 0.02
    diff_lr: float = 0.001
    diff_grad_clip: float = 1.0
    diff_epochs: int = 50
    diff_batch: int = 64
    calib_weight: float = 1.0
    calib_pairs: int = 32
    calib_temp: float = 1.0
    calib_mc_samples: int = 4
    calib_mc_steps: int = 10
    support_weight: float = 1.0
    support_k: int = 10
    support_tau_a: float = 0.02
    support_sigma_a0: float = 0.02
    support_sigma_a1: float = 0.005
    support_transform: bool = False
    acq_beta: float = 0.1
    acq_mc_samples: int = 32
    acq_mc_steps: int = 50
    mc_batch: int = 256
    knn_chunk: int = 256
    ea_pop: int = 128
    ea_elite: int = 64
    ea_gens: int = 50
    ea_mut_sigma_init: float = 0.12
    ea_mut_sigma_min: float = 0.02
    ea_crossover: float = 0.3
    minimum_std: float = 1e-6

    metadata: ClassVar[MethodMetadata] = MethodMetadata(
        method_id="spade",
        display_name="SPADE adaptation (official-core-derived)",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url="https://github.com/HarryYoung2018/spade",
        source_commit="586151bbb56e246f93ca97ce33f79887a13161bd",
        description="Calibrated scalar-utility diffusion, support regularization and LCB evolution.",
        adaptations=(
            "mit_attributed_official_core_with_local_generator_and_dtype_support",
            "native_chunked_joint_feature_knn_self_included_stable_ties",
            "compact_configurable_training_and_acquisition_budgets",
            "visible_only_standardization_and_final_epoch_checkpoint",
            "deterministic_ddim_only_no_unused_noise_draws",
            "project_all_proposals_before_fixed_target_acquisition",
            "population_independent_of_logged_size_with_offspring_slots",
            "final_population_rescored_no_historical_noisy_best_archive",
            "optional_support_transform_on_moments_not_rescaled_samples",
        ),
    )
    capabilities: ClassVar[MethodCapabilities] = MethodCapabilities(supports_box=True)

    def __post_init__(self) -> None:
        integers = {
            "diff_hidden": 1,
            "diff_t_dim": 2,
            "diff_steps": 2,
            "diff_epochs": 1,
            "diff_batch": 1,
            "calib_pairs": 0,
            "calib_mc_samples": 2,
            "calib_mc_steps": 2,
            "support_k": 2,
            "acq_mc_samples": 2,
            "acq_mc_steps": 2,
            "mc_batch": 1,
            "knn_chunk": 1,
            "ea_pop": 2,
            "ea_elite": 1,
            "ea_gens": 1,
        }
        zero_allowed = {
            "calib_weight",
            "support_weight",
            "support_tau_a",
            "support_sigma_a0",
            "support_sigma_a1",
            "acq_beta",
            "ea_crossover",
        }
        for item in fields(self):
            name, value = item.name, getattr(self, item.name)
            if name == "support_transform":
                if not isinstance(value, bool):
                    raise ValueError("support_transform must be bool")
            elif name in integers:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < integers[name]
                ):
                    raise ValueError(f"{name} must be an integer >= {integers[name]}")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or (value < 0 if name in zero_allowed else value <= 0)
            ):
                raise ValueError(f"invalid finite parameter {name}")
        if self.diff_t_dim % 2:
            raise ValueError("diff_t_dim must be even")
        if not 0 < self.diff_beta_start <= self.diff_beta_end < 1:
            raise ValueError("diffusion betas must be ordered in (0,1)")
        if max(self.calib_mc_steps, self.acq_mc_steps) > self.diff_steps:
            raise ValueError("MC steps cannot exceed the diffusion horizon")
        if self.ea_elite >= self.ea_pop or not 0 <= self.ea_crossover <= 1:
            raise ValueError(
                "elite must be smaller than population; crossover must be in [0,1]"
            )
        if self.ea_mut_sigma_min > self.ea_mut_sigma_init:
            raise ValueError("minimum mutation sigma cannot exceed initial sigma")

    def _fit(
        self,
        problem: OfflineProblem,
        data: MentoringData,
        neighbors: SupportNeighbors,
        context: RunContext,
        generator: torch.Generator,
    ) -> tuple[ScalarDiffusion, dict]:
        model = ScalarDiffusion(
            data.features.shape[1],
            self,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.diff_lr)
        neighbor_mean, radius = neighbors.query(data.features)
        history = {
            name: [] for name in ("diffusion", "calibration", "support", "total")
        }
        updates = 0
        for _ in range(self.diff_epochs):
            totals = {key: 0.0 for key in history}
            batches = 0
            order = torch.randperm(
                problem.sample_count, device=context.device, generator=generator
            )
            for indices in order.split(self.diff_batch):
                x, utility = data.features[indices], data.utility[indices]
                diffusion = model.loss(x, utility[:, None], generator)
                calibration = support = diffusion.new_zeros(())
                if self.calib_weight or self.support_weight:
                    samples = model.samples(
                        x,
                        count=self.calib_mc_samples,
                        steps=self.calib_mc_steps,
                        batch_size=self.mc_batch,
                        generator=generator,
                    )
                    mean, sigma = samples.mean(0), samples.std(0, unbiased=False)
                    if self.calib_weight:
                        calibration = calibration_loss(
                            mean,
                            utility,
                            pairs=self.calib_pairs,
                            temperature=self.calib_temp,
                            generator=generator,
                        )
                    if self.support_weight:
                        support = proximity_loss(
                            mean,
                            sigma,
                            neighbor_mean[indices],
                            radius[indices],
                            margin=self.support_tau_a,
                            floor=self.support_sigma_a0,
                            slope=self.support_sigma_a1,
                        )
                loss = (
                    diffusion
                    + self.calib_weight * calibration
                    + self.support_weight * support
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite SPADE training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.diff_grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                for key, value in zip(history, (diffusion, calibration, support, loss)):
                    totals[key] += float(value.detach())
                batches += 1
            for key, values in history.items():
                values.append(totals[key] / batches)
            updates += batches
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise RuntimeError("non-finite SPADE fitted parameters")
        model.eval().requires_grad_(False)
        return model, {
            "train_samples": problem.sample_count,
            "epochs": self.diff_epochs,
            "training_updates": updates,
            "loss_history": history,
            "history_reduction": "mean_of_minibatch_losses",
            "checkpoint_selection": "final_epoch_all_visible_rows",
            "effective_support_k": neighbors.k,
            "feature_mean": data.feature_mean.tolist(),
            "feature_std": data.feature_std.tolist(),
            "utility_mean": float(problem.train_utility.mean()),
            "utility_std": float(
                problem.train_utility.std(unbiased=False).clamp_min(self.minimum_std)
            ),
        }

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        if problem.sample_count < 2:
            raise ValueError("SPADE requires at least two visible observations")
        if context.dtype not in (torch.float32, torch.float64):
            raise ValueError("SPADE supports float32 and float64")
        data = MentoringData.from_problem(problem, self.minimum_std)
        neighbors = SupportNeighbors(
            data.features, data.utility, self.support_k, self.knn_chunk
        )
        model, summary = self._fit(problem, data, neighbors, context, generator)

        @torch.no_grad()
        def score(designs):
            features = data.at_target(problem, designs)
            samples = model.samples(
                features,
                count=self.acq_mc_samples,
                steps=self.acq_mc_steps,
                batch_size=self.mc_batch,
                generator=generator,
            )
            radius = neighbors.query(features)[1] if self.support_transform else None
            values = lower_confidence_bound(
                samples,
                self.acq_beta,
                radius=radius,
                margin=self.support_tau_a,
                floor=self.support_sigma_a0,
                slope=self.support_sigma_a1,
            )
            if not torch.isfinite(values).all():
                raise RuntimeError("non-finite SPADE acquisition")
            return values

        candidates, diagnostics = evolve_designs(
            problem, data, score, self, context.candidate_budget, generator
        )
        diagnostics.update(
            {
                "search_context": "fixed_target_fidelity",
                "support_geometry": "visible_standardized_joint_design_context_unit_weights",
                "support_self_included": True,
                "support_transform": self.support_transform,
                "ddim_eta": 0.0,
                "variance_correction": 0,
                "mc_utility_draws": diagnostics["acquisition_rows"]
                * self.acq_mc_samples,
                "source_commit": self.metadata.source_commit,
            }
        )
        return MethodResult(candidates, summary, diagnostics)
