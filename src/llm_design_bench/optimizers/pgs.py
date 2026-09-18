"""Offline policy-guided gradient search with certified replay transitions."""

from __future__ import annotations

import math
from dataclasses import asdict

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
)
from llm_design_bench.optimizers.pgs_sac import ConservativeSAC, SACBatch
from llm_design_bench.optimizers.pgs_transitions import (
    PGSTransitionConfig,
    build_logged_transitions,
    design_gradients,
    projected_gradient_step,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import select_top_unique_designs
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


def policy_states(data, designs, contexts, remaining_steps, max_horizon):
    """Identical normalized design/context/horizon representation in fit and rollout."""
    if (
        remaining_steps.shape != (len(designs),)
        or (remaining_steps < 0).any()
        or (remaining_steps > max_horizon).any()
    ):
        raise ValueError("remaining steps must be a valid per-state horizon vector")
    features = (
        torch.cat([designs, contexts], dim=1) - data.feature_mean
    ) / data.feature_std
    return torch.cat(
        [features, (remaining_steps.to(features.dtype) / max_horizon).unsqueeze(1)],
        dim=1,
    )


@register_method()
class PolicyGuidedSearchMethod(OfflineBBOMethod):
    metadata = MethodMetadata(
        method_id="pgs",
        display_name="PGS adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url="https://github.com/yassineCh/PGS",
        source_commit="54837299b33f986563b15176695e2c83472ffdda",
        description="CQL/SAC offline policy chooses coordinate-wise surrogate gradient steps.",
        adaptations=(
            "independent_pytorch_implementation_from_paper",
            "compact_configurable_surrogate_and_rl_budgets",
            "all_visible_rows_cosine_schedule_final_epoch_surrogate",
            "exact_same_fidelity_top_quantile_trajectories",
            "certified_projected_raw_gradient_actions_with_visible_diagonal_scale",
            "filtered_contiguous_fragments_with_explicit_remaining_horizon",
            "visible_utility_std_reward_scaling_without_reward_centering",
            "native_importance_corrected_cql_twin_critic_sac",
            "deterministic_tanh_mean_policy_at_fixed_target_fidelity",
            "rollout_horizon_bounded_by_retained_replay_support",
            "boundary_preserving_unique_logged_and_random_initializations",
            "no_oracle_checkpoint_selection_or_invalid_replay_fallback",
        ),
    )
    capabilities = MethodCapabilities(supports_box=True)

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        surrogate_epochs: int = 50,
        surrogate_learning_rate: float = 3e-4,
        batch_size: int = 128,
        rl_steps: int = 1000,
        rl_learning_rate: float = 3e-4,
        discount: float = 0.99,
        target_rate: float = 0.005,
        cql_weight: float = 5.0,
        cql_samples: int = 10,
        cql_temperature: float = 1.0,
        initial_alpha: float = 1.0,
        automatic_entropy: bool = True,
        backup_entropy: bool = False,
        top_fraction: float = 0.2,
        trajectories_per_group: int = 32,
        max_horizon: int = 50,
        gradient_floor: float = 1e-8,
        action_margin: float = 1.05,
        max_step_scale: float = 1e6,
        reconstruction_tolerance: float = 1e-5,
        solver_steps: int = 50,
        minimum_std: float = 1e-6,
    ) -> None:
        for name, value in (
            ("hidden_size", hidden_size),
            ("surrogate_epochs", surrogate_epochs),
            ("batch_size", batch_size),
            ("rl_steps", rl_steps),
            ("cql_samples", cql_samples),
            ("solver_steps", solver_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("surrogate_learning_rate", surrogate_learning_rate),
            ("rl_learning_rate", rl_learning_rate),
            ("discount", discount),
            ("target_rate", target_rate),
            ("cql_weight", cql_weight),
            ("cql_temperature", cql_temperature),
            ("initial_alpha", initial_alpha),
            ("minimum_std", minimum_std),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and name != "discount")
            ):
                raise ValueError(f"invalid {name}")
        if discount > 1 or target_rate > 1:
            raise ValueError("discount and target_rate must not exceed one")
        if not isinstance(automatic_entropy, bool) or not isinstance(
            backup_entropy, bool
        ):
            raise TypeError("entropy switches must be booleans")
        self.transition_config = PGSTransitionConfig(
            top_fraction=top_fraction,
            trajectories_per_group=trajectories_per_group,
            max_horizon=max_horizon,
            gradient_floor=gradient_floor,
            action_margin=action_margin,
            max_step_scale=max_step_scale,
            reconstruction_tolerance=reconstruction_tolerance,
        )
        for name, value in asdict(self.transition_config).items():
            setattr(self, name, value)
        self.hidden_size, self.surrogate_epochs = hidden_size, surrogate_epochs
        self.surrogate_learning_rate, self.batch_size = (
            surrogate_learning_rate,
            batch_size,
        )
        self.rl_steps, self.rl_learning_rate = rl_steps, rl_learning_rate
        self.discount, self.target_rate = discount, target_rate
        self.cql_weight, self.cql_samples, self.cql_temperature = (
            cql_weight,
            cql_samples,
            cql_temperature,
        )
        self.initial_alpha, self.automatic_entropy, self.backup_entropy = (
            initial_alpha,
            automatic_entropy,
            backup_entropy,
        )
        self.solver_steps, self.minimum_std = solver_steps, minimum_std

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        if context.dtype not in (torch.float32, torch.float64):
            raise ValueError("PGS supports float32 and float64 replay construction")
        data = MentoringData.from_problem(problem, self.minimum_std)
        proxies, surrogate_summary = fit_proxy_ensemble(
            data,
            context,
            generator,
            ensemble_size=1,
            hidden_size=self.hidden_size,
            epochs=self.surrogate_epochs,
            batch_size=self.batch_size,
            learning_rate=self.surrogate_learning_rate,
            validation_fraction=0.0,
        )
        proxy = proxies[0].eval().requires_grad_(False)
        for parameter in proxy.parameters():
            parameter.grad = None
            if not torch.isfinite(parameter).all():
                raise RuntimeError("non-finite fitted PGS surrogate")
        replay = build_logged_transitions(
            problem, proxy, data, generator, config=self.transition_config
        )
        left, right = replay.start_indices, replay.end_indices
        states = policy_states(
            data,
            problem.train_designs[left],
            problem.train_context[left],
            replay.remaining_steps,
            self.max_horizon,
        ).detach()
        next_states = policy_states(
            data,
            problem.train_designs[right],
            problem.train_context[right],
            replay.remaining_steps - 1,
            self.max_horizon,
        ).detach()
        reward_scale = (
            problem.train_utility.std(unbiased=False)
            .clamp_min(self.minimum_std)
            .detach()
        )
        rewards = replay.rewards / reward_scale
        learner = ConservativeSAC(
            states.shape[1],
            problem.design_dim,
            context,
            generator,
            hidden_size=self.hidden_size,
            learning_rate=self.rl_learning_rate,
            discount=self.discount,
            target_rate=self.target_rate,
            cql_weight=self.cql_weight,
            cql_samples=self.cql_samples,
            cql_temperature=self.cql_temperature,
            initial_alpha=self.initial_alpha,
            automatic_entropy=self.automatic_entropy,
            backup_entropy=self.backup_entropy,
        )
        first_metrics = None
        for _ in range(self.rl_steps):
            indices = torch.randint(
                len(left),
                (min(self.batch_size, len(left)),),
                device=context.device,
                generator=generator,
            )
            metrics = learner.update(
                SACBatch(
                    states[indices],
                    replay.actions[indices],
                    rewards[indices],
                    next_states[indices],
                    replay.terminals[indices],
                )
            )
            if first_metrics is None:
                first_metrics = dict(metrics)
        learner.actor.eval().requires_grad_(False)
        candidates = select_top_unique_designs(
            problem.train_designs, problem.train_utility, context.candidate_budget
        )
        logged_count = len(candidates)
        random_count = context.candidate_budget - logged_count
        if random_count:
            candidates = torch.cat(
                [
                    candidates,
                    problem.design_space.sample(
                        random_count,
                        generator=generator,
                        device=context.device,
                        dtype=context.dtype,
                    ),
                ]
            )
        horizon = min(self.solver_steps, int(replay.remaining_steps.max()))
        target = problem.target_context.unsqueeze(0).expand(
            context.candidate_budget, -1
        )
        for remaining in range(horizon, 0, -1):
            with torch.no_grad():
                state = policy_states(
                    data,
                    candidates,
                    target,
                    torch.full((len(candidates),), remaining, device=context.device),
                    self.max_horizon,
                )
                actions = learner.actor.mode(state)
            gradients = design_gradients(proxy, data, candidates, target)
            candidates = projected_gradient_step(
                candidates, gradients, actions, replay.step_scale, problem.design_space
            ).detach()
        return MethodResult(
            candidates=candidates,
            training_summary={
                "surrogate": surrogate_summary,
                "rl_updates": self.rl_steps,
                "state_dim": states.shape[1],
                "action_dim": problem.design_dim,
                "reward_divisor": float(reward_scale),
                "reward_scaling": "visible_utility_std_no_centering",
                "first_rl_metrics": first_metrics,
                "final_rl_metrics": metrics,
                "target_entropy": learner.target_entropy,
                "backup_entropy": self.backup_entropy,
                "checkpoint_selection": "final_rl_update_no_oracle",
            },
            diagnostics={
                **replay.diagnostics,
                "step_scale": replay.step_scale.cpu().tolist(),
                "transition_config": asdict(self.transition_config),
                "solver_steps_requested": self.solver_steps,
                "solver_steps_used": horizon,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "search_context": "fixed_target_fidelity",
                "rollout_action": "tanh_gaussian_mean",
            },
        )
