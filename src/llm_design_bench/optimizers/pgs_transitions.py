"""PGS transition-consistency gate, not a registered optimization method.

Logged replay and future policy rollout share exactly the same projected
gradient step. No clipping of inferred actions or fabricated reward labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.problem import OfflineProblem
from llm_design_bench.spaces import BoxSpace, DesignSpace, SimplexSpace


@dataclass(frozen=True)
class PGSTransitionConfig:
    top_fraction: float = 0.2
    trajectories_per_group: int = 32
    max_horizon: int = 50
    gradient_floor: float = 1e-8
    action_margin: float = 1.05
    max_step_scale: float = 1e6
    reconstruction_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("trajectories_per_group", "max_horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "top_fraction",
            "gradient_floor",
            "action_margin",
            "max_step_scale",
            "reconstruction_tolerance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.top_fraction > 1:
            raise ValueError("top_fraction must be <= 1")
        if self.action_margin <= 1 or self.max_step_scale < 1:
            raise ValueError(
                "action_margin must exceed 1 and max_step_scale must be >= 1"
            )


@dataclass(frozen=True)
class PGSTransitionData:
    start_indices: torch.Tensor
    end_indices: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminals: torch.Tensor
    remaining_steps: torch.Tensor
    step_scale: torch.Tensor
    reconstruction_errors: torch.Tensor
    diagnostics: dict


def project_designs(values: torch.Tensor, space: DesignSpace) -> torch.Tensor:
    """Euclidean projection, preserving logged boundaries without logit nudges."""
    if (
        values.ndim != 2
        or values.shape[1] != space.dimension
        or not torch.isfinite(values).all()
    ):
        raise ValueError("projection requires a finite design matrix")
    if isinstance(space, BoxSpace):
        bounds = space.bounds.to(values)
        return values.maximum(bounds[:, 0]).minimum(bounds[:, 1])
    if isinstance(space, SimplexSpace):
        # Removing a common offset does not change simplex projection and
        # avoids cancellation for very large, nearly equal coordinates.
        shifted = values - values.max(dim=1, keepdim=True).values
        ordered = shifted.sort(dim=1, descending=True).values
        cumulative = ordered.cumsum(1) - 1
        positions = torch.arange(
            1, space.dimension + 1, device=values.device, dtype=values.dtype
        )
        active_count = (ordered > cumulative / positions).sum(1, keepdim=True)
        threshold = cumulative.gather(1, active_count - 1) / active_count
        return (shifted - threshold).clamp_min(0)
    raise TypeError("PGS transitions support only simplex and box spaces")


def projected_gradient_step(
    designs: torch.Tensor,
    gradients: torch.Tensor,
    actions: torch.Tensor,
    step_scale: torch.Tensor,
    space: DesignSpace,
) -> torch.Tensor:
    """Shared replay/rollout map: project(x + scale * action * grad_x f)."""
    space.validate(designs)
    if gradients.shape != designs.shape or actions.shape != designs.shape:
        raise ValueError("gradient and action shapes must match designs")
    if step_scale.shape != (designs.shape[1],):
        raise ValueError("step_scale must have one value per design coordinate")
    for value in (gradients, actions, step_scale):
        if value.device != designs.device or value.dtype != designs.dtype:
            raise ValueError("step tensors must share design dtype and device")
        if not torch.isfinite(value).all():
            raise ValueError("step tensors must be finite")
    if (actions.abs() > 1).any() or (step_scale <= 0).any():
        raise ValueError("actions must be in [-1, 1] and scales positive")
    result = project_designs(designs + step_scale * actions * gradients, space)
    space.validate(result)
    return result


def design_gradients(
    model: nn.Module,
    data: MentoringData,
    designs: torch.Tensor,
    contexts: torch.Tensor,
) -> torch.Tensor:
    """Differentiate a row-independent proxy in raw design units, fixing context.

    Only input gradients are returned; this does not modify model weights,
    parameter .grad buffers, or the caller's input tensors.
    """
    with torch.enable_grad():
        points = designs.detach().requires_grad_(True)
        features = (
            torch.cat([points, contexts.detach()], dim=1) - data.feature_mean
        ) / data.feature_std
        predictions = model(features)
        if (
            predictions.shape != (len(designs),)
            or not torch.isfinite(predictions).all()
        ):
            raise ValueError("PGS proxy must return one finite scalar per row")
        gradient = (
            torch.autograd.grad(predictions.sum(), points, allow_unused=True)[0]
            if predictions.requires_grad
            else None
        )
    if gradient is None:
        gradient = torch.zeros_like(designs)
    if not torch.isfinite(gradient).all():
        raise ValueError("non-finite PGS design gradients")
    return gradient.detach()


def fragment_terminals(
    retained: torch.Tensor,
    original_terminals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Break trajectories at removed edges; do not bridge missing transitions.

    Remaining horizon makes terminal semantics explicit for the future finite-
    horizon policy/critics. The next state's remaining horizon is this minus 1.
    """
    if retained.ndim != 1 or not len(retained):
        raise ValueError("retained indices must be non-empty")
    if len(retained) > 1 and not (retained[1:] > retained[:-1]).all():
        raise ValueError("retained indices must be strictly increasing")
    terminals = original_terminals[retained].clone()
    terminals[-1] = True
    terminals[:-1] |= retained[1:] != retained[:-1] + 1
    remaining = []
    steps = 0
    for terminal in reversed(terminals.detach().cpu().tolist()):
        steps = 1 if terminal else steps + 1
        remaining.append(steps)
    return terminals, torch.tensor(
        list(reversed(remaining)), device=retained.device, dtype=torch.long
    )


def build_logged_transitions(
    problem: OfflineProblem,
    model: nn.Module,
    data: MentoringData,
    generator: torch.Generator,
    *,
    config: PGSTransitionConfig | None = None,
) -> PGSTransitionData:
    """Build and certify replay transitions using only visible logged labels.

    Select top utility quantiles separately within exact fidelity groups.
    Infer coordinate-wise actions by inversion of the shared gradient step.
    Reject near-zero-gradient moves and excessive gains instead of clipping
    their actions. Fit one diagonal action scale from accepted visible edges
    and require projected reconstruction before attaching logged rewards.
    """
    config = config if config is not None else PGSTransitionConfig()
    if problem.context_dim:
        _, labels = torch.unique(problem.train_context, dim=0, return_inverse=True)
    else:
        labels = torch.zeros(
            problem.sample_count, dtype=torch.long, device=problem.train_designs.device
        )
    starts, ends, terminal_parts, pool_sizes = [], [], [], []
    skipped_groups = 0
    for label in labels.unique(sorted=True):
        indices = torch.where(labels == label)[0]
        threshold = torch.quantile(
            problem.train_utility[indices], 1 - config.top_fraction
        )
        pool = indices[problem.train_utility[indices] >= threshold]
        if len(pool) < 2:
            skipped_groups += 1
            continue
        pool_sizes.append(len(pool))
        horizon = min(config.max_horizon, len(pool) - 1)
        for _ in range(config.trajectories_per_group):
            path = pool[
                torch.randperm(len(pool), generator=generator, device=pool.device)[
                    : horizon + 1
                ]
            ]
            starts.append(path[:-1])
            ends.append(path[1:])
            terminals = torch.zeros(horizon, dtype=torch.bool, device=pool.device)
            terminals[-1] = True
            terminal_parts.append(terminals)
    if not starts:
        raise ValueError(
            "PGS has no same-fidelity top-utility pool with at least two rows; no pool expansion fallback"
        )
    left, right = torch.cat(starts), torch.cat(ends)
    original_terminals = torch.cat(terminal_parts)
    # Evaluate each used logged state once, never at a synthetic next state.
    used, inverse = left.unique(sorted=True, return_inverse=True)
    gradients = design_gradients(
        model, data, problem.train_designs[used], problem.train_context[used]
    )[inverse]
    delta = problem.train_designs[right] - problem.train_designs[left]
    usable_gradient = gradients.abs() >= config.gradient_floor
    gradient_ok = (usable_gradient | (delta == 0)).all(1)
    safe_gradient = torch.where(usable_gradient, gradients, torch.ones_like(gradients))
    gains = torch.where(usable_gradient, delta / safe_gradient, torch.zeros_like(delta))
    gain_ok = torch.isfinite(gains).all(1) & (
        gains.abs() <= config.max_step_scale / config.action_margin
    ).all(1)
    retained = torch.where(gradient_ok & gain_ok)[0]
    if not len(retained):
        raise ValueError(
            "PGS has no reconstructible transitions after gradient/gain checks"
        )
    scale = (gains[retained].abs().amax(0) * config.action_margin).clamp_min(1).detach()
    actions = (gains[retained] / scale).detach()
    reconstructed = projected_gradient_step(
        problem.train_designs[left[retained]],
        gradients[retained],
        actions,
        scale,
        problem.design_space,
    )
    errors = (reconstructed - problem.train_designs[right[retained]]).abs().amax(1)
    reconstruction_ok = errors <= config.reconstruction_tolerance
    actions, errors, retained = (
        actions[reconstruction_ok],
        errors[reconstruction_ok],
        retained[reconstruction_ok],
    )
    if not len(retained):
        raise ValueError("PGS has no transitions satisfying reconstruction tolerance")
    terminals, remaining = fragment_terminals(retained, original_terminals)
    rewards = (
        problem.train_utility[right[retained]] - problem.train_utility[left[retained]]
    )
    if not torch.isfinite(rewards).all():
        raise ValueError("non-finite PGS logged utility differences")
    return PGSTransitionData(
        start_indices=left[retained],
        end_indices=right[retained],
        actions=actions,
        rewards=rewards.detach(),
        terminals=terminals,
        remaining_steps=remaining,
        step_scale=scale,
        reconstruction_errors=errors.detach(),
        diagnostics={
            "eligible_pool_sizes": pool_sizes,
            "skipped_small_pool_groups": skipped_groups,
            "generated_transitions": len(left),
            "retained_transitions": len(retained),
            "zero_displacement_transitions": int((delta[retained] == 0).all(1).sum()),
            "rejected_gradient": int((~gradient_ok).sum()),
            "rejected_gain": int((gradient_ok & ~gain_ok).sum()),
            "rejected_reconstruction": int((~reconstruction_ok).sum()),
            "max_reconstruction_error": float(errors.max()),
            "maximum_retained_horizon": int(remaining.max()),
            "reward_units": "raw_maximization_utility_difference",
            "terminal_semantics": "finite_horizon_retained_contiguous_fragment",
            "action_mapping": "projected_raw_design_gradient_with_visible_diagonal_scale",
        },
    )
