"""Independent gradient-matching adaptation; no research runtime imports."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import initialize_candidate_designs
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


class MatchingMLP(nn.Module):
    """Row-independent scalar model, required by summed-output differentiation."""

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        widths = (input_dim, 16 * embedding_dim, 4 * embedding_dim, embedding_dim)
        layers: list[nn.Module] = []
        for before, after in pairwise(widths):
            layers.extend([nn.Linear(before, after), nn.LeakyReLU(0.3)])
        self.network = nn.Sequential(*layers, nn.Linear(embedding_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def fidelity_buckets(
    problem: OfflineProblem, bucket_count: int
) -> list[tuple[torch.Tensor, ...]]:
    """Partition each exact logged-context group into ordered utility buckets.

    Singleton contexts are excluded only from gradient matching, not regression.
    The last bucket receives the remainder, matching the source partition rule.
    """
    if bucket_count < 2:
        raise ValueError("bucket_count must be >= 2")
    if problem.context_dim:
        _, labels = torch.unique(problem.train_context, dim=0, return_inverse=True)
    else:
        labels = torch.zeros(
            problem.sample_count, dtype=torch.long, device=problem.train_designs.device
        )
    groups = []
    for label in labels.unique(sorted=True):
        indices = torch.where(labels == label)[0]
        if len(indices) < 2:
            continue
        ordered = indices[torch.argsort(problem.train_utility[indices], stable=True)]
        count = min(bucket_count, len(ordered))
        width = len(ordered) // count
        groups.append(
            tuple(
                ordered[i * width : (i + 1) * width]
                if i < count - 1
                else ordered[i * width :]
                for i in range(count)
            )
        )
    if not groups:
        raise ValueError(
            "MATCH-OPT requires at least two visible observations at the same fidelity; "
            "no eligible pairs (plain MSE fallback is not supported)"
        )
    return groups


def trajectory_pairs(
    groups: list[tuple[torch.Tensor, ...]],
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample monotone trajectories; return adjacent row-index pairs."""
    pairs = []
    for buckets in groups:
        paths = torch.stack(
            [
                bucket[
                    torch.randperm(
                        len(bucket), generator=generator, device=bucket.device
                    )[: len(buckets[0])]
                ]
                for bucket in buckets
            ],
            dim=1,
        )
        pairs.append(
            torch.stack([paths[:, :-1].reshape(-1), paths[:, 1:].reshape(-1)], dim=1)
        )
    return torch.cat(pairs)


def gradient_integral(
    model: nn.Module,
    start: torch.Tensor,
    end: torch.Tensor,
    *,
    design_dim: int,
    quadrature_nodes: int = 5,
) -> torch.Tensor:
    """Approximate f(end)-f(start) by left-node gradient quadrature.

    Inputs are normalized design/context features. Context must be identical
    at both ends. Affine design normalization cancels between displacement and
    gradient. Keep the derivative graph so this loss can train model weights.
    The model must not couple rows (e.g. via batch normalization or attention).
    """
    if start.ndim != 2 or start.shape != end.shape or not len(start):
        raise ValueError("integral endpoints must be matching non-empty matrices")
    if not 1 <= design_dim <= start.shape[1] or quadrature_nodes < 1:
        raise ValueError("invalid integral dimensions or quadrature count")
    if not torch.isfinite(start).all() or not torch.isfinite(end).all():
        raise RuntimeError("non-finite MATCH-OPT endpoints")
    if not torch.equal(start[:, design_dim:], end[:, design_dim:]):
        raise ValueError("gradient matching cannot cross fidelity contexts")
    delta = (end - start).detach()
    integral = start.new_zeros(len(start))
    for index in range(quadrature_nodes):
        point = (start.detach() + delta * (index / quadrature_nodes)).requires_grad_(
            True
        )
        prediction = model(point)
        if prediction.shape != (len(point),):
            raise ValueError("matching model must return one scalar per row")
        gradient = torch.autograd.grad(prediction.sum(), point, create_graph=True)[0]
        integral = (
            integral
            + (gradient[:, :design_dim] * delta[:, :design_dim]).sum(-1)
            / quadrature_nodes
        )
    return integral


def search_matching_model(
    model: nn.Module,
    data: MentoringData,
    problem: OfflineProblem,
    starts: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> torch.Tensor:
    """Maximize the frozen surrogate, updating only feasible design coordinates."""
    coordinates = nn.Parameter(problem.design_space.to_unconstrained(starts).detach())
    optimizer = torch.optim.Adam([coordinates], lr=learning_rate)
    for _ in range(steps):
        designs = problem.design_space.from_unconstrained(coordinates)
        score = model(data.at_target(problem, designs))
        gradient = torch.autograd.grad(-score.sum(), coordinates)[0]
        if not torch.isfinite(score).all() or not torch.isfinite(gradient).all():
            raise RuntimeError("non-finite MATCH-OPT search gradient")
        optimizer.zero_grad(set_to_none=True)
        coordinates.grad = gradient
        optimizer.step()
        with torch.no_grad():
            coordinates.clamp_(-20, 20)
    return problem.design_space.from_unconstrained(coordinates).detach()


@register_method()
class GradientMatchingMethod(OfflineBBOMethod):
    metadata = MethodMetadata(
        method_id="match_opt",
        display_name="MATCH-OPT adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url="https://github.com/azzafadhel/MatchOpt",
        source_commit="aae3f579a04400206eaa7abe836961c7af94b508",
        description="Value regression and five-node line-integral gradient matching.",
        adaptations=(
            "independent_pytorch_implementation_from_paper",
            "compact_configurable_embedding_width_and_buckets",
            "exact_same_fidelity_utility_ordered_trajectories",
            "all_visible_rows_supervised_once_per_epoch",
            "random_pair_minibatch_per_supervised_update",
            "summed_output_input_gradient_for_row_independent_mlp",
            "visible_feature_and_utility_standardization",
            "final_epoch_checkpoint_without_oracle_selection",
            "fixed_target_fidelity_constrained_adam_search",
            "unique_logged_and_random_initializations",
            "reject_no_same_fidelity_pairs_without_mse_fallback",
        ),
    )
    capabilities = MethodCapabilities(supports_box=True)

    def __init__(
        self,
        *,
        embedding_dim: int = 8,
        surrogate_epochs: int = 50,
        batch_size: int = 128,
        bucket_count: int = 32,
        quadrature_nodes: int = 5,
        surrogate_learning_rate: float = 1e-4,
        matching_weight: float = 1.0,
        solver_steps: int = 150,
        solver_learning_rate: float = 1e-3,
        minimum_std: float = 1e-6,
    ) -> None:
        for name, value in (
            ("embedding_dim", embedding_dim),
            ("surrogate_epochs", surrogate_epochs),
            ("batch_size", batch_size),
            ("bucket_count", bucket_count),
            ("quadrature_nodes", quadrature_nodes),
            ("solver_steps", solver_steps),
        ):
            minimum = 2 if name == "bucket_count" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("surrogate_learning_rate", surrogate_learning_rate),
            ("matching_weight", matching_weight),
            ("solver_learning_rate", solver_learning_rate),
            ("minimum_std", minimum_std),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        self.embedding_dim = embedding_dim
        self.surrogate_epochs = surrogate_epochs
        self.batch_size = batch_size
        self.bucket_count = bucket_count
        self.quadrature_nodes = quadrature_nodes
        self.surrogate_learning_rate = surrogate_learning_rate
        self.matching_weight = matching_weight
        self.solver_steps = solver_steps
        self.solver_learning_rate = solver_learning_rate
        self.minimum_std = minimum_std

    def _fit(
        self,
        problem: OfflineProblem,
        data: MentoringData,
        groups: list[tuple[torch.Tensor, ...]],
        context: RunContext,
        generator: torch.Generator,
    ) -> tuple[nn.Module, dict]:
        seed = int(
            torch.randint(
                0, 2**31 - 1, (), generator=generator, device=context.device
            ).item()
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = MatchingMLP(data.features.shape[1], self.embedding_dim).to(
                device=context.device, dtype=context.dtype
            )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.surrogate_learning_rate
        )
        updates, pairs_used, zero_pairs = 0, 0, 0
        value_history, matching_history = [], []
        for _ in range(self.surrogate_epochs):
            pairs = trajectory_pairs(groups, generator)
            zero_pairs += int(
                (
                    problem.train_designs[pairs[:, 0]]
                    == problem.train_designs[pairs[:, 1]]
                )
                .all(-1)
                .sum()
            )
            order = torch.randperm(
                problem.sample_count, generator=generator, device=context.device
            )
            value_total, matching_total, epoch_updates = 0.0, 0.0, 0
            for indices in order.split(self.batch_size):
                selected = torch.randperm(
                    len(pairs), generator=generator, device=context.device
                )[: self.batch_size]
                left, right = pairs[selected].unbind(1)
                predicted_delta = gradient_integral(
                    model,
                    data.features[left],
                    data.features[right],
                    design_dim=problem.design_dim,
                    quadrature_nodes=self.quadrature_nodes,
                )
                matching_loss = F.mse_loss(
                    predicted_delta, data.utility[right] - data.utility[left]
                )
                value_loss = F.mse_loss(
                    model(data.features[indices]), data.utility[indices]
                )
                loss = value_loss + self.matching_weight * matching_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite MATCH-OPT training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise RuntimeError("non-finite MATCH-OPT training gradient")
                optimizer.step()
                value_total += float(value_loss.detach())
                matching_total += float(matching_loss.detach())
                epoch_updates += 1
                pairs_used += len(left)
            updates += epoch_updates
            value_history.append(value_total / epoch_updates)
            matching_history.append(matching_total / epoch_updates)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("non-finite MATCH-OPT fitted weights")
        model.eval().requires_grad_(False)
        eligible_rows = sum(sum(len(bucket) for bucket in group) for group in groups)
        return model, {
            "train_samples": problem.sample_count,
            "epochs": self.surrogate_epochs,
            "eligible_fidelity_groups": len(groups),
            "eligible_pairing_rows": eligible_rows,
            "singleton_rows_value_only": problem.sample_count - eligible_rows,
            "effective_bucket_counts": [len(group) for group in groups],
            "trajectory_pairs_per_epoch": len(pairs),
            "pair_samples_used": pairs_used,
            "zero_displacement_trajectory_pairs_total": zero_pairs,
            "training_updates": updates,
            "supervised_rows_seen": problem.sample_count * self.surrogate_epochs,
            "quadrature": "left_endpoints",
            "quadrature_nodes": self.quadrature_nodes,
            "value_loss_history": value_history,
            "matching_loss_history": matching_history,
            "history_reduction": "mean_of_minibatch_losses",
            "checkpoint_selection": "final_epoch_all_visible_rows",
        }

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        groups = fidelity_buckets(problem, self.bucket_count)
        data = MentoringData.from_problem(problem, self.minimum_std)
        model, summary = self._fit(problem, data, groups, context, generator)
        starts, logged_count, random_count = initialize_candidate_designs(
            problem,
            candidate_budget=context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        candidates = search_matching_model(
            model,
            data,
            problem,
            starts,
            steps=self.solver_steps,
            learning_rate=self.solver_learning_rate,
        )
        return MethodResult(
            candidates,
            summary,
            {
                "search": "frozen_matching_surrogate_adam",
                "solver_steps": self.solver_steps,
                "logged_initializations": logged_count,
                "random_initializations": random_count,
                "pairing_context": "exact_logged_fidelity",
                "search_context": "fixed_target_fidelity",
            },
        )
