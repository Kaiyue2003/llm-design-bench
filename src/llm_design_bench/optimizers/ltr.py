"""Independent RaM-ListNet adaptation with an oracle-free training boundary."""

from __future__ import annotations

import copy
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
from llm_design_bench.optimizers.mentoring_utils import MentoringData, MentoringMLP
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import initialize_candidate_designs
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


def listnet_loss(predictions: torch.Tensor, utilities: torch.Tensor) -> torch.Tensor:
    """Top-one ListNet cross entropy, averaged over lists (not list items)."""
    if predictions.ndim != 2 or predictions.shape != utilities.shape:
        raise ValueError("ListNet requires matching (lists, items) matrices")
    if not predictions.numel():
        raise ValueError("ListNet requires non-empty lists")
    if not torch.isfinite(predictions).all() or not torch.isfinite(utilities).all():
        raise RuntimeError("non-finite ListNet inputs")
    return -(utilities.softmax(-1) * predictions.log_softmax(-1)).sum(-1).mean()


def sample_lists(
    count: int,
    length: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Sample independent lists without replacement within each list."""
    if min(count, length, batch_size) < 1:
        raise ValueError("list sampling sizes must be positive")
    return torch.stack(
        [
            torch.randperm(count, generator=generator, device=device)[
                : min(count, length)
            ]
            for _ in range(batch_size)
        ]
    )


def prediction_statistics(
    predictions: torch.Tensor,
    minimum_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use unit scale for constant predictions instead of amplifying by epsilon."""
    if predictions.ndim != 1 or not predictions.numel():
        raise ValueError("predictions must be a non-empty vector")
    if not torch.isfinite(predictions).all():
        raise RuntimeError("non-finite LTR calibration predictions")
    raw_std = predictions.std(unbiased=False)
    scale = torch.where(raw_std >= minimum_std, raw_std, torch.ones_like(raw_std))
    return predictions.mean().detach(), scale.detach()


def search_ranker(
    model: nn.Module,
    data: MentoringData,
    problem: OfflineProblem,
    starts: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> torch.Tensor:
    """Maximize normalized ranking scores at fixed target fidelity."""
    coordinates = nn.Parameter(problem.design_space.to_unconstrained(starts).detach())
    optimizer = torch.optim.Adam([coordinates], lr=learning_rate)
    for _ in range(steps):
        designs = problem.design_space.from_unconstrained(coordinates)
        score = (model(data.at_target(problem, designs)) - mean) / scale
        gradient = torch.autograd.grad(-score.sum(), coordinates)[0]
        if not torch.isfinite(score).all() or not torch.isfinite(gradient).all():
            raise RuntimeError("non-finite LTR candidate gradient")
        optimizer.zero_grad(set_to_none=True)
        coordinates.grad = gradient
        optimizer.step()
        with torch.no_grad():
            coordinates.clamp_(-20, 20)
    return problem.design_space.from_unconstrained(coordinates).detach()


@register_method()
class LearningToRankMethod(OfflineBBOMethod):
    """Sampled-list ranking, offline checkpoint selection, calibrated search.

    Validation uses fixed sampled lists drawn from the same visible row pool
    as training; it does not estimate unseen-row generalization. Lists for
    training are resampled lazily each epoch. Defaults are compact benchmark
    adaptations, not the source paper's final experimental configuration.
    """

    metadata = MethodMetadata(
        method_id="ltr",
        display_name="LTR adaptation",
        family=MethodFamily.FORWARD_SURROGATE,
        implementation_kind=ImplementationKind.MULTI_FIDELITY_ADAPTATION,
        source_url="https://github.com/lamda-bbo/Offline-RaM",
        source_commit="389e4bcf68c3e645e3a36e0f84ebdf05a76c235f",
        description="RaM-ListNet ranking proxy and normalized-output gradient search.",
        adaptations=(
            "independent_pytorch_implementation_from_paper",
            "compact_configurable_network_and_list_budget",
            "lazy_resampled_lists_without_replacement_within_list",
            "fixed_validation_lists_share_visible_row_pool",
            "visible_feature_and_utility_standardization",
            "visible_prediction_mean_std_search_calibration",
            "unit_scale_for_constant_features_and_predictions",
            "fixed_target_fidelity_constrained_adam_search",
            "unique_logged_and_random_initializations",
            "uninformative_labels_return_initial_candidates_without_training",
        ),
    )
    capabilities = MethodCapabilities(supports_box=True)

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        surrogate_epochs: int = 50,
        list_length: int = 32,
        lists_per_epoch: int = 256,
        batch_size: int = 32,
        validation_lists: int = 32,
        surrogate_learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        solver_steps: int = 200,
        solver_learning_rate: float = 1e-3,
        minimum_std: float = 1e-6,
    ) -> None:
        for name, value in (
            ("hidden_size", hidden_size),
            ("surrogate_epochs", surrogate_epochs),
            ("list_length", list_length),
            ("lists_per_epoch", lists_per_epoch),
            ("batch_size", batch_size),
            ("validation_lists", validation_lists),
            ("solver_steps", solver_steps),
        ):
            minimum = 2 if name == "list_length" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("surrogate_learning_rate", surrogate_learning_rate),
            ("weight_decay", weight_decay),
            ("solver_learning_rate", solver_learning_rate),
            ("minimum_std", minimum_std),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and name != "weight_decay")
            ):
                raise ValueError(f"invalid {name}")
        self.hidden_size = hidden_size
        self.surrogate_epochs = surrogate_epochs
        self.list_length = list_length
        self.lists_per_epoch = lists_per_epoch
        self.batch_size = batch_size
        self.validation_lists = validation_lists
        self.surrogate_learning_rate = surrogate_learning_rate
        self.weight_decay = weight_decay
        self.solver_steps = solver_steps
        self.solver_learning_rate = solver_learning_rate
        self.minimum_std = minimum_std

    def _fit(
        self,
        data: MentoringData,
        context: RunContext,
        generator: torch.Generator,
    ) -> tuple[nn.Module, dict]:
        seed = int(
            torch.randint(
                0,
                2**31 - 1,
                (),
                generator=generator,
                device=context.device,
            ).item()
        )
        # Initialize on CPU inside a fork, without altering global CPU/CUDA RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = MentoringMLP(data.features.shape[1], self.hidden_size).to(
                device=context.device,
                dtype=context.dtype,
            )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.surrogate_learning_rate,
            weight_decay=self.weight_decay,
        )
        count = len(data.utility)
        validation = sample_lists(
            count,
            self.list_length,
            self.validation_lists,
            generator,
            context.device,
        )
        best_loss, best_state, selected_epoch = math.inf, None, 0
        history, updates = [], 0
        for epoch in range(self.surrogate_epochs):
            model.train()
            for offset in range(0, self.lists_per_epoch, self.batch_size):
                indices = sample_lists(
                    count,
                    self.list_length,
                    min(self.batch_size, self.lists_per_epoch - offset),
                    generator,
                    context.device,
                )
                loss = listnet_loss(
                    model(data.features[indices]), data.utility[indices]
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite LTR training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise RuntimeError("non-finite LTR training gradient")
                optimizer.step()
                updates += 1
            model.eval()
            with torch.no_grad():
                value = sum(
                    float(listnet_loss(model(data.features[ids]), data.utility[ids]))
                    * len(ids)
                    for ids in validation.split(self.batch_size)
                ) / len(validation)
            if not math.isfinite(value):
                raise RuntimeError("non-finite LTR validation loss")
            history.append(value)
            if value < best_loss:
                best_loss, selected_epoch = value, epoch + 1
                best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        model.eval().requires_grad_(False)
        return model, {
            "training_status": "trained",
            "ranking_loss": "listnet",
            "epochs": self.surrogate_epochs,
            "training_updates": updates,
            "effective_list_length": min(count, self.list_length),
            "lists_per_epoch": self.lists_per_epoch,
            "validation_lists": self.validation_lists,
            "validation_unit": "sampled_lists_shared_visible_rows",
            "checkpoint_selection": "minimum_fixed_list_validation_loss",
            "selected_epoch": selected_epoch,
            "best_validation_loss": best_loss,
            "validation_loss_history": history,
        }

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        data = MentoringData.from_problem(problem, self.minimum_std)
        informative = (
            problem.sample_count > 1
            and float(problem.train_utility.std(unbiased=False)) >= self.minimum_std
        )
        model = None
        summary = {
            "training_status": "skipped_uninformative_labels",
            "ranking_loss": "listnet",
            "training_updates": 0,
            "effective_list_length": min(problem.sample_count, self.list_length),
        }
        if informative:
            model, summary = self._fit(data, context, generator)
        summary["train_samples"] = problem.sample_count
        summary["feature_dim"] = int(data.features.shape[1])
        starts, logged_count, random_count = initialize_candidate_designs(
            problem,
            candidate_budget=context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        diagnostics = {
            "search": "normalized_ranker_adam"
            if informative
            else "initialization_only",
            "solver_steps": self.solver_steps if informative else 0,
            "logged_initializations": logged_count,
            "random_initializations": random_count,
        }
        candidates = starts
        if model is not None:
            with torch.no_grad():
                mean, scale = prediction_statistics(
                    model(data.features),
                    self.minimum_std,
                )
            diagnostics.update(
                prediction_mean=float(mean), prediction_scale=float(scale)
            )
            candidates = search_ranker(
                model,
                data,
                problem,
                starts,
                mean,
                scale,
                steps=self.solver_steps,
                learning_rate=self.solver_learning_rate,
            )
        return MethodResult(candidates.detach(), summary, diagnostics)
