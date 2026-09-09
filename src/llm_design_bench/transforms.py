from __future__ import annotations

from dataclasses import dataclass

import torch

from llm_design_bench.data import (
    OfflineDataSplit,
    OfflineTensorDataset,
    split_offline_dataset,
)
from llm_design_bench.problem import OfflineProblem, RunContext


@dataclass(frozen=True)
class ProblemPreparationConfig:
    validation_fraction: float = 0.2
    split_seed: int | None = None
    log1p_context_indices: tuple[int, ...] | None = None
    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.split_seed is not None and self.split_seed < 0:
            raise ValueError("split_seed must be non-negative")
        if self.log1p_context_indices is not None:
            if any(index < 0 for index in self.log1p_context_indices):
                raise ValueError("context transform indices must be non-negative")
            if len(set(self.log1p_context_indices)) != len(
                self.log1p_context_indices
            ):
                raise ValueError("context transform indices must be unique")
        if self.minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")


@dataclass(frozen=True)
class TensorStandardizer:
    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(
        cls,
        values: torch.Tensor,
        *,
        minimum_scale: float = 1e-6,
    ) -> "TensorStandardizer":
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise TypeError("values must be a floating-point torch.Tensor")
        if values.ndim not in {1, 2} or len(values) < 1:
            raise ValueError("values must be a non-empty vector or matrix")
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")
        mean = values.mean(dim=0)
        if values.ndim == 2 and values.shape[1] == 0:
            scale = torch.ones_like(mean)
        else:
            scale = values.std(dim=0, unbiased=False).clamp_min(minimum_scale)
        return cls(mean=mean.detach(), scale=scale.detach())

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.scale

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale + self.mean


@dataclass(frozen=True)
class FittedProblemTransforms:
    problem: OfflineProblem
    context_standardizer: TensorStandardizer
    utility_standardizer: TensorStandardizer
    log1p_context_indices: tuple[int, ...]

    def encode_designs(self, designs: torch.Tensor) -> torch.Tensor:
        return self.problem.design_space.encode_for_model(designs)

    def decode_designs(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.problem.design_space.decode_from_model(encoded)

    def transform_context(self, context: torch.Tensor) -> torch.Tensor:
        transformed = _context_pretransform(
            context,
            self.log1p_context_indices,
        )
        return self.context_standardizer.transform(transformed)

    def transform_utility(self, utility: torch.Tensor) -> torch.Tensor:
        return self.utility_standardizer.transform(utility)

    def inverse_utility(self, utility: torch.Tensor) -> torch.Tensor:
        return self.utility_standardizer.inverse(utility)

    def features(self, designs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if len(designs) != len(context):
            raise ValueError("designs and context must have equal row counts")
        return torch.cat(
            [self.encode_designs(designs), self.transform_context(context)],
            dim=1,
        )

    def features_at_target(self, designs: torch.Tensor) -> torch.Tensor:
        target = self.problem.target_context.unsqueeze(0).expand(len(designs), -1)
        return self.features(designs, target)


@dataclass(frozen=True)
class PreparedOfflineProblem:
    problem: OfflineProblem
    split: OfflineDataSplit
    transforms: FittedProblemTransforms

    @property
    def train_features(self) -> torch.Tensor:
        return self.transforms.features(
            self.split.train.designs,
            self.split.train.context,
        )

    @property
    def train_utility(self) -> torch.Tensor:
        return self.transforms.transform_utility(self.split.train.utility)

    @property
    def validation_features(self) -> torch.Tensor:
        return self.transforms.features(
            self.split.validation.designs,
            self.split.validation.context,
        )

    @property
    def validation_utility(self) -> torch.Tensor:
        return self.transforms.transform_utility(self.split.validation.utility)

    def features_at_target(self, designs: torch.Tensor) -> torch.Tensor:
        return self.transforms.features_at_target(designs)


def prepare_offline_problem(
    problem: OfflineProblem,
    context: RunContext,
    config: ProblemPreparationConfig = ProblemPreparationConfig(),
) -> PreparedOfflineProblem:
    """Fit preprocessing on training rows only and expose PyTorch tensors."""

    split_seed = (
        config.split_seed
        if config.split_seed is not None
        else context.split_seed
        if context.split_seed is not None
        else 0
    )
    dataset = OfflineTensorDataset.from_problem(problem)
    split = split_offline_dataset(
        dataset,
        validation_fraction=config.validation_fraction,
        seed=split_seed,
    )
    log1p_context_indices = (
        config.log1p_context_indices
        if config.log1p_context_indices is not None
        else (0,)
        if problem.context_dim > 0
        else ()
    )
    transformed_context = _context_pretransform(
        split.train.context,
        log1p_context_indices,
    )
    transforms = FittedProblemTransforms(
        problem=problem,
        context_standardizer=TensorStandardizer.fit(
            transformed_context,
            minimum_scale=config.minimum_scale,
        ),
        utility_standardizer=TensorStandardizer.fit(
            split.train.utility,
            minimum_scale=config.minimum_scale,
        ),
        log1p_context_indices=log1p_context_indices,
    )
    return PreparedOfflineProblem(
        problem=problem,
        split=split,
        transforms=transforms,
    )


def _context_pretransform(
    context: torch.Tensor,
    log1p_indices: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(context, torch.Tensor) or not context.is_floating_point():
        raise TypeError("context must be a floating-point torch.Tensor")
    if context.ndim != 2:
        raise ValueError("context must have two dimensions")
    if log1p_indices and max(log1p_indices) >= context.shape[1]:
        raise ValueError("context transform index is outside the context dimension")
    transformed = context.clone()
    for index in log1p_indices:
        if torch.any(transformed[:, index] <= -1.0):
            raise ValueError("log1p context values must be greater than -1")
        transformed[:, index] = torch.log1p(transformed[:, index])
    return transformed
