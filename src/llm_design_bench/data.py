from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from llm_design_bench.problem import OfflineProblem


@dataclass(frozen=True)
class OfflineTensorDataset(Dataset):
    """A PyTorch-only view of logged offline observations.

    Row indices are retained so every train/validation split can be audited
    against the original logged dataset.
    """

    designs: torch.Tensor
    context: torch.Tensor
    utility: torch.Tensor
    row_indices: torch.Tensor

    def __post_init__(self) -> None:
        _validate_float_matrix("designs", self.designs)
        _validate_float_matrix("context", self.context)
        _validate_float_vector("utility", self.utility)
        if not isinstance(self.row_indices, torch.Tensor):
            raise TypeError("row_indices must be a torch.Tensor")
        if self.row_indices.dtype != torch.long or self.row_indices.ndim != 1:
            raise TypeError("row_indices must be a one-dimensional torch.long tensor")
        count = len(self.designs)
        if len(self.context) != count or len(self.utility) != count:
            raise ValueError("designs, context, and utility must have equal row counts")
        if len(self.row_indices) != count:
            raise ValueError("row_indices must have one entry per observation")
        if not (
            self.designs.device
            == self.context.device
            == self.utility.device
            == self.row_indices.device
        ):
            raise ValueError("all dataset tensors must be on the same device")
        if not (self.designs.dtype == self.context.dtype == self.utility.dtype):
            raise ValueError("all floating-point dataset tensors must use the same dtype")
        if count and torch.any(self.row_indices < 0):
            raise ValueError("row_indices must be non-negative")

    @classmethod
    def from_problem(cls, problem: OfflineProblem) -> "OfflineTensorDataset":
        return cls(
            designs=problem.train_designs,
            context=problem.train_context,
            utility=problem.train_utility,
            row_indices=torch.arange(
                problem.sample_count,
                device=problem.train_designs.device,
                dtype=torch.long,
            ),
        )

    def __len__(self) -> int:
        return int(self.utility.shape[0])

    def __getitem__(self, index):
        return self.designs[index], self.context[index], self.utility[index]

    @property
    def design_dim(self) -> int:
        return int(self.designs.shape[1])

    @property
    def context_dim(self) -> int:
        return int(self.context.shape[1])

    def select(self, indices: torch.Tensor) -> "OfflineTensorDataset":
        indices = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.designs.device,
        )
        if indices.ndim != 1:
            raise ValueError("selection indices must be one-dimensional")
        if len(indices) and (
            torch.any(indices < 0) or torch.any(indices >= len(self))
        ):
            raise IndexError("selection index is outside the dataset")
        return OfflineTensorDataset(
            designs=self.designs[indices],
            context=self.context[indices],
            utility=self.utility[indices],
            row_indices=self.row_indices[indices],
        )

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "OfflineTensorDataset":
        dtype = dtype or self.designs.dtype
        return OfflineTensorDataset(
            designs=self.designs.to(device=device, dtype=dtype),
            context=self.context.to(device=device, dtype=dtype),
            utility=self.utility.to(device=device, dtype=dtype),
            row_indices=self.row_indices.to(device=device),
        )


@dataclass(frozen=True)
class OfflineDataSplit:
    train: OfflineTensorDataset
    validation: OfflineTensorDataset
    split_seed: int
    validation_fraction: float

    def __post_init__(self) -> None:
        if self.split_seed < 0:
            raise ValueError("split_seed must be non-negative")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.train.design_dim != self.validation.design_dim:
            raise ValueError("train and validation design dimensions must match")
        if self.train.context_dim != self.validation.context_dim:
            raise ValueError("train and validation context dimensions must match")
        combined = torch.cat(
            [self.train.row_indices.cpu(), self.validation.row_indices.cpu()]
        )
        if len(torch.unique(combined)) != len(combined):
            raise ValueError("train and validation rows must be disjoint")


def split_offline_dataset(
    dataset: OfflineTensorDataset,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> OfflineDataSplit:
    """Create a deterministic method-independent holdout split."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if len(dataset) < 1:
        raise ValueError("the offline dataset must not be empty")

    validation_count = int(round(len(dataset) * validation_fraction))
    if validation_fraction > 0.0 and len(dataset) > 1:
        validation_count = min(max(validation_count, 1), len(dataset) - 1)
    else:
        validation_count = 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator)
    validation_indices = order[:validation_count].to(dataset.designs.device)
    train_indices = order[validation_count:].to(dataset.designs.device)
    return OfflineDataSplit(
        train=dataset.select(train_indices),
        validation=dataset.select(validation_indices),
        split_seed=seed,
        validation_fraction=float(validation_fraction),
    )


def _validate_float_matrix(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have two dimensions")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _validate_float_vector(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must have one dimension")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
