from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch

from llm_design_bench.spaces import BoxSpace, DesignSpace, SimplexSpace


@dataclass(frozen=True)
class ProblemMetadata:
    task_name: str
    objective_name: str = "utility"
    utility_direction: str = "maximize"
    source: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_name:
            raise ValueError("task_name must not be empty")
        if not self.objective_name:
            raise ValueError("objective_name must not be empty")
        if self.utility_direction != "maximize":
            raise ValueError("offline problems must expose maximization utility")


@dataclass(frozen=True)
class OfflineProblem:
    """The complete data boundary visible to an offline optimization method."""

    train_designs: torch.Tensor
    train_context: torch.Tensor
    train_utility: torch.Tensor
    target_context: torch.Tensor
    design_space: DesignSpace
    metadata: ProblemMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.design_space, DesignSpace):
            raise TypeError("design_space must implement DesignSpace")
        _validate_float_tensor("train_designs", self.train_designs, dimensions=2)
        _validate_float_tensor("train_context", self.train_context, dimensions=2)
        _validate_float_tensor("train_utility", self.train_utility, dimensions=1)
        _validate_float_tensor("target_context", self.target_context, dimensions=1)

        count = len(self.train_designs)
        if count < 1:
            raise ValueError("the offline dataset must not be empty")
        if len(self.train_context) != count or len(self.train_utility) != count:
            raise ValueError("designs, context, and utility must have equal row counts")
        if self.train_designs.shape[1] != self.design_space.dimension:
            raise ValueError("train_designs do not match the design-space dimension")
        if self.train_context.shape[1] != len(self.target_context):
            raise ValueError("target_context must match the logged context dimension")
        if not (
            self.train_designs.device
            == self.train_context.device
            == self.train_utility.device
            == self.target_context.device
        ):
            raise ValueError("all problem tensors must be on the same device")
        if not (
            self.train_designs.dtype
            == self.train_context.dtype
            == self.train_utility.dtype
            == self.target_context.dtype
        ):
            raise ValueError("all problem tensors must use the same dtype")
        self.design_space.validate(self.train_designs)

    @property
    def sample_count(self) -> int:
        return len(self.train_designs)

    @property
    def design_dim(self) -> int:
        return self.design_space.dimension

    @property
    def context_dim(self) -> int:
        return int(self.train_context.shape[1])

    @property
    def train_features(self) -> torch.Tensor:
        return torch.cat([self.train_designs, self.train_context], dim=1)

    def features_at_target(self, designs: torch.Tensor) -> torch.Tensor:
        self.design_space.validate(designs)
        if designs.device != self.target_context.device:
            raise ValueError("designs and target_context must be on the same device")
        if designs.dtype != self.target_context.dtype:
            raise ValueError("designs and target_context must use the same dtype")
        context = self.target_context.unsqueeze(0).expand(len(designs), -1)
        return torch.cat([designs, context], dim=1)

    def standardized_utility(
        self,
        minimum_std: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if minimum_std <= 0:
            raise ValueError("minimum_std must be positive")
        mean = self.train_utility.mean()
        std = self.train_utility.std(unbiased=False).clamp_min(minimum_std)
        return (self.train_utility - mean) / std, mean, std

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        *,
        copy: bool = False,
    ) -> "OfflineProblem":
        dtype = dtype or self.train_designs.dtype

        def convert(value: torch.Tensor) -> torch.Tensor:
            converted = value.to(device=device, dtype=dtype)
            return converted.clone() if copy else converted

        return OfflineProblem(
            train_designs=convert(self.train_designs),
            train_context=convert(self.train_context),
            train_utility=convert(self.train_utility),
            target_context=convert(self.target_context),
            design_space=self.design_space.clone() if copy else self.design_space,
            metadata=self.metadata,
        )

    @classmethod
    def from_task(
        cls,
        task,
        *,
        design_space: DesignSpace | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        metadata: ProblemMetadata | None = None,
    ) -> "OfflineProblem":
        logged_x = task.logged_x
        designs = torch.as_tensor(
            logged_x.mixtures,
            device=device,
            dtype=dtype,
        ).clone()
        context_array = np.column_stack(
            [logged_x.model_scales, logged_x.training_steps]
        )
        context = torch.as_tensor(context_array, device=device, dtype=dtype).clone()
        utility = torch.as_tensor(task.logged_y, device=device, dtype=dtype).clone()
        target_context = torch.tensor(
            [
                float(getattr(task, "target_model_scale", 1.0)),
                float(getattr(task, "target_training_steps", 1.0)),
            ],
            device=device,
            dtype=dtype,
        )
        if design_space is None:
            bounds = getattr(task, "design_bounds", None)
            if bounds is None:
                design_space = SimplexSpace(int(task.mixture_dim))
            else:
                # Preserve the original limits so float32 decoding cannot round
                # outward and fail the evaluator's float64 domain check.
                design_space = BoxSpace(torch.as_tensor(bounds, dtype=torch.float64))
        if metadata is None:
            metric = getattr(task, "metric", None)
            objective_name = getattr(metric, "name", getattr(task, "name", "utility"))
            metadata = ProblemMetadata(
                task_name=str(getattr(task, "name", task.__class__.__name__)),
                objective_name=str(objective_name),
                source=task.__class__.__module__,
            )
        return cls(
            train_designs=designs,
            train_context=context,
            train_utility=utility,
            target_context=target_context,
            design_space=design_space,
            metadata=metadata,
        )


@dataclass(frozen=True)
class RunContext:
    method_seed: int
    candidate_budget: int
    device: torch.device | str = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    dataset_seed: int | None = None
    split_seed: int | None = None

    def __post_init__(self) -> None:
        if self.method_seed < 0:
            raise ValueError("method_seed must be non-negative")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError("dtype must be floating point")
        object.__setattr__(self, "device", torch.device(self.device))

    def make_generator(self) -> torch.Generator:
        generator_device = self.device if self.device.type in {"cpu", "cuda"} else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.method_seed)
        return generator


@dataclass(frozen=True)
class MethodResult:
    """Unevaluated candidates and method-local diagnostics.

    Oracle utilities intentionally do not belong in this object.
    """

    candidates: torch.Tensor
    training_summary: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_float_tensor("candidates", self.candidates, dimensions=2)


def _validate_float_tensor(name: str, value: torch.Tensor, dimensions: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if value.ndim != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
