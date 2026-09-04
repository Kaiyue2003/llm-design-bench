from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DesignSpace(ABC):
    """Torch-native domain contract for optimizer-produced designs."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def validate(self, designs: torch.Tensor) -> None: ...

    @abstractmethod
    def from_unconstrained(self, parameters: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def to_unconstrained(self, designs: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor: ...

    @abstractmethod
    def clone(self) -> "DesignSpace": ...


class SimplexSpace(DesignSpace):
    """Non-negative vectors whose components sum to one."""

    def __init__(self, dimension: int, tolerance: float = 1e-6) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self._dimension = int(dimension)
        self.tolerance = float(tolerance)

    @property
    def dimension(self) -> int:
        return self._dimension

    def validate(self, designs: torch.Tensor) -> None:
        _validate_matrix(designs, self.dimension)
        if torch.any(designs < -self.tolerance):
            raise ValueError("simplex components must be non-negative")
        sums = designs.sum(dim=1)
        if not torch.allclose(
            sums,
            torch.ones_like(sums),
            atol=self.tolerance,
            rtol=0.0,
        ):
            raise ValueError("simplex components must sum to one")

    def from_unconstrained(self, parameters: torch.Tensor) -> torch.Tensor:
        _validate_matrix(parameters, self.dimension)
        return torch.softmax(parameters, dim=1)

    def to_unconstrained(self, designs: torch.Tensor) -> torch.Tensor:
        self.validate(designs)
        epsilon = torch.finfo(designs.dtype).tiny
        return torch.log(designs.clamp_min(epsilon))

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        device = torch.device(device)
        uniform = torch.rand(
            (count, self.dimension),
            generator=generator,
            device=device,
            dtype=dtype,
        ).clamp_min(torch.finfo(dtype).tiny)
        exponential = -torch.log(uniform)
        return exponential / exponential.sum(dim=1, keepdim=True)

    def clone(self) -> "SimplexSpace":
        return SimplexSpace(self.dimension, tolerance=self.tolerance)


class BoxSpace(DesignSpace):
    """Axis-aligned continuous box with finite lower and upper bounds."""

    def __init__(self, bounds: torch.Tensor, tolerance: float = 1e-7) -> None:
        bounds = torch.as_tensor(bounds)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dimension, 2)")
        if bounds.shape[0] < 1:
            raise ValueError("bounds must contain at least one dimension")
        if not torch.isfinite(bounds).all():
            raise ValueError("bounds must be finite")
        if torch.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("every upper bound must be greater than its lower bound")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self._bounds = bounds.detach().clone()
        self.tolerance = float(tolerance)

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds.clone()

    @property
    def dimension(self) -> int:
        return int(self._bounds.shape[0])

    def validate(self, designs: torch.Tensor) -> None:
        _validate_matrix(designs, self.dimension)
        bounds = self._bounds.to(device=designs.device, dtype=designs.dtype)
        lower, upper = bounds[:, 0], bounds[:, 1]
        if torch.any(designs < lower - self.tolerance) or torch.any(
            designs > upper + self.tolerance
        ):
            raise ValueError("designs are outside the box bounds")

    def from_unconstrained(self, parameters: torch.Tensor) -> torch.Tensor:
        _validate_matrix(parameters, self.dimension)
        bounds = self._bounds.to(device=parameters.device, dtype=parameters.dtype)
        lower, upper = bounds[:, 0], bounds[:, 1]
        return lower + (upper - lower) * torch.sigmoid(parameters)

    def to_unconstrained(self, designs: torch.Tensor) -> torch.Tensor:
        self.validate(designs)
        bounds = self._bounds.to(device=designs.device, dtype=designs.dtype)
        lower, upper = bounds[:, 0], bounds[:, 1]
        epsilon = torch.finfo(designs.dtype).eps
        unit = ((designs - lower) / (upper - lower)).clamp(epsilon, 1.0 - epsilon)
        return torch.logit(unit)

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if count < 1:
            raise ValueError("count must be positive")
        device = torch.device(device)
        bounds = self._bounds.to(device=device, dtype=dtype)
        lower, upper = bounds[:, 0], bounds[:, 1]
        unit = torch.rand(
            (count, self.dimension),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return lower + (upper - lower) * unit

    def clone(self) -> "BoxSpace":
        return BoxSpace(self._bounds, tolerance=self.tolerance)


def _validate_matrix(values: torch.Tensor, dimension: int) -> None:
    if not isinstance(values, torch.Tensor):
        raise TypeError("designs must be a torch.Tensor")
    if not values.is_floating_point():
        raise TypeError("designs must use a floating-point dtype")
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"designs must have shape (n, {dimension})")
    if not torch.isfinite(values).all():
        raise ValueError("designs must be finite")
