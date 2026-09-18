from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CandidateBatch:
    # Keep constructor inputs dtype-generic; __post_init__ coerces to float64.
    mixtures: NDArray[np.generic]
    model_scales: NDArray[np.generic]
    training_steps: NDArray[np.generic]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mixtures", np.asarray(self.mixtures, dtype=float))
        object.__setattr__(
            self, "model_scales", np.asarray(self.model_scales, dtype=float)
        )
        object.__setattr__(
            self, "training_steps", np.asarray(self.training_steps, dtype=float)
        )

    def __len__(self) -> int:
        return len(self.mixtures)

    @classmethod
    def at_fidelity(
        cls,
        mixtures: ArrayLike,
        model_scale: float,
        training_steps: float,
    ) -> "CandidateBatch":
        mixtures = np.asarray(mixtures, dtype=float)
        if mixtures.ndim != 2:
            raise ValueError("mixtures must be a two-dimensional array")
        count = len(mixtures)
        return cls(
            mixtures=mixtures,
            model_scales=np.full(count, model_scale, dtype=float),
            training_steps=np.full(count, training_steps, dtype=float),
        )
