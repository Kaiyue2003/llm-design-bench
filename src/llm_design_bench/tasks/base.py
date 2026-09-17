from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from llm_design_bench.types import CandidateBatch


class Task(Protocol):
    mixture_dim: int
    target_model_scale: float
    target_training_steps: float

    @property
    def logged_x(self) -> CandidateBatch: ...

    @property
    def logged_y(self) -> NDArray[np.generic]: ...

    def validate(self, batch: CandidateBatch) -> None: ...

    def predict(self, batch: CandidateBatch) -> NDArray[np.generic]: ...

    def cost(self, batch: CandidateBatch) -> NDArray[np.generic]: ...

    def at_target_fidelity(self, designs: NDArray[np.generic]) -> CandidateBatch: ...
