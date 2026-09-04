from typing import Protocol

import numpy as np

from llm_design_bench.types import CandidateBatch


class Task(Protocol):
    mixture_dim: int
    target_model_scale: float
    target_training_steps: float

    @property
    def logged_x(self) -> CandidateBatch: ...

    @property
    def logged_y(self) -> np.ndarray: ...

    def validate(self, batch: CandidateBatch) -> None: ...

    def predict(self, batch: CandidateBatch) -> np.ndarray: ...

    def cost(self, batch: CandidateBatch) -> np.ndarray: ...

    def at_target_fidelity(self, designs: np.ndarray) -> CandidateBatch: ...
