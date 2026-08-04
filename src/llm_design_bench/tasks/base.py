from typing import Protocol

import numpy as np

from llm_design_bench.types import CandidateBatch


class Task(Protocol):
    @property
    def logged_x(self) -> CandidateBatch: ...

    @property
    def logged_y(self) -> np.ndarray: ...

    def validate(self, batch: CandidateBatch) -> None: ...

    def predict(self, batch: CandidateBatch) -> np.ndarray: ...

    def cost(self, batch: CandidateBatch) -> np.ndarray: ...
