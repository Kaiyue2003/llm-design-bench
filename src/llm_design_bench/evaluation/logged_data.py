"""Logged-data views shared by current task factories, without runner imports."""

import numpy as np

from llm_design_bench.types import CandidateBatch


class LoggedDatasetView:
    """Expose selected logs while delegating task metadata to the original task.

    This is an evaluator-side adapter used to construct an ``OfflineProblem``;
    it is not itself a method-facing object or an oracle-access boundary.
    """

    def __init__(self, task, logged_x: CandidateBatch, logged_y: np.ndarray) -> None:
        self._task = task
        self._logged_x = logged_x
        self._logged_y = logged_y

    @property
    def logged_x(self) -> CandidateBatch:
        return self._logged_x

    @property
    def logged_y(self) -> np.ndarray:
        return self._logged_y

    def __getattr__(self, name: str):
        return getattr(self._task, name)
