import numpy as np
import pytest

from llm_design_bench.tasks.data_recipes import DataRecipesTask
from llm_design_bench.types import CandidateBatch


@pytest.fixture
def task() -> DataRecipesTask:
    task = object.__new__(DataRecipesTask)
    task.simplex_tolerance = 1e-6
    return task


def test_accepts_valid_simplex(task: DataRecipesTask) -> None:
    task.validate(CandidateBatch.at_fidelity(np.array([[0.2] * 5]), 1000, 19_500))


@pytest.mark.parametrize(
    "mixture",
    [
        [0.2, 0.2, 0.2, 0.2, -0.1],
        [0.2, 0.2, 0.2, 0.2, 0.3],
        [np.nan, 0.25, 0.25, 0.25, 0.25],
    ],
)
def test_rejects_invalid_simplex(
    task: DataRecipesTask,
    mixture: list[float],
) -> None:
    batch = CandidateBatch.at_fidelity(np.array([mixture]), 1000, 19_500)
    with pytest.raises(ValueError):
        task.validate(batch)
