from pathlib import Path

import numpy as np
import pytest

from llm_design_bench.tasks.data_recipes import DataRecipesTask
from llm_design_bench.types import CandidateBatch

DATA_RECIPES = Path(__file__).resolve().parents[2] / "data-recipes"


def test_accepts_valid_simplex() -> None:
    task = DataRecipesTask(DATA_RECIPES)
    task.validate(CandidateBatch.at_fidelity(np.array([[0.2] * 5]), 1000, 19_500))


@pytest.mark.parametrize(
    "mixture",
    [
        [0.2, 0.2, 0.2, 0.2, -0.1],
        [0.2, 0.2, 0.2, 0.2, 0.3],
        [np.nan, 0.25, 0.25, 0.25, 0.25],
    ],
)
def test_rejects_invalid_simplex(mixture: list[float]) -> None:
    task = DataRecipesTask(DATA_RECIPES)
    batch = CandidateBatch.at_fidelity(np.array([mixture]), 1000, 19_500)
    with pytest.raises(ValueError):
        task.validate(batch)
