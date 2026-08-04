from pathlib import Path

import numpy as np

from llm_design_bench.registry import make
from llm_design_bench.types import CandidateBatch

DATA_RECIPES = Path(__file__).resolve().parents[2] / "data-recipes"


def test_data_recipes_logged_dataset_contract() -> None:
    task = make("data-recipes", data_recipes_root=DATA_RECIPES)
    assert task.logged_x.mixtures.shape == (454, 5)
    assert task.logged_y.shape == (454,)


def test_data_recipes_predict_smoke() -> None:
    task = make("data-recipes", data_recipes_root=DATA_RECIPES)
    batch = CandidateBatch.at_fidelity(np.array([[0.2] * 5]), 1000, 19_500)
    utility = task.predict(batch)
    assert utility.shape == (1,)
    assert np.isfinite(utility).all()
    assert utility[0] < 0
