import numpy as np
import pandas as pd

from llm_design_bench.registry import make


def _write_fake_data_recipes(root):
    (root / "opt_algos").mkdir()
    (root / "results").mkdir()
    (root / "opt_algos" / "benchmarks.py").write_text(
        "class DataModelBenchmark: pass\n",
        encoding="utf-8",
    )
    rows = [
        {
            "token_probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
            "group": "20M",
            "history": pd.DataFrame(
                {
                    "_step": [19_500],
                    "eval/RedPajamaStackExchange/CrossEntropyLoss": [2.5],
                }
            ),
        },
        {
            "token_probabilities": [0.4, 0.1, 0.2, 0.2, 0.1],
            "group": "1B",
            "history": pd.DataFrame(
                {
                    "_step": [19_500],
                    "eval/RedPajamaStackExchange/CrossEntropyLoss": [1.25],
                }
            ),
        },
        {
            "token_probabilities": [0.1, 0.4, 0.1, 0.2, 0.2],
            "group": "1B",
            "history": pd.DataFrame(
                {
                    "_step": [19_500],
                    "eval/RedPajamaStackExchange/CrossEntropyLoss": [1.75],
                }
            ),
        },
    ]
    pd.DataFrame(rows).to_pickle(root / "results" / "data_mixing_runs.pkl")


def test_data_recipes_logged_model_scale_filter(tmp_path) -> None:
    _write_fake_data_recipes(tmp_path)

    task = make("data-recipes", data_recipes_root=tmp_path, logged_model_scale=1000)

    assert task.logged_x.mixtures.shape == (2, 5)
    assert np.all(task.logged_x.model_scales == 1000)
    assert np.all(task.logged_x.training_steps == 19_500)
    assert np.allclose(task.logged_y, [-1.25, -1.75])


def test_data_recipes_1b_alias_filters_logged_data(tmp_path) -> None:
    _write_fake_data_recipes(tmp_path)

    task = make("data-recipes-1b", data_recipes_root=tmp_path)

    assert task.logged_x.mixtures.shape == (2, 5)
    assert np.all(task.logged_x.model_scales == 1000)

