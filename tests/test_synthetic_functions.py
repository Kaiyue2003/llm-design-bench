import numpy as np
import pytest

from llm_design_bench.optimizers.bdi import BackwardDistillationOptimizer
from llm_design_bench.optimizers.coms import ConservativeObjectiveModelOptimizer
from llm_design_bench.tasks.synthetic_functions import (
    DEFAULT_SYNTHETIC_FUNCTIONS,
    SYNTHETIC_CATEGORIES,
    SyntheticFunctionTask,
)


def test_full_synthetic_catalog_matches_sfu_category_count() -> None:
    assert len(DEFAULT_SYNTHETIC_FUNCTIONS) == 47
    assert {name for names in SYNTHETIC_CATEGORIES.values() for name in names} == set(
        DEFAULT_SYNTHETIC_FUNCTIONS
    )


@pytest.mark.parametrize("function_name", DEFAULT_SYNTHETIC_FUNCTIONS)
def test_synthetic_function_known_minimum(function_name: str) -> None:
    task = SyntheticFunctionTask(function_name, logged_samples=16, seed=0)
    optimum = task.at_target_fidelity(task.spec.global_minimizer[None, :])
    objective = task.objective(optimum.mixtures)
    utility = task.predict(optimum)

    tolerance = max(1e-2, 1e-3 * abs(task.spec.global_minimum_value))
    assert np.isclose(objective[0], task.spec.global_minimum_value, atol=tolerance)
    assert np.isclose(utility[0], task.oracle_utility, atol=tolerance)


@pytest.mark.parametrize(
    "optimizer",
    [
        ConservativeObjectiveModelOptimizer(
            recommendations=4,
            seed=7,
            hidden_size=16,
            epochs=2,
            batch_size=8,
            adversarial_steps=2,
            particle_steps=2,
        ),
        BackwardDistillationOptimizer(
            recommendations=4,
            seed=7,
            steps=2,
        ),
    ],
)
def test_box_offline_optimizers_keep_candidates_in_bounds(optimizer) -> None:
    task = SyntheticFunctionTask("booth", logged_samples=32, seed=3)
    trace = optimizer.optimize(task)

    lower = task.design_bounds[:, 0]
    upper = task.design_bounds[:, 1]
    assert trace.name in {"coms", "bdi"}
    assert len(trace.recommendations) == 4
    assert np.all(trace.recommendations.mixtures >= lower - 1e-6)
    assert np.all(trace.recommendations.mixtures <= upper + 1e-6)
    assert np.all(np.isfinite(trace.recommendation_utility))
