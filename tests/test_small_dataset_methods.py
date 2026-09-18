"""Current methods remain finite with very small logged datasets."""

import warnings

import numpy as np
import pytest
import torch

from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import OfflineProblem, RunContext
from llm_design_bench.types import CandidateBatch
from toy_offline_task import ToyOfflineTask


@pytest.fixture(params=["offline_mlp", "coms", "bdi"])
def method(request):
    if request.param == "bdi":
        return make_method("bdi", steps=2), torch.float64
    neural = {"hidden_size": 8, "epochs": 2, "batch_size": 4, "particle_steps": 2}
    if request.param == "coms":
        return make_method("coms", adversarial_steps=2, **neural), torch.float32
    return make_method("offline_mlp", **neural), torch.float32


def _constant_task(sample_count: int, utility: float = -1.25) -> ToyOfflineTask:
    task = ToyOfflineTask()
    task.logged_x = CandidateBatch(
        mixtures=task.logged_x.mixtures[:sample_count],
        model_scales=task.logged_x.model_scales[:sample_count],
        training_steps=task.logged_x.training_steps[:sample_count],
    )
    task.logged_y = np.full(sample_count, utility)
    return task


@pytest.mark.parametrize("sample_count", [1, 4], ids=["singleton", "constant"])
@pytest.mark.parametrize("utility", [-1.25, 0.0], ids=["negative", "zero"])
def test_method_handles_constant_utility_without_search_queries(
    method, sample_count, utility
) -> None:
    optimizer, dtype = method
    task = _constant_task(sample_count, utility)
    original_utility = task.logged_y.copy()
    original_designs = task.logged_x.mixtures.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        problem = OfflineProblem.from_task(task, dtype=dtype)
        result = optimizer.run(
            problem,
            RunContext(method_seed=7, candidate_budget=3, device="cpu", dtype=dtype),
        )

    assert task.predict_calls == 0, "candidate search cannot query the oracle"
    assert result.candidates.shape == (3, task.mixture_dim)
    assert torch.isfinite(result.candidates).all()
    problem.design_space.validate(result.candidates)
    assert result.training_summary["train_samples"] == sample_count
    assert result.training_summary["utility_mean"] == utility
    assert result.training_summary["utility_std"] > 0
    for value in result.training_summary.values():
        if isinstance(value, (int, float)):
            assert np.isfinite(value)

    batch = task.at_target_fidelity(result.candidates.detach().cpu().numpy())
    evaluated = task.predict(batch)
    assert task.predict_calls == 1
    assert evaluated.shape == (3,)
    assert np.isfinite(evaluated).all()
    np.testing.assert_array_equal(
        batch.model_scales, np.full(3, task.target_model_scale)
    )
    np.testing.assert_array_equal(
        batch.training_steps, np.full(3, task.target_training_steps)
    )
    np.testing.assert_allclose(evaluated, task._oracle(batch.mixtures))
    np.testing.assert_array_equal(task.logged_y, original_utility)
    np.testing.assert_array_equal(task.logged_x.mixtures, original_designs)


def test_empty_dataset_is_rejected_before_method_or_evaluator(method, monkeypatch):
    optimizer, dtype = method
    task = _constant_task(0)

    def forbidden_optimize(*args, **kwargs):
        pytest.fail("invalid input must not reach method optimization")

    monkeypatch.setattr(optimizer, "optimize", forbidden_optimize)
    with pytest.raises(ValueError, match="must not be empty"):
        optimizer.run(
            OfflineProblem.from_task(task, dtype=dtype),
            RunContext(method_seed=7, candidate_budget=3, dtype=dtype),
        )
    assert task.predict_calls == 0
