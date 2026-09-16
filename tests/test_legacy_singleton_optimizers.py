"""Legacy optimizers remain finite with very small logged datasets."""

import numpy as np
import pytest

from llm_design_bench.optimizers import bdi, coms, mlp_surrogate
from llm_design_bench.types import CandidateBatch
from test_offline_optimizers import ToyOfflineTask


@pytest.fixture(params=["offline_mlp", "coms", "bdi"])
def legacy_optimizer(request):
    common = {"recommendations": 3, "seed": 7, "device": "cpu"}
    if request.param == "bdi":
        return bdi, bdi.BackwardDistillationOptimizer(steps=2, **common)
    neural = {
        "hidden_size": 8,
        "epochs": 2,
        "batch_size": 4,
        "particle_steps": 2,
        **common,
    }
    if request.param == "coms":
        return coms, coms.ConservativeObjectiveModelOptimizer(
            adversarial_steps=2, **neural
        )
    return mlp_surrogate, mlp_surrogate.OfflineMLPOptimizer(**neural)


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
def test_legacy_optimizer_handles_constant_utility_without_search_queries(
    legacy_optimizer, sample_count, utility, monkeypatch
) -> None:
    module, optimizer = legacy_optimizer
    task = _constant_task(sample_count, utility)
    original_finalize = module.finalize_offline_trace
    final_calls = []
    evaluated_batches = []
    original_predict = task.predict

    def predict(batch):
        assert final_calls, "oracle must not be called during candidate search"
        evaluated_batches.append(batch)
        return original_predict(batch)

    def finalize(name, final_task, mixtures):
        assert final_task is task
        assert task.predict_calls == 0
        final_calls.append(name)
        return original_finalize(name, final_task, mixtures)

    monkeypatch.setattr(task, "predict", predict)
    monkeypatch.setattr(module, "finalize_offline_trace", finalize)

    trace = optimizer.optimize(task)

    assert final_calls == [trace.name]
    assert task.predict_calls == 1
    assert len(evaluated_batches) == 1
    assert evaluated_batches[0] is trace.recommendations
    assert len(trace.queried) == 0
    assert trace.query_utility.size == 0
    assert trace.query_cost.size == 0
    assert trace.cumulative_cost == 0.0
    assert trace.recommendations.mixtures.shape == (3, task.mixture_dim)
    assert np.isfinite(trace.recommendations.mixtures).all()
    assert (trace.recommendations.mixtures >= 0).all()
    np.testing.assert_allclose(
        trace.recommendations.mixtures.sum(axis=1), 1.0, atol=1e-6
    )
    np.testing.assert_array_equal(
        trace.recommendations.model_scales, np.full(3, task.target_model_scale)
    )
    np.testing.assert_array_equal(
        trace.recommendations.training_steps, np.full(3, task.target_training_steps)
    )
    assert trace.recommendation_utility.shape == (3,)
    assert np.isfinite(trace.recommendation_utility).all()
    np.testing.assert_allclose(
        trace.recommendation_utility, task._oracle(trace.recommendations.mixtures)
    )


def test_legacy_optimizer_rejects_empty_data_before_evaluation(
    legacy_optimizer, monkeypatch
) -> None:
    module, optimizer = legacy_optimizer
    task = _constant_task(0)

    def forbidden_finalize(*args, **kwargs):
        pytest.fail("empty logged data must fail before final evaluation")

    monkeypatch.setattr(module, "finalize_offline_trace", forbidden_finalize)

    with pytest.raises(ValueError, match="at least one"):
        optimizer.optimize(task)

    assert task.predict_calls == 0
