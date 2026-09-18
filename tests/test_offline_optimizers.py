"""Current offline methods propose designs without receiving the evaluator."""

import numpy as np
import pytest
import torch
from toy_offline_task import ToyOfflineTask

from llm_design_bench.evaluation.data_manifest import stratified_percentile_mask
from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import OfflineProblem, RunContext


@pytest.mark.parametrize(
    "method_id,kwargs,dtype",
    [
        (
            "offline_mlp",
            {"hidden_size": 16, "epochs": 2, "batch_size": 4, "particle_steps": 2},
            torch.float32,
        ),
        (
            "coms",
            {
                "hidden_size": 16,
                "epochs": 2,
                "batch_size": 4,
                "adversarial_steps": 2,
                "particle_steps": 2,
            },
            torch.float32,
        ),
        ("bdi", {"steps": 2}, torch.float64),
    ],
)
def test_offline_method_returns_simplex_without_oracle_queries(
    method_id, kwargs, dtype
) -> None:
    task = ToyOfflineTask()
    problem = OfflineProblem.from_task(task)
    result = make_method(method_id, **kwargs).run(
        problem,
        RunContext(method_seed=7, candidate_budget=4, dtype=dtype),
    )
    assert task.predict_calls == 0
    assert result.candidates.shape == (4, 3)
    assert result.candidates.dtype == dtype
    problem.design_space.validate(result.candidates)
    # Evaluation remains an explicit operation after proposal, outside the method.
    recommendations = task.at_target_fidelity(result.candidates.cpu().numpy())
    utility = task.predict(recommendations)
    assert task.predict_calls == 1
    assert utility.shape == (4,)
    assert np.isfinite(utility).all()


def test_scale_stratified_split_exposes_only_low_utility_within_each_scale():
    scales = np.array([20, 20, 20, 20, 1000, 1000, 1000, 1000])
    utility = np.array([-6.0, -5.0, -4.0, -3.0, -2.0, -1.9, -1.8, -1.7])
    mask = stratified_percentile_mask(utility, scales)
    assert np.array_equal(np.flatnonzero(mask), [0, 1, 4, 5])
    for scale in np.unique(scales):
        assert np.all(
            utility[mask & (scales == scale)]
            <= np.percentile(utility[scales == scale], 40.0)
        )
    fixed_1b = mask & (scales == 1000)
    assert np.array_equal(np.flatnonzero(fixed_1b), [4, 5])
