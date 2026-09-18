"""Boundary regressions for the method-visible data and utility transform."""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

from llm_design_bench.problem import OfflineProblem
from llm_design_bench.types import CandidateBatch


def _task(utility, *, count=None):
    utility = np.asarray(utility, dtype=np.float64)
    if count is None:
        count = utility.size
    return SimpleNamespace(
        logged_x=CandidateBatch.at_fidelity(
            np.tile([0.2, 0.3, 0.5], (count, 1)), 1000, 19_500
        ),
        logged_y=utility,
        mixture_dim=3,
        target_model_scale=1000,
        target_training_steps=19_500,
    )


@pytest.mark.parametrize("value", [-2.5, 0.0, 7.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_singleton_utility_is_zero_and_scale_is_finite_without_warnings(value, dtype):
    task = _task([value])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        problem = OfflineProblem.from_task(task, dtype=dtype)
        utility, mean, std = problem.standardized_utility()

    torch.testing.assert_close(utility, torch.zeros(1, dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(mean, torch.tensor(value, dtype=dtype), rtol=0, atol=0)
    torch.testing.assert_close(std, torch.tensor(1e-6, dtype=dtype), rtol=0, atol=0)
    for tensor in (problem.train_features, utility, mean, std):
        assert tensor.dtype == dtype
        assert tensor.device.type == "cpu"
        assert torch.isfinite(tensor).all()
    assert problem.train_features.shape == (1, 5)
    np.testing.assert_array_equal(task.logged_y, [value])


@pytest.mark.parametrize("count", [2, 5])
@pytest.mark.parametrize("value", [-2.5, 0.0, 7.0])
def test_constant_utility_uses_minimum_scale(count, value):
    problem = OfflineProblem.from_task(_task([value] * count))
    utility, mean, std = problem.standardized_utility()
    torch.testing.assert_close(utility, torch.zeros(count), rtol=0, atol=0)
    torch.testing.assert_close(mean, torch.tensor(value), rtol=0, atol=0)
    torch.testing.assert_close(std, torch.tensor(1e-6), rtol=0, atol=0)


@pytest.mark.parametrize(
    "values",
    [[-3.5, -1.25], [-5.0, -2.0, -1.0, 0.0, 4.5], [0.0, 1e-8, -1e-8]],
    ids=["two-samples", "varied-samples", "below-scale-floor"],
)
def test_multi_sample_transform_preserves_population_standard_deviation(values):
    task = _task(values)
    original_utility = task.logged_y.copy()
    original_mixtures = task.logged_x.mixtures.copy()
    original = torch.as_tensor(task.logged_y, dtype=torch.float32)
    expected_mean = original.mean()
    # Check this public transform; Prepared methods retain their own train-only fit.
    expected_std = original.std(unbiased=False).clamp_min(1e-6)
    problem = OfflineProblem.from_task(task)
    utility, mean, std = problem.standardized_utility()

    torch.testing.assert_close(mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(std, expected_std, rtol=0, atol=0)
    torch.testing.assert_close(
        utility, (original - expected_mean) / expected_std, rtol=0, atol=0
    )
    expected_features = torch.as_tensor(
        np.column_stack(
            [
                original_mixtures,
                task.logged_x.model_scales,
                task.logged_x.training_steps,
            ]
        ),
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        problem.train_features, expected_features, rtol=0, atol=0
    )
    np.testing.assert_array_equal(task.logged_y, original_utility)
    np.testing.assert_array_equal(task.logged_x.mixtures, original_mixtures)


def test_empty_utility_is_rejected_before_computing_statistics():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="must not be empty"):
            OfflineProblem.from_task(_task([]))


@pytest.mark.parametrize("utility", [-2.0, [[-2.0], [-1.0]], [[-2.0, -1.0]]])
def test_utility_must_be_one_dimensional(utility):
    with pytest.raises(ValueError, match="train_utility must have 1 dimensions"):
        OfflineProblem.from_task(_task(utility))


@pytest.mark.parametrize("utility", [[-2.0], [-3.0, -2.0, -1.0]])
def test_utility_count_must_match_design_rows(utility):
    with pytest.raises(ValueError, match="equal row counts"):
        OfflineProblem.from_task(_task(utility, count=2))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("count", [1, 2])
def test_nonfinite_utility_is_rejected(value, count):
    utility = [value] + [-2.0] * (count - 1)
    with pytest.raises(ValueError, match="train_utility must be finite"):
        OfflineProblem.from_task(_task(utility))
