"""Boundary and compatibility checks for the legacy utility transform."""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

from llm_design_bench.optimizers.offline_utils import batch_features, offline_data
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
        target_model_scale=1000,
        max_training_steps=19_600,
    )


@pytest.mark.parametrize("value", [-2.5, 0.0, 7.0])
def test_singleton_utility_is_zero_and_scale_is_finite_without_warnings(value):
    task = _task([value])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = offline_data(task)

    torch.testing.assert_close(data.utility, torch.zeros(1), rtol=0, atol=0)
    torch.testing.assert_close(data.utility_mean, torch.tensor(value), rtol=0, atol=0)
    torch.testing.assert_close(data.utility_std, torch.tensor(1e-6), rtol=0, atol=0)
    for tensor in (data.features, data.utility, data.utility_mean, data.utility_std):
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert torch.isfinite(tensor).all()
    assert data.features.shape == (1, 5)
    np.testing.assert_array_equal(task.logged_y, [value])


@pytest.mark.parametrize("count", [2, 5])
@pytest.mark.parametrize("value", [-2.5, 0.0, 7.0])
def test_constant_utility_uses_existing_minimum_scale(count, value):
    data = offline_data(_task([value] * count))
    torch.testing.assert_close(data.utility, torch.zeros(count), rtol=0, atol=0)
    torch.testing.assert_close(data.utility_mean, torch.tensor(value), rtol=0, atol=0)
    torch.testing.assert_close(data.utility_std, torch.tensor(1e-6), rtol=0, atol=0)


@pytest.mark.parametrize(
    "values",
    [[-3.5, -1.25], [-5.0, -2.0, -1.0, 0.0, 4.5], [0.0, 1e-8, -1e-8]],
    ids=["two-samples", "varied-samples", "below-scale-floor"],
)
def test_multi_sample_transform_matches_legacy_formula_exactly(values):
    task = _task(values)
    original_utility = task.logged_y.copy()
    original_mixtures = task.logged_x.mixtures.copy()
    utility = torch.as_tensor(task.logged_y, dtype=torch.float32)
    expected_mean = utility.mean()
    expected_std = utility.std().clamp_min(1e-6)
    data = offline_data(task)

    torch.testing.assert_close(data.utility_mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(data.utility_std, expected_std, rtol=0, atol=0)
    torch.testing.assert_close(
        data.utility, (utility - expected_mean) / expected_std, rtol=0, atol=0
    )
    torch.testing.assert_close(
        data.features, batch_features(task.logged_x, task), rtol=0, atol=0
    )
    np.testing.assert_array_equal(task.logged_y, original_utility)
    np.testing.assert_array_equal(task.logged_x.mixtures, original_mixtures)


def test_empty_utility_is_rejected_before_computing_statistics():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="at least one"):
            offline_data(_task([]))


@pytest.mark.parametrize("utility", [-2.0, [[-2.0], [-1.0]], [[-2.0, -1.0]]])
def test_utility_must_be_one_dimensional(utility):
    with pytest.raises(ValueError, match="one-dimensional"):
        offline_data(_task(utility))


@pytest.mark.parametrize("utility", [[-2.0], [-3.0, -2.0, -1.0]])
def test_utility_count_must_match_feature_rows(utility):
    with pytest.raises(ValueError, match="one utility per"):
        offline_data(_task(utility, count=2))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("count", [1, 2])
def test_nonfinite_utility_is_rejected(value, count):
    utility = [value] + [-2.0] * (count - 1)
    with pytest.raises(ValueError, match="finite"):
        offline_data(_task(utility))
