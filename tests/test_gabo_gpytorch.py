"""Independent numerical contracts for GABO's fixed latent GPyTorch model."""

import json

import gpytorch
import pytest
import torch

from llm_design_bench.optimizers import get_method_metadata, make_method
from llm_design_bench.optimizers.gabo import LatentGP, expected_improvement
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import SimplexSpace


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _data(dtype=torch.float64, device="cpu"):
    x = torch.tensor(
        [[-1.2, 0.3], [-0.3, 1.1], [0.4, -0.7], [0.9, 0.5], [1.7, 1.2]],
        dtype=dtype,
        device=device,
    )
    y = torch.tensor([-0.8, 0.7, -0.2, 1.3, 0.1], dtype=dtype, device=device)
    candidates = torch.tensor(
        [[-0.6, 0.4], [0.3, 0.2], [1.5, -0.2]], dtype=dtype, device=device
    )
    return x, y, candidates


def _dense_reference(x, y, candidates, ridge=1e-3):
    # Do not call the GPyTorch kernel or reuse its cached statistics. The
    # fixed settings here reproduce the pre-migration scalar RBF equations.
    distances = torch.pdist(x)
    positive = distances[distances > 0]
    lengthscale = (
        positive.median().clamp_min(0.1) if len(positive) else x.new_tensor(1.0)
    )
    mean = y.mean()
    scale = y.std(unbiased=False).clamp_min(0.1)

    def kernel(first, second):
        distance = (first[:, None] - second[None, :]).square().sum(-1)
        return (-0.5 * distance / lengthscale.square()).exp()

    covariance = kernel(x, x) + ridge * torch.eye(
        len(x), dtype=x.dtype, device=x.device
    )
    cross = kernel(candidates, x)
    weights = torch.linalg.solve(covariance, (y - mean) / scale)
    posterior_mean = cross @ weights * scale + mean
    variance = 1 - (cross * torch.linalg.solve(covariance, cross.T).T).sum(-1)
    return posterior_mean, variance.clamp_min(1e-8).sqrt() * scale


def _assert_close(actual, expected, dtype):
    tolerance = (
        {"rtol": 3e-4, "atol": 2e-5}
        if dtype == torch.float32
        else {"rtol": 1e-8, "atol": 1e-9}
    )
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_latent_posterior_and_candidate_gradients_match_dense_equations(
    dtype, warm_cache
):
    x, y, candidates = _data(dtype)
    gp = LatentGP(x, y)
    if warm_cache:
        with torch.no_grad():
            gp.predict(candidates)
    actual_inputs = candidates.clone().requires_grad_(True)
    expected_inputs = candidates.clone().requires_grad_(True)
    actual_mean, actual_std = gp.predict(actual_inputs)
    expected_mean, expected_std = _dense_reference(x, y, expected_inputs)
    _assert_close(actual_mean, expected_mean, dtype)
    _assert_close(actual_std, expected_std, dtype)
    actual_gradient = torch.autograd.grad(
        actual_mean.sum() + 0.37 * actual_std.sum(), actual_inputs
    )[0]
    expected_gradient = torch.autograd.grad(
        expected_mean.sum() + 0.37 * expected_std.sum(), expected_inputs
    )[0]
    assert torch.isfinite(actual_gradient).all()
    assert actual_gradient.abs().max() > 0
    _assert_close(actual_gradient, expected_gradient, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytic_ei_and_its_gradient_match_independent_normal_formula(dtype):
    x, y, candidates = _data(dtype)
    actual_inputs = candidates.clone().requires_grad_(True)
    expected_inputs = candidates.clone().requires_grad_(True)
    actual_mean, actual_std = LatentGP(x, y).predict(actual_inputs)
    expected_mean, expected_std = _dense_reference(x, y, expected_inputs)
    std = expected_std.clamp_min(1e-8)
    standardized = (expected_mean - y.max()) / std
    normal = torch.distributions.Normal(std.new_tensor(0.0), std.new_tensor(1.0))
    expected_ei = (expected_mean - y.max()) * normal.cdf(
        standardized
    ) + std * normal.log_prob(standardized).exp()
    actual_ei = expected_improvement(actual_mean, actual_std, y.max())
    _assert_close(actual_ei, expected_ei, dtype)
    actual_gradient = torch.autograd.grad(actual_ei.sum(), actual_inputs)[0]
    expected_gradient = torch.autograd.grad(expected_ei.sum(), expected_inputs)[0]
    _assert_close(actual_gradient, expected_gradient, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "case", ["duplicate_points", "constant_targets", "all_identical"]
)
def test_duplicate_inputs_and_constant_targets_remain_finite(dtype, case):
    x, y, candidates = _data(dtype)
    if case in {"duplicate_points", "all_identical"}:
        x[1] = x[0]
    if case in {"constant_targets", "all_identical"}:
        y.fill_(2.5)
    if case == "all_identical":
        x[:] = x[0]
    gp = LatentGP(x, y)
    actual_mean, actual_std = gp.predict(candidates)
    expected_mean, expected_std = _dense_reference(x, y, candidates)
    assert torch.isfinite(actual_mean).all()
    assert torch.isfinite(actual_std).all()
    assert torch.all(actual_std > 0)
    _assert_close(actual_mean, expected_mean, dtype)
    _assert_close(actual_std, expected_std, dtype)
    if case in {"constant_targets", "all_identical"}:
        assert float(gp.scale) == pytest.approx(0.1)
        torch.testing.assert_close(actual_mean, torch.full_like(actual_mean, 2.5))
    if case == "all_identical":
        assert float(gp.lengthscale) == 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_parameters_no_input_transform_or_training_graph(dtype):
    x, y, candidates = _data(dtype)
    x = (x * 7.0 + 50.0).requires_grad_(True)
    y = y.requires_grad_(True)
    rng_before = torch.get_rng_state().clone()
    gp = LatentGP(x, y, ridge=0.003)
    assert isinstance(gp.model, gpytorch.models.ExactGP)
    assert all(not parameter.requires_grad for parameter in gp.model.parameters())
    assert torch.equal(gp.model.train_inputs[0], x.detach())
    assert not gp.model.train_inputs[0].requires_grad
    assert not gp.model.train_targets.requires_grad
    assert gp.model.covar_module.outputscale.item() == 1.0
    assert gp.likelihood.noise.item() == pytest.approx(0.003, rel=1e-6)
    actual_lengthscale = gp.model.covar_module.base_kernel.lengthscale
    _assert_close(
        actual_lengthscale,
        torch.full_like(actual_lengthscale, gp.lengthscale.item()),
        dtype,
    )
    candidates = (candidates * 7.0 + 50.0).requires_grad_(True)
    mean, std = gp.predict(candidates)
    (mean.sum() + std.sum()).backward()
    assert candidates.grad is not None
    assert x.grad is None
    assert y.grad is None
    assert all(parameter.grad is None for parameter in gp.model.parameters())
    assert torch.equal(torch.get_rng_state(), rng_before)
    diagnostics = gp.diagnostics()
    assert diagnostics["backend"] == "gpytorch"
    assert diagnostics["hyperparameter_training_steps"] == 0
    assert diagnostics["input_transform"] == "none"
    assert diagnostics["additional_jitter"] == 0.0
    assert diagnostics["posterior"] == "latent_function"
    json.dumps(diagnostics, allow_nan=False)


def test_small_positive_distances_preserve_lengthscale_floor():
    x = torch.tensor([[0.0], [0.001], [0.002]], dtype=torch.float64)
    gp = LatentGP(x, torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64))
    assert gp.lengthscale.item() == 0.1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_posterior_is_latent_without_candidate_noise_and_ignores_ambient_variance_floor(
    dtype,
):
    x, y, candidates = _data(dtype)
    gp = LatentGP(x, y, ridge=0.05)
    expected_mean, expected_std = _dense_reference(x, y, candidates, ridge=0.05)
    with (
        gpytorch.settings.min_variance(float_value=0.9, double_value=0.9),
        gpytorch.settings.fast_pred_var(True),
        gpytorch.settings.prior_mode(True),
        gpytorch.settings.max_cholesky_size(0),
    ):
        mean, std = gp.predict(candidates)
    _assert_close(mean, expected_mean, dtype)
    _assert_close(std, expected_std, dtype)
    observed_std = (expected_std.square() + 0.05 * gp.scale.square()).sqrt()
    assert not torch.allclose(std, observed_std)


@pytest.mark.parametrize("ridge", [0.0, -0.01, float("nan"), float("inf"), True])
def test_invalid_ridge_is_rejected(ridge):
    x, y, _ = _data()
    with pytest.raises(ValueError, match="ridge"):
        LatentGP(x, y, ridge=ridge)


@pytest.mark.parametrize("case", ["empty", "target_shape", "dtype", "nan"])
def test_invalid_training_data_is_rejected(case):
    x, y, _ = _data()
    if case == "empty":
        x, y = x[:0], y[:0]
    elif case == "target_shape":
        y = y[:, None]
    elif case == "dtype":
        y = y.float()
    else:
        x[0, 0] = float("nan")
    with pytest.raises((ValueError, TypeError)):
        LatentGP(x, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_posterior_and_gradient_match_dense_equations(dtype):
    x, y, candidates = _data(dtype, "cuda")
    candidates.requires_grad_(True)
    reference_candidates = candidates.detach().clone().requires_grad_(True)
    mean, std = LatentGP(x, y).predict(candidates)
    expected_mean, expected_std = _dense_reference(x, y, reference_candidates)
    _assert_close(mean, expected_mean, dtype)
    _assert_close(std, expected_std, dtype)
    gradient = torch.autograd.grad(mean.sum() + std.sum(), candidates)[0]
    expected_gradient = torch.autograd.grad(
        expected_mean.sum() + expected_std.sum(), reference_candidates
    )[0]
    _assert_close(gradient, expected_gradient, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tiny_gabo_uses_gpytorch_without_changing_run_dtype(dtype):
    space = SimplexSpace(3)
    designs = space.sample(12, generator=torch.Generator().manual_seed(17), dtype=dtype)
    problem = OfflineProblem(
        train_designs=designs,
        train_context=torch.ones((12, 2), dtype=dtype),
        train_utility=-(designs - 0.3).square().sum(-1),
        target_context=torch.ones(2, dtype=dtype),
        design_space=space,
        metadata=ProblemMetadata(task_name="gabo-gpytorch-test"),
    )
    kwargs = {
        "epochs": 1,
        "steps": 2,
        "batch_size": 4,
        "hidden_size": 8,
        "latent_dim": 2,
        "initial_points": 4,
        "acquisition_steps": 2,
        "acquisition_restarts": 2,
        "critic_steps": 1,
    }
    context = RunContext(method_seed=38, split_seed=7, candidate_budget=4, dtype=dtype)
    result = make_method("gabo", **kwargs).run(problem, context)
    replay = make_method("gabo", **kwargs).run(problem, context)
    assert torch.equal(result.candidates, replay.candidates)
    assert result.candidates.dtype == dtype
    assert result.candidates.shape == (4, 3)
    space.validate(result.candidates)
    assert result.training_summary["latent_gp_backend"] == "gpytorch"
    assert result.diagnostics["latent_gp"]["backend"] == "gpytorch"
    assert result.diagnostics["latent_gp"]["dtype"] == str(dtype)
    assert result.diagnostics["latent_gp"]["hyperparameter_training_steps"] == 0
    assert any("GPyTorch" in item for item in get_method_metadata("gabo").adaptations)
