"""Numerical contracts for the GPyTorch backend, independent of its kernels."""

import math
from dataclasses import replace

import gpytorch
import pytest
import torch

from llm_design_bench.optimizers import make_method
from llm_design_bench.optimizers.exact_gp import (
    fit_exact_rbf_gp,
    stable_posterior_cholesky,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import BoxSpace

_MINIMUM_STD = 1e-6
_INITIAL_LENGTHSCALE = 0.7
_INITIAL_OUTPUT_SCALE = 1.3
_INITIAL_NOISE = 0.03
_JITTER = 1e-6


def _problem(dtype: torch.dtype = torch.float64) -> OfflineProblem:
    return OfflineProblem(
        train_designs=torch.tensor(
            [[-0.8, 0.2], [-0.1, -0.5], [0.4, 0.7], [0.9, -0.2]],
            dtype=dtype,
        ),
        train_context=torch.tensor(
            [[1.0, 10.0], [2.0, 10.0], [4.0, 10.0], [8.0, 10.0]],
            dtype=dtype,
        ),
        train_utility=torch.tensor([-1.4, -0.6, -0.9, -1.8], dtype=dtype),
        target_context=torch.tensor([4.0, 10.0], dtype=dtype),
        design_space=BoxSpace(torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])),
        metadata=ProblemMetadata(task_name="gp-numerical-contract"),
    )


def _fit(problem: OfflineProblem, *, steps: int = 0):
    # Zero steps fixes known hyperparameters; public method budgets remain > 0.
    return fit_exact_rbf_gp(
        problem,
        RunContext(
            method_seed=38, candidate_budget=3, dtype=problem.train_designs.dtype
        ),
        training_steps=steps,
        learning_rate=0.02,
        initial_lengthscale=_INITIAL_LENGTHSCALE,
        initial_output_scale=_INITIAL_OUTPUT_SCALE,
        initial_noise=_INITIAL_NOISE,
        minimum_std=_MINIMUM_STD,
        jitter=_JITTER,
    )


def _visible_normalization(problem: OfflineProblem):
    # Intentionally do not use GP caches or problem.standardized_utility().
    features = torch.cat((problem.train_designs, problem.train_context), dim=1)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(_MINIMUM_STD)
    utility_mean = problem.train_utility.mean()
    utility_std = problem.train_utility.std(unbiased=False).clamp_min(_MINIMUM_STD)
    targets = (problem.train_utility - utility_mean) / utility_std
    return (features - mean) / std, targets, mean, std


def _dense_kernel(first, second, lengthscale, output_scale):
    distances = ((first[:, None] - second[None, :]) / lengthscale).square().sum(-1)
    return output_scale * (-0.5 * distances).exp()


def _dense_nll(train_inputs, targets, lengthscale, output_scale, noise):
    covariance = _dense_kernel(train_inputs, train_inputs, lengthscale, output_scale)
    covariance = covariance + (noise + _JITTER) * torch.eye(
        len(targets), dtype=targets.dtype, device=targets.device
    )
    sign, logdet = torch.linalg.slogdet(covariance)
    assert sign.item() == 1.0
    nll = (
        0.5 * targets @ torch.linalg.solve(covariance, targets)
        + 0.5 * logdet
        + 0.5 * len(targets) * math.log(2.0 * math.pi)
    )
    return nll, covariance


def _dense_reference(problem, designs, *, lengthscale, output_scale, noise):
    train_inputs, targets, feature_mean, feature_std = _visible_normalization(problem)
    features = torch.cat(
        (designs, problem.target_context.expand(len(designs), -1)), dim=1
    )
    test_inputs = (features - feature_mean) / feature_std
    nll, train_covariance = _dense_nll(
        train_inputs, targets, lengthscale, output_scale, noise
    )
    cross = _dense_kernel(train_inputs, test_inputs, lengthscale, output_scale)
    mean = cross.T @ torch.linalg.solve(train_covariance, targets)
    covariance = _dense_kernel(test_inputs, test_inputs, lengthscale, output_scale)
    covariance = covariance - cross.T @ torch.linalg.solve(train_covariance, cross)
    return nll, mean, covariance


def _candidates(dtype=torch.float64, *, requires_grad=False):
    return torch.tensor(
        [[-0.2, 0.4], [0.5, 0.6], [0.8, -0.3]],
        dtype=dtype,
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("steps", [0, 5])
def test_gp_nll_and_latent_posterior_match_independent_dense_reference(steps):
    problem = _problem()
    gp = _fit(problem, steps=steps)
    candidates = _candidates()
    # For zero-step fit, reference constants also detect incorrect initialization.
    lengthscale = gp.lengthscale if steps else _INITIAL_LENGTHSCALE
    output_scale = gp.output_scale if steps else _INITIAL_OUTPUT_SCALE
    noise = gp.noise if steps else _INITIAL_NOISE
    nll, expected_mean, expected_covariance = _dense_reference(
        problem,
        candidates,
        lengthscale=lengthscale,
        output_scale=output_scale,
        noise=noise,
    )

    mean, covariance = gp.posterior_standardized(problem, candidates)

    assert gp.final_negative_log_likelihood == pytest.approx(nll.item(), rel=1e-9)
    torch.testing.assert_close(mean, expected_mean, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(covariance, expected_covariance, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        gp.posterior_mean_standardized(problem, candidates),
        mean,
        rtol=1e-10,
        atol=1e-10,
    )
    assert covariance.dtype == torch.float64
    torch.testing.assert_close(covariance, covariance.T, rtol=0, atol=1e-12)
    assert torch.linalg.eigvalsh(covariance).min() > -1e-10


@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("mean_only", [False, True])
def test_gp_posterior_candidate_gradients_match_dense_reference(warm_cache, mean_only):
    problem = _problem()
    gp = _fit(problem, steps=3)
    if warm_cache:
        # Solvers query initial predictions under no_grad before differentiating.
        with torch.no_grad():
            gp.posterior_mean_standardized(problem, _candidates())
    candidates = _candidates(requires_grad=True)
    reference_candidates = candidates.detach().clone().requires_grad_(True)
    if mean_only:
        mean = gp.posterior_mean_standardized(problem, candidates)
        actual_objective = mean.sum()
    else:
        mean, covariance = gp.posterior_standardized(problem, candidates)
        actual_objective = mean.sum() + 0.37 * covariance.sum()
    _, expected_mean, expected_covariance = _dense_reference(
        problem,
        reference_candidates,
        lengthscale=gp.lengthscale,
        output_scale=gp.output_scale,
        noise=gp.noise,
    )
    expected_objective = expected_mean.sum()
    if not mean_only:
        expected_objective = expected_objective + 0.37 * expected_covariance.sum()
    actual_gradient = torch.autograd.grad(actual_objective, candidates)[0]
    expected_gradient = torch.autograd.grad(expected_objective, reference_candidates)[0]

    assert torch.isfinite(actual_gradient).all()
    assert actual_gradient.abs().max() > 1e-3
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-8, atol=1e-9)


def test_gp_fit_preserves_log_parameter_adam_optimization():
    problem = _problem()
    train_inputs, targets, _, _ = _visible_normalization(problem)
    log_lengthscale = torch.nn.Parameter(
        torch.full(
            (train_inputs.shape[1],),
            math.log(_INITIAL_LENGTHSCALE),
            dtype=torch.float64,
        )
    )
    log_output_scale = torch.nn.Parameter(
        torch.tensor(math.log(_INITIAL_OUTPUT_SCALE), dtype=torch.float64)
    )
    log_noise = torch.nn.Parameter(
        torch.tensor(math.log(_INITIAL_NOISE), dtype=torch.float64)
    )
    parameters = [log_lengthscale, log_output_scale, log_noise]
    optimizer = torch.optim.Adam(parameters, lr=0.02)
    for _ in range(3):
        nll, _ = _dense_nll(
            train_inputs,
            targets,
            log_lengthscale.exp(),
            log_output_scale.exp(),
            log_noise.exp(),
        )
        optimizer.zero_grad(set_to_none=True)
        nll.backward()
        optimizer.step()

    gp = _fit(problem, steps=3)

    torch.testing.assert_close(
        gp.lengthscale, log_lengthscale.exp(), rtol=1e-9, atol=1e-10
    )
    torch.testing.assert_close(
        gp.output_scale, log_output_scale.exp(), rtol=1e-9, atol=1e-10
    )
    torch.testing.assert_close(gp.noise, log_noise.exp(), rtol=1e-9, atol=1e-10)


def test_gp_uses_frozen_gpytorch_modules_and_latent_not_noisy_posterior():
    problem = _problem()
    gp = _fit(problem, steps=2)
    assert isinstance(gp.model, gpytorch.models.ExactGP)
    assert isinstance(gp.likelihood, gpytorch.likelihoods.GaussianLikelihood)
    assert isinstance(gp.model.mean_module, gpytorch.means.ZeroMean)
    assert isinstance(gp.model.covar_module, gpytorch.kernels.ScaleKernel)
    assert isinstance(gp.model.covar_module.base_kernel, gpytorch.kernels.RBFKernel)
    assert (
        gp.model.covar_module.base_kernel.ard_num_dims
        == problem.train_features.shape[1]
    )
    assert not gp.model.training
    assert not gp.likelihood.training
    assert all(not parameter.requires_grad for parameter in gp.model.parameters())
    assert all(parameter.dtype == torch.float64 for parameter in gp.model.parameters())
    # Constraint buffers must not round jitter through the default float32 dtype.
    assert gp.likelihood.noise_covar.raw_noise_constraint.lower_bound.item() == _JITTER
    assert torch.equal(
        gp.likelihood.noise,
        gp.likelihood.raw_noise.exp() + torch.tensor(_JITTER, dtype=torch.float64),
    )

    candidates = _candidates()
    normalized_inputs = (
        problem.features_at_target(candidates) - gp.feature_mean
    ) / gp.feature_std
    latent_mean, latent_covariance = gp.posterior_standardized(problem, candidates)
    with torch.no_grad():
        observation = gp.likelihood(gp.model(normalized_inputs))
    # Numerical training jitter is bundled into likelihood.noise, while gp.noise
    # reports only learned variance. Neither belongs in latent test covariance.
    expected_diagonal = (gp.noise + gp.jitter) * torch.eye(3, dtype=torch.float64)
    torch.testing.assert_close(observation.mean, latent_mean, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        observation.covariance_matrix - latent_covariance,
        expected_diagonal,
        rtol=1e-9,
        atol=1e-10,
    )


def test_gp_normalization_uses_visible_data_not_target_fidelity():
    problem = _problem()
    changed_target = replace(
        problem, target_context=torch.tensor([1e9, -1e8], dtype=torch.float64)
    )
    gp = _fit(problem, steps=3)
    other = _fit(changed_target, steps=3)
    expected_inputs, expected_targets, mean, std = _visible_normalization(problem)

    torch.testing.assert_close(gp.feature_mean, mean)
    torch.testing.assert_close(gp.feature_std, std)
    torch.testing.assert_close(gp.train_inputs, expected_inputs)
    torch.testing.assert_close(gp.train_targets, expected_targets)
    assert gp.utility_mean == problem.train_utility.mean()
    assert gp.utility_std == problem.train_utility.std(unbiased=False)
    for field in (
        "feature_mean",
        "feature_std",
        "train_inputs",
        "train_targets",
        "lengthscale",
        "noise",
    ):
        assert torch.equal(getattr(gp, field), getattr(other, field))
    assert gp.final_negative_log_likelihood == other.final_negative_log_likelihood


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("constant_utility", [False, True])
def test_gp_repeated_inputs_and_constant_fidelity_are_finite(dtype, constant_utility):
    problem = _problem(dtype)
    repeated_designs = problem.train_designs[[0, 0, 2, 2]].clone()
    utility = (
        torch.full((4,), -1.5, dtype=dtype)
        if constant_utility
        else problem.train_utility
    )
    problem = replace(
        problem,
        train_designs=repeated_designs,
        train_context=problem.target_context.expand(4, -1).clone(),
        train_utility=utility,
    )
    gp = _fit(problem, steps=3)
    candidates = repeated_designs[[0, 0, 2]].clone().requires_grad_(True)
    mean, covariance = gp.posterior_standardized(problem, candidates)
    gradient = torch.autograd.grad(mean.sum() + covariance.sum(), candidates)[0]
    factor, used_jitter = stable_posterior_cholesky(covariance, jitter=_JITTER)

    assert math.isfinite(gp.final_negative_log_likelihood)
    assert mean.dtype == covariance.dtype == gradient.dtype == dtype
    assert torch.isfinite(mean).all()
    assert torch.isfinite(covariance).all()
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(factor).all()
    assert used_jitter >= _JITTER
    assert torch.equal(gp.train_inputs[:, -2:], torch.zeros((4, 2), dtype=dtype))
    if constant_utility:
        assert torch.equal(gp.train_targets, torch.zeros(4, dtype=dtype))
        assert gp.utility_std.item() == pytest.approx(_MINIMUM_STD)


def test_gp_fit_is_deterministic_and_does_not_consume_global_rng():
    with torch.random.fork_rng():
        torch.manual_seed(927)
        before = torch.random.get_rng_state().clone()
        first = _fit(_problem(), steps=3)
        assert torch.equal(torch.random.get_rng_state(), before)
        second = _fit(_problem(), steps=3)
        assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(first.lengthscale, second.lengthscale)
    assert torch.equal(first.output_scale, second.output_scale)
    assert torch.equal(first.noise, second.noise)
    assert first.final_negative_log_likelihood == second.final_negative_log_likelihood
    first_mean, first_covariance = first.posterior_standardized(
        _problem(), _candidates()
    )
    second_mean, second_covariance = second.posterior_standardized(
        _problem(), _candidates()
    )
    assert torch.equal(first_mean, second_mean)
    assert torch.equal(first_covariance, second_covariance)


@pytest.mark.parametrize(
    "method_id,kwargs",
    [
        ("ga_on_gp", {"gp_training_steps": 2, "solver_steps": 2}),
        ("bo_qei", {"gp_training_steps": 2, "acquisition_steps": 2, "mc_samples": 8}),
    ],
)
def test_gp_methods_report_gpytorch_backend_and_preserve_formal_float64(
    method_id, kwargs
):
    problem = _problem()
    context = RunContext(method_seed=38, candidate_budget=3, dtype=torch.float64)
    result = make_method(method_id, **kwargs).run(problem, context)

    problem.design_space.validate(result.candidates)
    assert result.candidates.shape == (3, 2)
    assert result.candidates.dtype == torch.float64
    assert result.training_summary["backend"] == "gpytorch"
    assert result.training_summary["gpytorch_version"] == gpytorch.__version__
    assert result.training_summary["inference"] == "dense_exact_cholesky"
    assert result.training_summary["posterior"] == "latent_function"
    assert result.training_summary["train_samples"] == problem.sample_count
    assert math.isfinite(result.training_summary["final_negative_log_likelihood"])
