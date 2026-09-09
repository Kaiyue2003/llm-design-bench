from dataclasses import replace
from importlib.resources import files

import numpy as np
import pytest
import torch

from llm_design_bench.evaluation.seed_runner import (
    MethodSpec,
    SeedBenchmarkConfig,
    run_method_seed_benchmark,
)
from llm_design_bench.optimizers import make_method
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.optimizers.spade import SpadeMethod, evolve_designs
from llm_design_bench.optimizers.spade_core import (
    ScalarDiffusion,
    SupportNeighbors,
    calibration_loss,
    lower_confidence_bound,
    proximity_loss,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace
from llm_design_bench.types import CandidateBatch

CONFIG = {
    "diff_hidden": 8,
    "diff_t_dim": 4,
    "diff_steps": 4,
    "diff_epochs": 2,
    "diff_batch": 3,
    "calib_mc_samples": 3,
    "calib_mc_steps": 2,
    "acq_mc_samples": 3,
    "acq_mc_steps": 2,
    "ea_pop": 4,
    "ea_elite": 2,
    "ea_gens": 2,
    "mc_batch": 2,
    "knn_chunk": 2,
}


def problem(simplex=True, context=True):
    x = torch.tensor([[0.0, 1.0], [0.2, 0.8], [0.8, 0.2], [1.0, 0.0]])
    return OfflineProblem(
        x,
        torch.tensor([[20.0, 10.0], [60.0, 10.0], [20.0, 10.0], [60.0, 10.0]])
        if context
        else torch.empty(4, 0),
        torch.tensor([-2.0, -1.0, -3.0, -4.0]),
        torch.tensor([1000.0, 19500.0]) if context else torch.empty(0),
        SimplexSpace(2)
        if simplex
        else BoxSpace(torch.tensor([[0.0, 1.0], [0.0, 1.0]])),
        ProblemMetadata(task_name="spade-test"),
    )


def core():
    return ScalarDiffusion(
        2,
        SpadeMethod(**CONFIG),
        generator=torch.Generator().manual_seed(3),
        device="cpu",
        dtype=torch.float64,
    )


def test_core_preserves_global_rng_and_q_sample_formula():
    state = torch.random.get_rng_state().clone()
    model = core()
    y = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    noise = -y
    t = torch.tensor([0, 3])
    alpha = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 4, dtype=y.dtype), 0)[t, None]
    assert torch.allclose(
        model.q_sample(y, t, noise), alpha.sqrt() * y + (1 - alpha).sqrt() * noise
    )
    samples = model.samples(
        y.repeat(1, 2),
        count=17,
        steps=2,
        batch_size=1,
        generator=torch.Generator().manual_seed(9),
    )
    assert samples.shape == (17, 2) and samples.requires_grad
    assert torch.equal(state, torch.random.get_rng_state())


def test_ddim_zero_epsilon_matches_closed_form():
    model = core()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    x = torch.zeros(3, 2, dtype=torch.float64)
    initial = torch.randn(
        3, 1, dtype=x.dtype, generator=torch.Generator().manual_seed(7)
    )
    a = model.alpha_bar
    expected = initial / (a[3].sqrt() + 1e-8) * a[0].sqrt() / (a[0].sqrt() + 1e-8)
    assert torch.allclose(model.ddim(x, 2, torch.Generator().manual_seed(7)), expected)


def test_calibration_gradients_pass_through_sampling():
    model = core()
    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)
    y = torch.tensor([0.4, 0.8], dtype=x.dtype)

    def loss():
        mean = model.samples(
            x,
            count=3,
            steps=2,
            batch_size=2,
            generator=torch.Generator().manual_seed(8),
        ).mean(0)
        return calibration_loss(
            mean,
            y,
            pairs=8,
            temperature=1.0,
            generator=torch.Generator().manual_seed(1),
        )

    parameter = model.out[-1].bias
    analytic = torch.autograd.grad(loss(), parameter)[0].item()
    epsilon = 1e-5
    with torch.no_grad():
        parameter.add_(epsilon)
        high = loss().item()
        parameter.sub_(2 * epsilon)
        low = loss().item()
        parameter.add_(epsilon)
    assert analytic != 0
    assert analytic == pytest.approx((high - low) / (2 * epsilon), rel=1e-5, abs=1e-7)


def test_calibration_ties_singleton_and_rank_direction():
    generator = torch.Generator().manual_seed(1)
    assert (
        calibration_loss(
            torch.tensor([2.0]),
            torch.tensor([1.0]),
            pairs=32,
            temperature=1.0,
            generator=generator,
        )
        == 1
    )
    assert (
        calibration_loss(
            torch.ones(3), torch.ones(3), pairs=32, temperature=1.0, generator=generator
        )
        == 0
    )
    y = torch.tensor([0.0, 1.0, 2.0])
    # Remove the differing moment term to compare only sampled rank penalties.
    correct = calibration_loss(
        y, y, pairs=100, temperature=1.0, generator=torch.Generator().manual_seed(4)
    )
    reversed_loss = (
        calibration_loss(
            -y,
            y,
            pairs=100,
            temperature=1.0,
            generator=torch.Generator().manual_seed(4),
        )
        - ((-y - y) ** 2).mean()
    )
    assert correct < reversed_loss


def test_knn_chunking_self_neighbors_and_stable_ties():
    x = torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    y = torch.arange(4.0)
    for chunk in (1, 2, 10):
        helper = SupportNeighbors(x, y, 2, chunk)
        mean, radius = helper.query(x)
        distances = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")
        indices = distances.argsort(stable=True)[:, :2]
        assert torch.equal(mean, y[indices].mean(1))
        assert torch.equal(radius, distances.gather(1, indices)[:, -1])
        assert mean[0] == 0.5 and radius[0] == 0
    assert SupportNeighbors(x, y, 10, 2).k == 4


def test_support_hinges_and_degenerate_lcb_floor():
    radius = torch.tensor([1.0, 0.5])
    mean = torch.tensor([2.0, 1.0], requires_grad=True)
    sigma = torch.tensor([0.0, 0.1])
    neighbors = torch.tensor([1.0, 1.0])
    loss = proximity_loss(
        mean, sigma, neighbors, radius, margin=0.02, floor=0.02, slope=0.005
    )
    d = (radius + 1e-8).log()
    expected = (
        (mean - neighbors - 0.02 * d).clamp_min(0)
        + (0.02 + 0.005 * d - sigma).clamp_min(0)
    ).mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert mean.grad.abs().sum() > 0
    samples = torch.ones(4, 1)
    assert lower_confidence_bound(samples, 0.1).item() == 1
    assert lower_confidence_bound(
        samples, 0.1, radius=torch.ones(1)
    ).item() == pytest.approx(0.998)


@pytest.mark.parametrize("simplex", [True, False])
def test_evolution_scores_only_valid_designs_and_final_population(simplex):
    p = problem(simplex)
    data = MentoringData.from_problem(p, 1e-6)
    seen = []

    def score(designs):
        p.design_space.validate(designs)
        seen.append(designs.clone())
        return designs[:, 0]

    output, diagnostics = evolve_designs(
        p, data, score, SpadeMethod(**CONFIG), 9, torch.Generator().manual_seed(1)
    )
    assert len(seen) == 3 and all(len(batch) == 9 for batch in seen)
    assert output.shape == (9, 2)
    assert torch.equal(
        output, seen[-1][seen[-1][:, 0].argsort(descending=True, stable=True)]
    )
    assert diagnostics["acquisition_rows"] == 27


@pytest.mark.parametrize(
    "simplex,with_context", [(True, True), (True, False), (False, True), (False, False)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_method_contract_reproducibility_and_no_mutation(simplex, with_context, dtype):
    p = problem(simplex, with_context)
    snapshot = p.train_designs.clone()
    state = torch.random.get_rng_state().clone()
    np_state = np.random.get_state()
    method = make_method("spade", **CONFIG)
    context = RunContext(method_seed=38, candidate_budget=7, dtype=dtype)
    result = method.run(p, context)
    assert result.candidates.shape == (7, 2) and result.candidates.dtype == dtype
    p.design_space.validate(result.candidates)
    assert torch.equal(result.candidates, method.run(p, context).candidates)
    assert torch.equal(state, torch.random.get_rng_state())
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert torch.equal(p.train_designs, snapshot)
    assert result.training_summary["effective_support_k"] == 4
    assert result.diagnostics["acquisition_rows"] == 21


def test_frozen_model_and_fixed_target_context(monkeypatch):
    p = problem()
    original = ScalarDiffusion.samples
    observed = []
    data = MentoringData.from_problem(p, 1e-6)

    def tracked(self, features, **kwargs):
        if not torch.is_grad_enabled():
            assert all(not parameter.requires_grad for parameter in self.parameters())
            expected = (p.target_context - data.feature_mean[2:]) / data.feature_std[2:]
            assert torch.equal(features[:, 2:], expected.expand(len(features), -1))
            observed.append(True)
        return original(self, features, **kwargs)

    monkeypatch.setattr(ScalarDiffusion, "samples", tracked)
    make_method("spade", **CONFIG).run(p, RunContext(method_seed=1, candidate_budget=3))
    assert len(observed) == 3


@pytest.mark.parametrize("transform", [True, False])
def test_constant_utility_duplicate_designs_and_small_data(transform):
    p = problem()
    p = replace(
        p,
        train_designs=p.train_designs[:2].clone(),
        train_context=p.train_context[:2].clone(),
        train_utility=torch.ones(2),
    )
    p.train_designs[1] = p.train_designs[0]
    p.train_context[1] = p.train_context[0]
    result = make_method("spade", **CONFIG, support_transform=transform).run(
        p, RunContext(method_seed=3, candidate_budget=6)
    )
    assert torch.isfinite(result.candidates).all()
    assert result.training_summary["effective_support_k"] == 2


def test_oracle_only_after_return_and_resolved_budgets(tmp_path, monkeypatch):
    finished = []
    original = SpadeMethod.optimize

    def tracked(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        finished.append(True)
        return result

    monkeypatch.setattr(SpadeMethod, "optimize", tracked)

    class Oracle:
        calls = 0

        def at_target_fidelity(self, designs):
            return CandidateBatch.at_fidelity(designs, 1000.0, 19500.0)

        def predict(self, candidates):
            assert finished
            self.calls += 1
            return -np.square(candidates.mixtures).sum(1)

    oracle = Oracle()

    def run(p):
        return run_method_seed_benchmark(
            oracle,
            p,
            [MethodSpec("spade", CONFIG)],
            reference_utility=np.array([-2.0, 0.0]),
            config=SeedBenchmarkConfig(
                seeds=(38,), candidate_budget=3, results_dir=tmp_path
            ),
            write_results=False,
        )

    p = problem()
    result = run(p)
    assert result.per_seed.status.tolist() == ["success"]
    import json

    resolved = json.loads(result.per_seed.method_config_json.iloc[0])
    assert resolved["support_k"] == 10 and resolved["diff_hidden"] == 8
    singleton = replace(
        p,
        train_designs=p.train_designs[:1],
        train_context=p.train_context[:1],
        train_utility=p.train_utility[:1],
    )
    failed = run(singleton)
    assert failed.per_seed.status.tolist() == ["failed"] and oracle.calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diff_t_dim": 3},
        {"diff_epochs": True},
        {"diff_hidden": 0},
        {"diff_steps": 2},
        {"calib_mc_samples": 1},
        {"acq_mc_samples": 0},
        {"support_k": 1},
        {"diff_beta_end": 1},
        {"diff_beta_start": 0.1},
        {"ea_pop": 2},
        {"ea_crossover": 1.1},
        {"diff_lr": float("nan")},
        {"calib_weight": -1},
        {"support_transform": 1},
        {"knn_chunk": 0},
        {"ea_mut_sigma_min": 1.0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SpadeMethod(**kwargs)


def test_notice_is_packaged():
    notice = (
        files("llm_design_bench").joinpath("third_party/SPADE_LICENSE.txt").read_text()
    )
    assert (
        "Copyright (c) 2026 SPADE Authors" in notice
        and "Permission is hereby granted" in notice
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda():
    result = make_method("spade", **CONFIG).run(
        problem(), RunContext(method_seed=38, candidate_budget=3, device="cuda")
    )
    assert result.candidates.is_cuda
