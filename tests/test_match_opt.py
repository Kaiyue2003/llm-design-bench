import numpy as np
import pytest
import torch
from torch import nn

from llm_design_bench.evaluation.seed_runner import (
    MethodSpec,
    SeedBenchmarkConfig,
    run_method_seed_benchmark,
)
from llm_design_bench.optimizers import make_method
from llm_design_bench.optimizers.match_opt import (
    MatchingMLP,
    fidelity_buckets,
    gradient_integral,
    search_matching_model,
    trajectory_pairs,
)
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace
from llm_design_bench.types import CandidateBatch

CONFIG = {
    "embedding_dim": 2,
    "surrogate_epochs": 2,
    "batch_size": 2,
    "bucket_count": 3,
    "solver_steps": 3,
}


def problem(simplex=True, context=True):
    return OfflineProblem(
        torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5], [0.1, 0.9]]),
        torch.tensor(
            [[20.0, 10.0], [20.0, 10.0], [60.0, 10.0], [60.0, 10.0], [60.0, 11.0]]
        )
        if context
        else torch.empty(5, 0),
        torch.tensor([-2.0, -1.0, -4.0, -3.0, -5.0]),
        torch.tensor([1000.0, 19500.0]) if context else torch.empty(0),
        SimplexSpace(2)
        if simplex
        else BoxSpace(torch.tensor([[0.0, 1.0], [0.0, 1.0]])),
        ProblemMetadata(task_name="matching-test"),
    )


class Polynomial(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def forward(self, x):
        return self.weight * x[:, 0] ** 2 + 3 * x[:, 1]


def test_left_quadrature_matches_analytic_formula_not_endpoint_difference():
    model = Polynomial()
    start = torch.tensor([[0.0, 2.0], [1.0, 2.0]], dtype=torch.float64)
    end = torch.tensor([[1.0, 2.0], [3.0, 2.0]], dtype=torch.float64)
    result = gradient_integral(model, start, end, design_dim=1)
    delta = end[:, 0] - start[:, 0]
    expected = model.weight * 2 * (start[:, 0] + delta * 0.4) * delta
    assert torch.allclose(result, expected)
    assert not torch.allclose(result, model(end) - model(start))
    assert not start.requires_grad and not end.requires_grad


def test_linear_integral_sign_and_matching_weight_gradient():
    model = nn.Sequential(nn.Linear(2, 1, bias=False), nn.Flatten(0)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[2.0, 3.0]], dtype=torch.float64))
    start = torch.tensor([[0.0, 4.0]], dtype=torch.float64)
    end = torch.tensor([[1.0, 4.0]], dtype=torch.float64)
    predicted = gradient_integral(model, start, end, design_dim=1)
    assert predicted.item() == pytest.approx(2.0)
    assert gradient_integral(model, end, start, design_dim=1).item() == pytest.approx(
        -2.0
    )
    loss = (predicted - 4).square().mean()
    loss.backward()
    assert model[0].weight.grad[0, 0] < 0
    assert model[0].weight.grad[0, 1] == 0  # context contributes no displacement


def test_higher_order_parameter_gradient_matches_finite_difference():
    model = Polynomial()
    start = torch.tensor([[0.1, 2.0], [0.5, 2.0]], dtype=torch.float64)
    end = start + torch.tensor([[0.4, 0.0], [0.3, 0.0]], dtype=torch.float64)

    def objective():
        return (gradient_integral(model, start, end, design_dim=1) - 1).square().mean()

    gradient = torch.autograd.grad(objective(), model.weight)[0].item()
    before = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.copy_(before + 1e-5)
    upper = objective().item()
    with torch.no_grad():
        model.weight.copy_(before - 1e-5)
    lower = objective().item()
    assert gradient == pytest.approx((upper - lower) / 2e-5, abs=1e-8)


def test_summed_gradient_matches_full_batch_jacobian():
    torch.manual_seed(4)
    model = MatchingMLP(3, 2).double()
    start = torch.tensor([[0.2, 0.4, 1.0], [0.5, 0.3, 1.0]], dtype=torch.float64)
    end = start + torch.tensor(
        [[0.1, -0.1, 0.0], [-0.2, 0.2, 0.0]], dtype=torch.float64
    )
    expected = torch.zeros(2, dtype=torch.float64)
    for i in range(5):
        point = start + (end - start) * i / 5
        jacobian = torch.autograd.functional.jacobian(model, point, create_graph=True)
        expected = (
            expected + (torch.einsum("aab->ab", jacobian) * (end - start)).sum(-1) / 5
        )
    actual = gradient_integral(model, start, end, design_dim=2)
    assert torch.allclose(actual, expected, atol=1e-12)
    actual_grads = torch.autograd.grad(
        actual.sum(), tuple(model.parameters()), allow_unused=True
    )
    expected_grads = torch.autograd.grad(
        expected.sum(), tuple(model.parameters()), allow_unused=True
    )
    for a, b in zip(actual_grads, expected_grads):
        assert (a is None and b is None) or torch.allclose(a, b, atol=1e-12)


def test_pairs_are_monotone_exact_context_local_and_seeded():
    p = problem()
    groups = fidelity_buckets(p, 128)
    assert [len(group) for group in groups] == [2, 2]
    rng = torch.random.get_rng_state().clone()
    pairs = trajectory_pairs(groups, torch.Generator().manual_seed(38))
    assert pairs.shape == (2, 2)
    assert torch.equal(
        pairs, trajectory_pairs(groups, torch.Generator().manual_seed(38))
    )
    assert torch.equal(rng, torch.random.get_rng_state())
    left, right = pairs.unbind(1)
    assert torch.equal(p.train_context[left], p.train_context[right])
    assert (p.train_utility[right] >= p.train_utility[left]).all()
    assert 4 not in pairs  # same scale, different step count is a singleton
    with pytest.raises(ValueError, match="cross fidelity"):
        gradient_integral(
            MatchingMLP(4, 2), p.train_features[:1], p.train_features[2:3], design_dim=2
        )


def test_context_free_bucket_remainder_ties_and_small_groups():
    p = problem(context=False)
    p.train_utility.fill_(1.0)
    groups = fidelity_buckets(p, 2)
    assert [len(bucket) for bucket in groups[0]] == [2, 3]
    pairs = trajectory_pairs(groups, torch.Generator().manual_seed(38))
    assert pairs.shape == (2, 2)
    assert (pairs[:, 0] < 2).all() and (pairs[:, 1] >= 2).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("simplex,context", [(True, True), (False, False)])
def test_run_contract_reproducibility_and_visible_row_coverage(dtype, simplex, context):
    p = problem(simplex, context)
    snapshot = [x.clone() for x in (p.train_designs, p.train_context, p.train_utility)]
    rng = torch.random.get_rng_state().clone()
    method = make_method("match_opt", **CONFIG)
    ctx = RunContext(method_seed=38, candidate_budget=7, dtype=dtype)
    first, second = method.run(p, ctx), method.run(p, ctx)
    assert torch.equal(first.candidates, second.candidates)
    assert first.training_summary == second.training_summary
    assert torch.equal(rng, torch.random.get_rng_state())
    for old, new in zip(snapshot, (p.train_designs, p.train_context, p.train_utility)):
        assert torch.equal(old, new)
    assert first.candidates.shape == (7, 2) and first.candidates.dtype == dtype
    assert first.training_summary["training_updates"] == 6
    assert first.training_summary["supervised_rows_seen"] == 10
    assert first.training_summary["singleton_rows_value_only"] == (1 if context else 0)
    assert first.training_summary["pair_samples_used"] > 0
    other = method.run(p, RunContext(method_seed=39, candidate_budget=7, dtype=dtype))
    assert not torch.equal(first.candidates, other.candidates)


def test_duplicate_designs_and_constant_labels_are_finite_and_diagnosed():
    p = problem()
    p.train_designs[1].copy_(p.train_designs[0])
    p.train_utility.fill_(0.0)
    result = make_method("match_opt", **CONFIG).run(
        p, RunContext(method_seed=38, candidate_budget=1)
    )
    assert result.training_summary["zero_displacement_trajectory_pairs_total"] == 2
    assert result.candidates.isfinite().all()


@pytest.mark.parametrize("simplex", [True, False])
def test_search_improves_linear_utility_and_fixes_target_context(simplex):
    p = problem(simplex).to("cpu", torch.float64)
    data = MentoringData.from_problem(p, 1e-6)
    seen = []

    class Linear(nn.Module):
        def forward(self, x):
            seen.append(x.detach().clone())
            return x[:, 0]

    starts = p.train_designs[2:3]
    result = search_matching_model(
        Linear(), data, p, starts, steps=10, learning_rate=0.05
    )
    assert result[0, 0] > starts[0, 0]
    target = (p.target_context - data.feature_mean[2:]) / data.feature_std[2:]
    assert all(torch.allclose(x[:, 2:], target.unsqueeze(0)) for x in seen)


def test_evaluator_only_runs_after_generation_and_never_for_unsupported_data(
    tmp_path, monkeypatch
):
    method_type = type(make_method("match_opt"))
    original = method_type.optimize
    completed = []

    def tracked(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        completed.append(True)
        return result

    monkeypatch.setattr(method_type, "optimize", tracked)

    class Evaluator:
        calls = 0

        def at_target_fidelity(self, x):
            return CandidateBatch.at_fidelity(x, 1000.0, 19500.0)

        def predict(self, batch):
            assert completed
            self.calls += 1
            return -np.square(batch.mixtures).sum(1)

    evaluator = Evaluator()

    def run(p):
        return run_method_seed_benchmark(
            evaluator,
            p,
            [MethodSpec("match_opt", CONFIG)],
            reference_utility=np.array([-5.0, 0.0]),
            config=SeedBenchmarkConfig(
                seeds=(38,), candidate_budget=3, results_dir=tmp_path
            ),
            write_results=False,
        )

    assert run(problem()).per_seed.status.tolist() == ["success"]
    p = problem()
    p.train_context[:, 0] = torch.arange(5.0)
    result = run(p)
    assert result.per_seed.status.tolist() == ["failed"]
    assert "no eligible pairs" in result.per_seed.error_message.iloc[0]
    assert evaluator.calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embedding_dim": True},
        {"surrogate_epochs": 0},
        {"batch_size": 0},
        {"bucket_count": 1},
        {"quadrature_nodes": 1.5},
        {"solver_steps": -1},
        {"matching_weight": 0},
        {"minimum_std": -1},
        {"solver_learning_rate": float("nan")},
        {"surrogate_learning_rate": float("inf")},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_method("match_opt", **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_run():
    result = make_method("match_opt", **CONFIG).run(
        problem(), RunContext(method_seed=38, candidate_budget=2, device="cuda")
    )
    assert result.candidates.is_cuda and result.candidates.isfinite().all()
