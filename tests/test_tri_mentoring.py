import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from llm_design_bench.optimizers import make_method
from llm_design_bench.optimizers.mentoring_utils import (
    MentoringData,
    differentiable_sgd_step,
    mentor_proxy,
    pairwise_consensus,
    sample_local_designs,
    soft_label_meta_loss,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace

CONFIG = dict(
    hidden_size=8,
    surrogate_epochs=2,
    batch_size=2,
    solver_steps=2,
    neighbor_samples=4,
    surrogate_learning_rate=0.01,
)


def problem(simplex=True, count=6):
    x = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9], [0.5, 0.5], [0.2, 0.8]]
    )[:count]
    return OfflineProblem(
        x,
        torch.arange(count, dtype=x.dtype).unsqueeze(1),
        torch.linspace(-2, -1, count),
        torch.tensor([8.0]),
        SimplexSpace(2)
        if simplex
        else BoxSpace(torch.tensor([[0.0, 1.0], [0.0, 1.0]])),
        ProblemMetadata(task_name="mentoring-test"),
    )


def test_pairwise_majority_uses_unique_pairs_and_only_marks_disagreement():
    pairs, labels, masks = pairwise_consensus(
        torch.tensor([[3.0, 2.0, 1.0], [3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    )
    assert pairs.tolist() == [[0, 0, 1], [1, 2, 2]]
    assert labels.tolist() == [1.0, 1.0, 0.0]
    assert masks.tolist() == [
        [False, False, True],
        [False, False, False],
        [True, True, False],
    ]
    _, ties, tied_masks = pairwise_consensus(torch.ones(3, 2))
    assert ties.tolist() == [0.0]
    assert not tied_masks.any()


def test_virtual_update_is_differentiable_and_does_not_mutate_model():
    model = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(0.4)
    label = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    before = model.weight.detach().clone()
    loss = (model(torch.ones(1, 1, dtype=torch.float64)).squeeze() - label).square()
    updated = differentiable_sgd_step(model, loss, 0.1)
    assert torch.equal(model.weight, before)
    assert torch.allclose(
        updated["weight"], torch.tensor([[0.38]], dtype=torch.float64)
    )
    derivative = torch.autograd.grad(updated["weight"].sum(), label)[0]
    assert derivative.item() == pytest.approx(0.2)


class ScalarProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))

    def forward(self, x):
        return x[:, 0] * self.weight


def meta_fixture():
    features = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    data = MentoringData(features, -features[:, 0], torch.zeros(1), torch.ones(1))
    return ScalarProxy(), features, torch.tensor([[0], [1]]), data


def test_soft_label_meta_gradient_matches_finite_difference():
    model, x, pairs, data = meta_fixture()
    label = torch.tensor([0.6], dtype=torch.float64, requires_grad=True)
    loss = soft_label_meta_loss(model, x, pairs, label, data, 0.1)
    gradient = torch.autograd.grad(loss, label)[0].item()
    epsilon = 1e-5
    upper = soft_label_meta_loss(model, x, pairs, label.detach() + epsilon, data, 0.1)
    lower = soft_label_meta_loss(model, x, pairs, label.detach() - epsilon, data, 0.1)
    assert abs(gradient) > 1e-4
    assert gradient == pytest.approx(((upper - lower) / (2 * epsilon)).item(), rel=1e-5)


def test_refined_labels_improve_logged_error_and_empty_pairs_are_noop():
    model, x, pairs, data = meta_fixture()
    hard_model = copy.deepcopy(model)
    labels = torch.zeros(1, dtype=torch.float64)
    before = model.weight.detach().clone()
    assert (
        mentor_proxy(
            model,
            x,
            pairs[:, :0],
            labels[:0],
            data,
            learning_rate=0.1,
            label_learning_rate=0.5,
            soft_labels=True,
        )
        == 0
    )
    assert torch.equal(before, model.weight)
    change = mentor_proxy(
        model,
        x,
        pairs,
        labels,
        data,
        learning_rate=0.1,
        label_learning_rate=0.5,
        soft_labels=True,
    )
    mentor_proxy(
        hard_model,
        x,
        pairs,
        labels,
        data,
        learning_rate=0.1,
        label_learning_rate=0.5,
        soft_labels=False,
    )
    assert 0 < change <= 1
    assert F.mse_loss(model(x), data.utility) < F.mse_loss(hard_model(x), data.utility)


@pytest.mark.parametrize("simplex", [True, False])
def test_neighborhoods_keep_design_constraints_and_target_context(simplex):
    p = problem(simplex)
    parameters = p.design_space.to_unconstrained(p.train_designs[:1])
    neighbors = sample_local_designs(
        p, parameters, 16, 0.1, torch.Generator().manual_seed(3)
    )
    p.design_space.validate(neighbors)
    assert torch.equal(p.features_at_target(neighbors)[:, -1], torch.full((16,), 8.0))
    assert not neighbors.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_run_is_reproducible_rng_isolated_and_does_not_mutate_input(dtype):
    p = problem()
    snapshot = p.train_designs.clone()
    method = make_method("tri_mentoring", **CONFIG)
    rng_state = torch.random.get_rng_state().clone()
    # More starts than logs ensures seeded random initialization is exercised.
    context = RunContext(method_seed=38, candidate_budget=7, dtype=dtype)
    first = method.run(p, context)
    second = method.run(p, context)
    assert torch.equal(first.candidates, second.candidates)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(p.train_designs, snapshot)
    assert first.candidates.dtype == dtype
    changed = method.run(p, RunContext(method_seed=39, candidate_budget=7, dtype=dtype))
    assert not torch.equal(first.candidates, changed.candidates)


@pytest.mark.parametrize("count", [1, 2, 6])
def test_small_constant_utility_data(count):
    p = problem(count=count)
    p.train_utility.fill_(-1)
    result = make_method("tri_mentoring", **CONFIG).run(
        p, RunContext(method_seed=38, candidate_budget=2)
    )
    p.design_space.validate(result.candidates)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"surrogate_epochs": True},
        {"batch_size": 0},
        {"solver_steps": 0},
        {"neighbor_samples": 1},
        {"neighbor_samples": 2.5},
        {"neighbor_noise_std": float("nan")},
        {"mentoring_learning_rate": 0},
        {"label_learning_rate": float("inf")},
        {"validation_fraction": 1},
        {"validation_fraction": float("nan")},
        {"soft_labels": "yes"},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        make_method("tri_mentoring", **kwargs)


def test_candidate_adaptation_is_reset_between_starts(monkeypatch):
    import llm_design_bench.optimizers.tri_mentoring as module

    snapshots = []
    original = module.copy.deepcopy

    def tracked_copy(value, *args, **kwargs):
        if (
            isinstance(value, list)
            and len(value) == 3
            and all(isinstance(item, nn.Module) for item in value)
        ):
            snapshots.append(
                [p.detach().clone() for m in value for p in m.parameters()]
            )
        return original(value, *args, **kwargs)

    monkeypatch.setattr(module.copy, "deepcopy", tracked_copy)
    make_method("tri_mentoring", **CONFIG).run(
        problem(), RunContext(method_seed=38, candidate_budget=3)
    )
    assert len(snapshots) == 3
    for snapshot in snapshots[1:]:
        assert all(torch.equal(a, b) for a, b in zip(snapshots[0], snapshot))
