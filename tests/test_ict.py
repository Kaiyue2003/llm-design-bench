import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from llm_design_bench.optimizers import make_method
from llm_design_bench.optimizers.ict_utils import (
    co_teach_round,
    exchanged_small_loss_indices,
    reweight_and_update_proxy,
    sample_weight_meta_loss,
    weighted_pseudo_loss,
)
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import SimplexSpace

CONFIG = dict(
    hidden_size=8,
    surrogate_epochs=2,
    batch_size=2,
    adaptation_steps=2,
    solver_steps=2,
    neighbor_samples=4,
    remember_count=2,
    surrogate_learning_rate=0.01,
)


def problem(count=4):
    x = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]])[:count]
    return OfflineProblem(
        x,
        torch.arange(count, dtype=x.dtype).unsqueeze(1),
        torch.linspace(-2, -1, count),
        torch.tensor([8.0]),
        SimplexSpace(2),
        ProblemMetadata(task_name="ict-test"),
    )


class ScalarProxy(nn.Module):
    def __init__(self, value=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value, dtype=torch.float64))

    def forward(self, x):
        return x[:, 0] * self.weight


def fixture():
    x = torch.tensor([[-1.0], [2.0]], dtype=torch.float64)
    data = MentoringData(x, -x[:, 0], torch.zeros(1), torch.ones(1))
    return ScalarProxy(), x, x[:, 0].clone(), data


def test_selection_exchanges_other_students_low_losses_with_stable_ties():
    selected = exchanged_small_loss_indices(
        torch.tensor([[0.0, 3.0, 1.0, 1.0], [5.0, 0.0, 2.0, 3.0]]),
        torch.zeros(4),
        2,
    )
    assert selected[0].tolist() == [1, 2]
    assert selected[1].tolist() == [0, 2]


def test_weighting_is_per_selected_row_and_teacher_labels_are_detached():
    model, x, labels, _ = fixture()
    labels.requires_grad_(True)
    weights = torch.tensor([0.0, 2.0], dtype=torch.float64)
    loss = weighted_pseudo_loss(model, x, labels, weights)
    assert loss.item() == pytest.approx(2.56)
    assert torch.autograd.grad(loss, labels, allow_unused=True)[0] is None
    with pytest.raises(ValueError, match="matching"):
        weighted_pseudo_loss(model, x, labels, weights[:, None])


def test_sample_weight_meta_gradient_matches_finite_difference_without_mutation():
    model, x, labels, data = fixture()
    before = model.weight.detach().clone()
    weights = torch.ones(2, dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(
        sample_weight_meta_loss(model, x, labels, weights, data, 0.1), weights
    )[0]
    for i in range(2):
        delta = torch.zeros_like(weights)
        delta[i] = 1e-5
        upper = sample_weight_meta_loss(
            model, x, labels, weights.detach() + delta, data, 0.1
        )
        lower = sample_weight_meta_loss(
            model, x, labels, weights.detach() - delta, data, 0.1
        )
        assert gradient[i].item() == pytest.approx(
            ((upper - lower) / 2e-5).item(), rel=1e-5
        )
    assert gradient.abs().sum() > 0
    assert torch.equal(before, model.weight)


def test_reweighting_improves_logged_loss_and_real_step_uses_original_parameters():
    model, x, labels, data = fixture()
    unweighted = copy.deepcopy(model)
    before = model.weight.detach().clone()
    weights = reweight_and_update_proxy(
        model,
        x,
        labels,
        data,
        learning_rate=0.1,
        weight_learning_rate=0.1,
        reweighting=True,
    )
    reweight_and_update_proxy(
        unweighted,
        x,
        labels,
        data,
        learning_rate=0.1,
        weight_learning_rate=0.1,
        reweighting=False,
    )
    assert weights.shape == labels.shape
    assert ((weights >= 0) & (weights <= 2)).all()
    expected = (
        before - 0.1 * (weights * 2 * (before * x[:, 0] - labels) * x[:, 0]).mean()
    )
    assert torch.allclose(model.weight, expected)
    assert F.mse_loss(model(x), data.utility) < F.mse_loss(unweighted(x), data.utility)


def test_co_teaching_keeps_teacher_unchanged_and_exchanges_preupdate_selections(
    monkeypatch,
):
    import llm_design_bench.optimizers.ict_utils as module

    _, x, _, data = fixture()
    models = [ScalarProxy(0.2), ScalarProxy(0.4), ScalarProxy(-0.3)]
    before = models[0].weight.detach().clone()
    seen = []
    original = module.reweight_and_update_proxy

    def tracked(model, features, labels, *args, **kwargs):
        seen.append((features.clone(), labels.clone()))
        assert not labels.requires_grad
        return original(model, features, labels, *args, **kwargs)

    monkeypatch.setattr(module, "reweight_and_update_proxy", tracked)
    co_teach_round(
        models,
        0,
        x,
        data,
        remember_count=1,
        learning_rate=0.1,
        weight_learning_rate=0.1,
        reweighting=True,
    )
    assert torch.equal(models[0].weight, before)
    assert models[0].weight.grad is None
    assert len(seen) == 2
    assert all(torch.equal(features, x[:1]) for features, _ in seen)
    assert all(torch.equal(labels, (x[:1, 0] * before)) for _, labels in seen)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_run_reproducible_rng_isolated_and_input_unchanged(dtype):
    p = problem()
    before = p.train_designs.clone()
    rng = torch.random.get_rng_state().clone()
    context = RunContext(method_seed=38, candidate_budget=7, dtype=dtype)
    method = make_method("ict", **CONFIG)
    first, second = method.run(p, context), method.run(p, context)
    assert torch.equal(first.candidates, second.candidates)
    assert first.candidates.dtype == dtype
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(before, p.train_designs)
    other = method.run(p, RunContext(method_seed=39, candidate_budget=7, dtype=dtype))
    assert not torch.equal(first.candidates, other.candidates)


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("reweighting", [True, False])
def test_small_constant_logged_data_and_reweighting_ablation(count, reweighting):
    p = problem(count)
    p.train_utility.fill_(-1)
    result = make_method("ict", **CONFIG, reweighting=reweighting).run(
        p, RunContext(method_seed=38, candidate_budget=3)
    )
    p.design_space.validate(result.candidates)
    if not reweighting:
        assert result.diagnostics["sample_weight_total_absolute_change"] == 0


def test_two_stage_protocol_rotates_teachers_then_freezes_shared_ensemble(monkeypatch):
    import llm_design_bench.optimizers.ict as module

    rounds, phases, final_states = [], [], []
    original_round, original_advance = module.co_teach_round, module._advance_designs

    def tracked_round(models, teacher, features, data, **kwargs):
        rounds.append(teacher)
        assert all(p.requires_grad for m in models for p in m.parameters())
        return original_round(models, teacher, features, data, **kwargs)

    def tracked_advance(parameters, optimizer, models, p, data):
        frozen = all(
            not param.requires_grad for m in models for param in m.parameters()
        )
        phases.append((len(parameters), frozen))
        if frozen:
            final_states.append(
                [param.detach().clone() for m in models for param in m.parameters()]
            )
        return original_advance(parameters, optimizer, models, p, data)

    monkeypatch.setattr(module, "co_teach_round", tracked_round)
    monkeypatch.setattr(module, "_advance_designs", tracked_advance)
    result = make_method("ict", **CONFIG).run(
        problem(), RunContext(method_seed=38, candidate_budget=7)
    )
    assert rounds == [0, 1, 2, 0, 1, 2]
    assert phases == [(1, False), (1, False), (7, True), (7, True)]
    assert all(torch.equal(a, b) for a, b in zip(*final_states))
    assert result.diagnostics["mentoring_updates"] == 12
    assert result.diagnostics["teacher_rounds"] == 6
    assert result.diagnostics["frozen_ensemble_final_search"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"surrogate_epochs": True},
        {"batch_size": 0},
        {"adaptation_steps": 0},
        {"solver_steps": 0},
        {"neighbor_samples": 0},
        {"remember_count": 129},
        {"remember_count": 1.5},
        {"neighbor_noise_std": float("nan")},
        {"mentoring_learning_rate": 0},
        {"weight_learning_rate": float("inf")},
        {"validation_fraction": 1},
        {"validation_fraction": float("nan")},
        {"reweighting": "yes"},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        make_method("ict", **kwargs)


def test_metadata_explicitly_marks_adaptation():
    metadata = make_method("ict").metadata
    assert metadata.display_name == "ICT adaptation"
    assert metadata.implementation_kind.value == "multi_fidelity_adaptation"
    assert "functional_sgd_replaces_higher_adam" in metadata.adaptations
    assert metadata.source_commit == "6d51fcc04e7a0a60c7ad578ae4c4f743bc678ae4"


def test_constant_feature_columns_use_unit_scale_not_epsilon_amplification():
    p = problem(count=1)
    data = MentoringData.from_problem(p, 1e-6)
    assert torch.equal(data.feature_std, torch.ones(3))
    assert data.at_target(p, p.train_designs)[0, -1] == 8
