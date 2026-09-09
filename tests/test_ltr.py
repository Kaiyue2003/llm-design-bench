import math

import pytest
import torch
from torch import nn

from llm_design_bench.optimizers import make_method
from llm_design_bench.optimizers.ltr import (
    listnet_loss,
    prediction_statistics,
    sample_lists,
    search_ranker,
)
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace

CONFIG = {
    "hidden_size": 8,
    "surrogate_epochs": 3,
    "batch_size": 2,
    "list_length": 3,
    "lists_per_epoch": 5,
    "validation_lists": 3,
    "solver_steps": 3,
}


def problem(count=4, simplex=True, context=True):
    x = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]])[:count]
    return OfflineProblem(
        x,
        torch.arange(count, dtype=x.dtype).unsqueeze(1)
        if context
        else torch.empty(count, 0),
        torch.linspace(-2, -1, count),
        torch.tensor([8.0]) if context else torch.empty(0),
        SimplexSpace(2)
        if simplex
        else BoxSpace(torch.tensor([[0.0, 1.0], [0.0, 1.0]])),
        ProblemMetadata(task_name="ltr-test"),
    )


def test_listnet_formula_direction_shift_invariance_and_singleton_dimensions():
    predictions = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    utilities = torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float64)
    loss = listnet_loss(predictions, utilities)
    assert loss.item() == pytest.approx(math.log(3))
    gradient = torch.autograd.grad(loss, predictions)[0]
    assert gradient[0, -1] < 0 < gradient[0, 0]
    assert listnet_loss(predictions - gradient, utilities) < loss
    assert torch.allclose(loss, listnet_loss(predictions + 1000, utilities - 1000))
    assert torch.allclose(gradient, predictions.softmax(-1) - utilities.softmax(-1))
    assert listnet_loss(torch.ones(1, 1), torch.ones(1, 1)).item() == 0
    assert listnet_loss(torch.tensor([[1e4, -1e4]]), torch.zeros(1, 2)).isfinite()
    with pytest.raises(ValueError, match="matching"):
        listnet_loss(predictions.squeeze(0), utilities)
    with pytest.raises(RuntimeError, match="non-finite"):
        listnet_loss(predictions * float("nan"), utilities)


def test_list_sampling_is_local_bounded_without_replacement():
    rng = torch.random.get_rng_state().clone()

    def sample():
        return sample_lists(
            4, 8, 5, torch.Generator().manual_seed(38), torch.device("cpu")
        )

    first = sample()
    assert first.shape == (5, 4)
    assert torch.equal(first, sample())
    assert all(len(row.unique()) == 4 for row in first)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_prediction_calibration_uses_mean_std_with_constant_fallback():
    mean, scale = prediction_statistics(torch.tensor([1.0, 3.0]), 1e-6)
    assert mean.item() == 2 and scale.item() == 1
    mean, scale = prediction_statistics(torch.tensor([4.0]), 1e-6)
    assert mean.item() == 4 and scale.item() == 1


@pytest.mark.parametrize("simplex", [True, False])
def test_search_ascends_utility_and_holds_fidelity_and_model_fixed(simplex):
    p = problem(simplex=simplex).to("cpu", torch.float64)
    data = MentoringData.from_problem(p, 1e-6)
    seen = []

    class LinearRanker(nn.Module):
        def forward(self, x):
            seen.append(x.detach().clone())
            return x[:, 0] + 3 * x[:, -1]

    start = p.train_designs[2:3].clone()
    result = search_ranker(
        LinearRanker(),
        data,
        p,
        start,
        torch.tensor(0.0),
        torch.tensor(1.0),
        steps=10,
        learning_rate=0.05,
    )
    assert result[0, 0] > start[0, 0]
    p.design_space.validate(result)
    target = (p.target_context - data.feature_mean[-1:]) / data.feature_std[-1:]
    assert all(torch.allclose(features[:, -1], target) for features in seen)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("simplex,has_context", [(True, True), (False, False)])
def test_run_reproducibility_rng_isolation_and_exact_budget(
    dtype, simplex, has_context
):
    p = problem(simplex=simplex, context=has_context)
    before = [t.clone() for t in (p.train_designs, p.train_context, p.train_utility)]
    rng = torch.random.get_rng_state().clone()
    method = make_method("ltr", **CONFIG)
    ctx = RunContext(method_seed=38, candidate_budget=6, dtype=dtype)
    first, second = method.run(p, ctx), method.run(p, ctx)
    assert torch.equal(first.candidates, second.candidates)
    assert first.training_summary == second.training_summary
    assert first.candidates.shape == (6, 2) and first.candidates.dtype == dtype
    assert first.training_summary["training_updates"] == 9  # includes short batches
    assert (
        first.training_summary["validation_unit"] == "sampled_lists_shared_visible_rows"
    )
    assert torch.equal(rng, torch.random.get_rng_state())
    for original, tensor in zip(
        before, (p.train_designs, p.train_context, p.train_utility)
    ):
        assert torch.equal(original, tensor)
    other = method.run(p, RunContext(method_seed=39, candidate_budget=6, dtype=dtype))
    assert not torch.equal(first.candidates, other.candidates)


@pytest.mark.parametrize("count", [1, 2, 4])
def test_uninformative_logs_are_explicit_and_do_not_train(count, monkeypatch):
    p = problem(count)
    p.train_utility.fill_(-1)
    method = make_method("ltr", **CONFIG)

    def forbidden(*args):
        raise AssertionError("uninformative labels must not train")

    monkeypatch.setattr(method, "_fit", forbidden)
    result = method.run(p, RunContext(method_seed=38, candidate_budget=7))
    p.design_space.validate(result.candidates)
    assert result.training_summary["training_status"] == "skipped_uninformative_labels"
    assert result.diagnostics["solver_steps"] == 0
    assert result.candidates.shape == (7, 2)


def test_two_rows_and_single_candidate_train_with_bounded_lists():
    result = make_method("ltr", **(CONFIG | {"list_length": 100})).run(
        problem(2),
        RunContext(method_seed=38, candidate_budget=1),
    )
    assert result.training_summary["effective_list_length"] == 2
    assert result.training_summary["training_status"] == "trained"


def test_ranker_learns_and_calibration_uses_visible_predictions_only(monkeypatch):
    import llm_design_bench.optimizers.ltr as module

    original = module.prediction_statistics
    calibrated = []

    def record(predictions, minimum_std):
        calibrated.append(predictions.clone())
        return original(predictions, minimum_std)

    monkeypatch.setattr(module, "prediction_statistics", record)
    result = make_method(
        "ltr",
        **(
            CONFIG
            | {
                "surrogate_epochs": 10,
                "list_length": 4,
                "surrogate_learning_rate": 0.01,
            }
        ),
    ).run(problem(), RunContext(method_seed=38, candidate_budget=7))
    history = result.training_summary["validation_loss_history"]
    assert history[-1] < history[0]
    assert len(calibrated) == 1 and calibrated[0].shape == (4,)
    assert result.diagnostics["prediction_mean"] == pytest.approx(
        calibrated[0].mean().item()
    )
    assert result.diagnostics["prediction_scale"] == pytest.approx(
        calibrated[0].std(unbiased=False).item(),
    )


def test_training_materializes_only_one_list_batch_at_a_time(monkeypatch):
    import llm_design_bench.optimizers.ltr as module

    requested = []
    original = module.sample_lists

    def record(count, length, batch_size, generator, device):
        requested.append(batch_size)
        return original(count, length, batch_size, generator, device)

    monkeypatch.setattr(module, "sample_lists", record)
    make_method("ltr", **CONFIG).run(
        problem(), RunContext(method_seed=38, candidate_budget=2)
    )
    assert requested == [3] + [2, 2, 1] * 3  # fixed validation then lazy training


def test_best_checkpoint_is_a_snapshot_and_search_model_is_frozen(monkeypatch):
    import llm_design_bench.optimizers.ltr as module

    original_loss, original_model = module.listnet_loss, module.MentoringMLP
    models, snapshots = [], []

    def make_model(*args):
        model = original_model(*args)
        models.append(model)
        return model

    def loss(prediction, utility):
        if torch.is_grad_enabled():
            return original_loss(prediction, utility)
        snapshots.append({k: v.clone() for k, v in models[0].state_dict().items()})
        return prediction.new_tensor(len(snapshots))  # force first epoch selection

    monkeypatch.setattr(module, "MentoringMLP", make_model)
    monkeypatch.setattr(module, "listnet_loss", loss)
    method = make_method("ltr", **(CONFIG | {"validation_lists": 1}))
    p = problem()
    ctx = RunContext(method_seed=38, candidate_budget=2)
    model, summary = method._fit(
        MentoringData.from_problem(p, 1e-6), ctx, ctx.make_generator()
    )
    assert summary["selected_epoch"] == 1
    assert any(not torch.equal(snapshots[0][k], snapshots[-1][k]) for k in snapshots[0])
    assert all(torch.equal(v, snapshots[0][k]) for k, v in model.state_dict().items())
    assert not any(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"list_length": 1},
        {"lists_per_epoch": 0},
        {"validation_lists": 0},
        {"hidden_size": True},
        {"surrogate_epochs": 1.5},
        {"batch_size": -1},
        {"solver_steps": 0},
        {"minimum_std": 0},
        {"weight_decay": -1},
        {"solver_learning_rate": float("nan")},
        {"surrogate_learning_rate": float("inf")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make_method("ltr", **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_small_run():
    result = make_method("ltr", **CONFIG).run(
        problem(),
        RunContext(method_seed=38, candidate_budget=2, device="cuda"),
    )
    assert result.candidates.is_cuda and result.candidates.isfinite().all()
