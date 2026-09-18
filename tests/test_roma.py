import pytest
import torch
from torch import nn
from torch.func import functional_call

from llm_design_bench.optimizers import make_method
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.optimizers.probabilistic_surrogate import GaussianMLP
from llm_design_bench.optimizers.roma_utils import (
    adapt_candidate_weights,
    adaptation_loss,
    adversarial_weights,
    gaussian_nll,
    noisy_design_features,
    relative_projected_step,
    trust_region_score,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace

CONFIG = dict(
    hidden_size=8,
    surrogate_epochs=2,
    batch_size=2,
    weight_perturbation_steps=2,
    adaptation_steps=2,
    solver_steps=2,
)


def problem(count=4, simplex=True):
    x = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.5, 0.5]])[:count]
    return OfflineProblem(
        x,
        torch.arange(count, dtype=x.dtype).unsqueeze(1),
        torch.linspace(-2, -1, count),
        torch.tensor([8.0]),
        SimplexSpace(2)
        if simplex
        else BoxSpace(torch.tensor([[0.0, 1.0], [0.0, 1.0]])),
        ProblemMetadata(task_name="roma-test"),
    )


class LinearGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.4, 0.2, -0.1], dtype=torch.float64))
        self.logstd = nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))

    def forward(self, x):
        return x @ self.weight, self.logstd.expand(len(x))


def test_relative_projection_bounds_every_tensor_and_handles_zero_norms():
    reference = {"weight": torch.tensor([3.0, 4.0]), "zero": torch.zeros(2)}
    updated = relative_projected_step(
        reference,
        reference,
        (torch.tensor([10.0, 0.0]), torch.ones(2)),
        step_size=2.0,
        radius=0.1,
        ascent=True,
    )
    assert torch.allclose(updated["weight"], torch.tensor([3.5, 4.0]))
    assert torch.equal(updated["zero"], torch.zeros(2))
    stationary = relative_projected_step(
        reference,
        reference,
        (torch.zeros(2), torch.zeros(2)),
        step_size=1.0,
        radius=0.1,
        ascent=False,
    )
    assert torch.equal(stationary["weight"], reference["weight"])
    with pytest.raises(RuntimeError, match="non-finite"):
        relative_projected_step(
            reference,
            reference,
            (torch.full((2,), float("nan")), torch.zeros(2)),
            step_size=1.0,
            radius=0.1,
            ascent=True,
        )


def test_adversarial_ascent_increases_nll_without_mutating_base_and_is_bounded():
    model = LinearGaussian()
    x = torch.tensor([[1.0, 2.0, 0.0], [2.0, 1.0, 1.0]], dtype=torch.float64)
    y = torch.zeros(2, dtype=torch.float64)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    weights = adversarial_weights(model, x, y, steps=3, radius=0.01)
    assert gaussian_nll(*functional_call(model, weights, (x,)), y) > gaussian_nll(
        *model(x), y
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
        assert (weights[name] - parameter).norm() <= 0.010001 * parameter.norm()
    # Outer gradient must reach the base weights while holding perturbations fixed.
    shifted = {
        name: p + (weights[name] - p).detach() for name, p in model.named_parameters()
    }
    loss = gaussian_nll(*functional_call(model, shifted, (x,)), y)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0


def test_gaussian_nll_is_row_wise_and_matches_distribution():
    mean = torch.tensor([0.2, 0.4])
    logstd = torch.tensor([-1.0, -0.5])
    labels = torch.tensor([0.0, 1.0])
    expected = -torch.distributions.Normal(mean, logstd.exp()).log_prob(labels).mean()
    constant = 0.5 * torch.log(torch.tensor(2 * torch.pi))
    assert torch.allclose(gaussian_nll(mean, logstd, labels), expected - constant)
    with pytest.raises(ValueError, match="matching"):
        gaussian_nll(mean, logstd, labels[:, None])


def test_design_noise_preserves_fidelity_and_input_and_uses_local_generator():
    x = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    snapshot = x.clone()
    first = noisy_design_features(x, 2, 0.2, torch.Generator().manual_seed(38))
    second = noisy_design_features(x, 2, 0.2, torch.Generator().manual_seed(38))
    assert torch.equal(first, second)
    assert torch.equal(first[:, 2:], x[:, 2:])
    assert not torch.equal(first[:, :2], x[:, :2])
    assert torch.equal(snapshot, x)


def test_second_order_smoothness_gradient_matches_finite_difference():
    p = problem(simplex=False).to("cpu", torch.float64)
    data = MentoringData.from_problem(p, 1e-6)
    model = LinearGaussian()
    coordinates = p.design_space.to_unconstrained(p.train_designs[:1]).requires_grad_(
        True
    )
    weights = dict(model.named_parameters())
    previous_mean = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)

    def loss(values):
        return adaptation_loss(
            model,
            values,
            coordinates,
            previous_mean,
            p,
            data,
            consistency_weight=0.0,
            uncertainty_weight=0.0,
        )

    value = loss(weights)
    gradient = torch.autograd.grad(value, model.weight, retain_graph=True)[0]
    assert torch.autograd.grad(value, previous_mean, allow_unused=True)[0] is None
    for i in range(3):
        delta = torch.zeros_like(model.weight)
        delta[i] = 1e-5
        upper = loss({**weights, "weight": model.weight.detach() + delta})
        lower = loss({**weights, "weight": model.weight.detach() - delta})
        assert gradient[i].item() == pytest.approx(
            ((upper - lower) / 2e-5).item(), abs=1e-7
        )
    assert gradient[:2].abs().sum() > 0
    assert gradient[2] == 0  # Constant target fidelity is not a search coordinate.


def test_adaptation_decreases_smoothness_and_keeps_base_fixed():
    p = problem(simplex=False).to("cpu", torch.float64)
    data = MentoringData.from_problem(p, 1e-6)
    model = LinearGaussian()
    point = p.design_space.to_unconstrained(p.train_designs[:1]).requires_grad_(True)
    weights = dict(model.named_parameters())
    before = {name: v.detach().clone() for name, v in weights.items()}
    target = model(data.at_target(p, p.train_designs[:1]))[0].detach()
    adapted = adapt_candidate_weights(
        model,
        point,
        target,
        p,
        data,
        steps=3,
        radius=0.01,
        consistency_weight=0.0,
        uncertainty_weight=0.0,
    )
    original_loss = adaptation_loss(
        model,
        weights,
        point,
        target,
        p,
        data,
        consistency_weight=0.0,
        uncertainty_weight=0.0,
    )
    final_loss = adaptation_loss(
        model,
        adapted,
        point,
        target,
        p,
        data,
        consistency_weight=0.0,
        uncertainty_weight=0.0,
    )
    assert final_loss < original_loss
    for name, param in weights.items():
        assert torch.equal(param, before[name])
        assert (adapted[name] - param).norm() <= 0.010001 * param.norm()
        assert not adapted[name].requires_grad


def test_trust_penalty_is_score_space_and_uses_logstd():
    mean = torch.tensor([3.0], requires_grad=True)
    logstd = torch.tensor([0.5])
    initial = torch.tensor([0.5], requires_grad=True)
    value = trust_region_score(
        mean, logstd, initial, uncertainty_weight=1.0, region=4.0
    )
    assert value.item() == pytest.approx(2.0)
    derivative, initial_derivative = torch.autograd.grad(
        value, (mean, initial), allow_unused=True
    )
    assert derivative.item() == pytest.approx(0.5)
    assert initial_derivative is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reproducibility_rng_isolation_and_no_input_mutation(dtype):
    p = problem()
    snapshot = p.train_designs.clone()
    rng = torch.random.get_rng_state().clone()
    method = make_method("roma", **CONFIG)
    context = RunContext(method_seed=38, candidate_budget=6, dtype=dtype)
    first, second = method.run(p, context), method.run(p, context)
    assert torch.equal(first.candidates, second.candidates)
    assert first.candidates.dtype == dtype
    assert torch.equal(snapshot, p.train_designs)
    assert torch.equal(rng, torch.random.get_rng_state())
    other = method.run(p, RunContext(method_seed=39, candidate_budget=6, dtype=dtype))
    assert not torch.equal(first.candidates, other.candidates)


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("simplex", [True, False])
def test_small_constant_logged_data(count, simplex):
    p = problem(count, simplex)
    p.train_utility.fill_(-1.0)
    result = make_method("roma", **CONFIG).run(
        p, RunContext(method_seed=38, candidate_budget=2)
    )
    p.design_space.validate(result.candidates)


@pytest.mark.parametrize("override", [{"weight_radius": 0.0}, {"adaptation_steps": 0}])
def test_ablations_disable_corresponding_updates(override):
    result = make_method("roma", **(CONFIG | override)).run(
        problem(), RunContext(method_seed=38, candidate_budget=2)
    )
    assert result.diagnostics["adaptation_updates"] == 0
    assert result.diagnostics["total_adapted_weight_delta_norm"] == 0
    if "weight_radius" in override:
        assert result.training_summary["weight_perturbation_updates"] == 0
        assert result.training_summary["total_adversarial_perturbation_norm"] == 0


def test_candidate_state_is_independent_and_base_is_never_modified(monkeypatch):
    import llm_design_bench.optimizers.roma as module

    seen_base, seen_previous, last_adapted = [], [], []
    original = module.adapt_candidate_weights

    def tracked(model, coordinates, previous_mean, p, data, **kwargs):
        current_features = data.at_target(
            p, p.design_space.from_unconstrained(coordinates)
        )
        seen_base.append([v.detach().clone() for v in model.parameters()])
        # First step of each candidate uses base; later steps use its own old model.
        expected = (
            model(current_features)[0]
            if len(seen_base) % 2
            else functional_call(model, last_adapted[-1], (current_features,))[0]
        )
        assert torch.allclose(previous_mean, expected)
        assert not previous_mean.requires_grad
        seen_previous.append(previous_mean.clone())
        adapted = original(model, coordinates, previous_mean, p, data, **kwargs)
        last_adapted.append(adapted)
        return adapted

    monkeypatch.setattr(module, "adapt_candidate_weights", tracked)
    result = make_method("roma", **CONFIG).run(
        problem(), RunContext(method_seed=38, candidate_budget=3)
    )
    assert len(seen_base) == 6
    for snapshot in seen_base[1:]:
        assert all(torch.equal(a, b) for a, b in zip(seen_base[0], snapshot))
    assert result.diagnostics["adaptation_updates"] == 12
    assert result.diagnostics["total_adapted_weight_delta_norm"] > 0
    assert result.training_summary["total_adversarial_perturbation_norm"] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"surrogate_epochs": True},
        {"batch_size": 0},
        {"solver_steps": 0},
        {"adaptation_steps": -1},
        {"weight_perturbation_steps": 0},
        {"weight_radius": -1},
        {"input_noise_std": float("nan")},
        {"region": 0},
        {"gradient_clip": float("inf")},
        {"consistency_weight": -1},
        {"uncertainty_weight": -1},
        {"initial_min_std": 0.3},
        {"initial_max_std": 0},
        {"solver_learning_rate": True},
        {"surrogate_learning_rate": 0},
        {"minimum_std": 0},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        make_method("roma", **kwargs)


def test_softplus_network_and_explicit_metadata():
    model = GaussianMLP(
        3,
        hidden_size=4,
        num_layers=2,
        initial_min_std=0.1,
        initial_max_std=0.2,
        activation="softplus",
    )
    assert sum(isinstance(m, nn.Softplus) for m in model.modules()) == 2
    metadata = make_method("roma").metadata
    assert metadata.display_name == "RoMA adaptation"
    assert metadata.implementation_kind.value == "multi_fidelity_adaptation"
    assert "explicit_per_tensor_relative_weight_projection" in metadata.adaptations
