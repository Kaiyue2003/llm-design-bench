import pytest
import torch
from torch import nn

from llm_design_bench.optimizers import method_names
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.optimizers.pgs_transitions import (
    PGSTransitionConfig,
    build_logged_transitions,
    design_gradients,
    fragment_terminals,
    project_designs,
    projected_gradient_step,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata
from llm_design_bench.spaces import BoxSpace, SimplexSpace


def problem(simplex=True, context=True, dtype=torch.float64):
    first = torch.linspace(0, 1, 10, dtype=dtype).repeat(2)
    return OfflineProblem(
        torch.stack([first, 1 - first], 1),
        torch.tensor([[20.0, 10.0]] * 10 + [[60.0, 10.0]] * 10, dtype=dtype)
        if context
        else torch.empty(20, 0, dtype=dtype),
        torch.arange(20, dtype=dtype) - 20,
        torch.tensor([1000.0, 19500.0], dtype=dtype)
        if context
        else torch.empty(0, dtype=dtype),
        SimplexSpace(2)
        if simplex
        else BoxSpace(torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=dtype)),
        ProblemMetadata(task_name="pgs-transition-test"),
    )


class Linear(nn.Module):
    def __init__(self, dim, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(1, dim + 1, dtype=dtype))

    def forward(self, x):
        return x @ self.weight


def build(p, *, model=None, config=None):
    data = MentoringData.from_problem(p, 1e-6)
    model = (
        model
        if model is not None
        else Linear(data.features.shape[1], p.train_designs.dtype).to(
            p.train_designs.device
        )
    )
    return build_logged_transitions(
        p,
        model,
        data,
        torch.Generator(device=p.train_designs.device).manual_seed(38),
        config=config
        or PGSTransitionConfig(trajectories_per_group=3, top_fraction=0.5),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_simplex_projection_known_solution_boundary_and_translation_invariance(dtype):
    space = SimplexSpace(3)
    values = torch.tensor(
        [[2.0, 0.0, 0.0], [0.2, 0.3, 0.5], [-1.0, -1.0, -1.0]], dtype=dtype
    )
    projected = project_designs(values, space)
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [1 / 3, 1 / 3, 1 / 3]], dtype=dtype
    )
    assert torch.allclose(projected, expected)
    assert torch.allclose(project_designs(values + 20, space), projected, atol=1e-6)
    assert torch.allclose(project_designs(projected, space), projected)
    space.validate(projected)


def test_box_projection_and_out_of_range_actions_are_not_silently_clipped():
    space = BoxSpace(torch.tensor([[0.0, 1.0], [-1.0, 2.0]]))
    assert torch.equal(
        project_designs(torch.tensor([[-2.0, 4.0]]), space), torch.tensor([[0.0, 2.0]])
    )
    with pytest.raises(ValueError, match="actions must"):
        projected_gradient_step(
            torch.tensor([[0.5, 0.5]]),
            torch.ones(1, 2),
            torch.tensor([[2.0, 0.0]]),
            torch.ones(2),
            space,
        )


def test_box_projection_rounds_outward_float32_endpoints_inward():
    space = BoxSpace(torch.tensor([[-32.768, 32.768], [0., 1.]], dtype=torch.float64))
    projected = project_designs(torch.tensor([[-100., -1.], [100., 2.]]), space)
    assert (projected.double() >= space.bounds[:, 0]).all()
    assert (projected.double() <= space.bounds[:, 1]).all()
    assert torch.equal(projected[:, 1], torch.tensor([0., 1.]))
    assert projected.dtype == torch.float32


@pytest.mark.parametrize(
    "simplex,context", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_replay_reconstructs_logged_states_rewards_and_exact_fidelity(
    simplex, context, dtype
):
    p = problem(simplex, context, dtype)
    data = MentoringData.from_problem(p, 1e-6)
    model = Linear(data.features.shape[1], dtype)
    replay = build(p, model=model)
    left, right = replay.start_indices, replay.end_indices
    gradients = design_gradients(
        model, data, p.train_designs[left], p.train_context[left]
    )
    reconstructed = projected_gradient_step(
        p.train_designs[left],
        gradients,
        replay.actions,
        replay.step_scale,
        p.design_space,
    )
    assert torch.allclose(reconstructed, p.train_designs[right], atol=1e-6)
    assert torch.equal(p.train_context[left], p.train_context[right])
    assert torch.equal(replay.rewards, p.train_utility[right] - p.train_utility[left])
    assert (replay.actions.abs() < 1).all()
    assert not replay.actions.requires_grad and not replay.rewards.requires_grad
    assert replay.reconstruction_errors.max() < 1e-6
    assert (replay.remaining_steps[replay.terminals] == 1).all()
    assert (replay.remaining_steps[~replay.terminals] > 1).all()
    if context:
        assert replay.diagnostics["eligible_pool_sizes"] == [5, 5]
        assert set(left.tolist()) <= set(range(5, 10)) | set(range(15, 20))


def test_gradients_are_raw_design_derivatives_and_do_not_mutate_state():
    p = problem()
    data = MentoringData.from_problem(p, 1e-6)
    model = Linear(4, torch.float64)
    before = model.weight.detach().clone()
    snapshot = p.train_designs.clone()
    gradient = design_gradients(model, data, p.train_designs, p.train_context)
    assert torch.allclose(
        gradient, (model.weight[:2] / data.feature_std[:2]).expand_as(gradient)
    )
    assert torch.equal(model.weight, before) and model.weight.grad is None
    assert torch.equal(snapshot, p.train_designs) and not p.train_designs.requires_grad


def test_build_is_reproducible_and_rng_isolated():
    p = problem()
    rng = torch.random.get_rng_state().clone()
    first, second = build(p), build(p)
    for name in (
        "start_indices",
        "end_indices",
        "actions",
        "rewards",
        "terminals",
        "remaining_steps",
        "step_scale",
    ):
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert first.diagnostics == second.diagnostics
    assert torch.equal(rng, torch.random.get_rng_state())


def test_default_top_quantile_does_not_expand_small_pools_or_merge_steps():
    p = problem()
    p.train_context[:, 1] = torch.arange(20, dtype=torch.float64)
    with pytest.raises(ValueError, match="no same-fidelity"):
        build(p)
    p = problem()
    with pytest.raises(ValueError, match="no pool expansion"):
        build(p, config=PGSTransitionConfig(top_fraction=0.01))


def test_top_quantile_ties_are_included_and_horizon_is_bounded():
    p = problem()
    p.train_utility.fill_(0)
    replay = build(
        p, config=PGSTransitionConfig(trajectories_per_group=2, max_horizon=50)
    )
    assert replay.diagnostics["eligible_pool_sizes"] == [10, 10]
    assert replay.terminals.sum() == 4
    assert replay.remaining_steps.max() == 9
    assert replay.diagnostics["generated_transitions"] == 36
    assert (replay.rewards == 0).all()


def test_near_zero_gradient_moves_and_excessive_gains_are_rejected():
    p = problem()
    model = Linear(4, torch.float64)
    with torch.no_grad():
        model.weight[0] = 0
    with pytest.raises(ValueError, match="no reconstructible"):
        build(p, model=model)
    with torch.no_grad():
        model.weight[:2].fill_(1e-10)
    with pytest.raises(ValueError, match="no reconstructible"):
        build(
            p,
            model=model,
            config=PGSTransitionConfig(
                top_fraction=0.5, gradient_floor=1e-12, max_step_scale=1
            ),
        )


def test_zero_gradient_is_allowed_only_when_that_coordinate_does_not_move():
    p = problem(simplex=False)
    p.train_designs[:, 0] = 0.5
    model = Linear(4, torch.float64)
    with torch.no_grad():
        model.weight[0] = 0
    replay = build(p, model=model)
    assert (replay.actions[:, 0] == 0).all()
    assert replay.diagnostics["rejected_gradient"] == 0


def test_removed_edges_split_episodes_without_bridging_and_keep_correct_horizon():
    terminal, remaining = fragment_terminals(
        torch.tensor([0, 2, 3, 4]),
        torch.tensor([False, False, True, False, False, True]),
    )
    assert terminal.tolist() == [True, True, False, True]
    assert remaining.tolist() == [1, 1, 2, 1]


def test_partial_gradient_rejections_preserve_contiguous_replay_fragments():
    class PartialGradient(nn.Module):
        def forward(self, features):
            return torch.relu(features[:, 0]) + features[:, 1]

    p = problem()
    replay = build(
        p,
        model=PartialGradient(),
        config=PGSTransitionConfig(
            top_fraction=1,
            trajectories_per_group=3,
        ),
    )
    counts = replay.diagnostics
    assert counts["rejected_gradient"] > 0
    assert counts["generated_transitions"] == sum(
        counts[key]
        for key in (
            "retained_transitions",
            "rejected_gradient",
            "rejected_gain",
            "rejected_reconstruction",
        )
    )
    continuing = ~replay.terminals[:-1]
    assert torch.equal(
        replay.end_indices[:-1][continuing], replay.start_indices[1:][continuing]
    )
    assert torch.equal(
        replay.remaining_steps[1:][continuing],
        replay.remaining_steps[:-1][continuing] - 1,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_fraction": 0},
        {"top_fraction": 1.1},
        {"max_horizon": True},
        {"trajectories_per_group": 0},
        {"gradient_floor": float("nan")},
        {"action_margin": 1.0},
        {"max_step_scale": 0.1},
        {"reconstruction_tolerance": -1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        PGSTransitionConfig(**kwargs)


def test_pgs_method_is_registered_after_policy_integration():
    assert "pgs" in method_names()


def test_wrong_reconstruction_is_rejected_not_assigned_a_reward(monkeypatch):
    import llm_design_bench.optimizers.pgs_transitions as module

    monkeypatch.setattr(
        module, "projected_gradient_step", lambda designs, *args: designs
    )
    with pytest.raises(ValueError, match="reconstruction tolerance"):
        build(problem())


def test_default_configuration_builds_only_visible_top_quantile_edges():
    p = problem()
    data = MentoringData.from_problem(p, 1e-6)
    replay = build_logged_transitions(
        p, Linear(4, torch.float64), data, torch.Generator().manual_seed(38)
    )
    assert replay.diagnostics["eligible_pool_sizes"] == [2, 2]
    assert replay.diagnostics["generated_transitions"] == 64
    assert replay.terminals.all() and (replay.remaining_steps == 1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_reconstruction():
    replay = build(problem().to("cuda", torch.float32))
    assert replay.actions.is_cuda and replay.reconstruction_errors.max() < 1e-5
