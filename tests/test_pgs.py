import math

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
from llm_design_bench.optimizers.mentoring_utils import MentoringData
from llm_design_bench.optimizers.pgs import policy_states
from llm_design_bench.optimizers.pgs_sac import (
    ConservativeSAC,
    GaussianActor,
    SACBatch,
    bellman_target,
    conservative_penalty,
    polyak_update,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace
from llm_design_bench.types import CandidateBatch

CONFIG = {
    "hidden_size": 16,
    "surrogate_epochs": 3,
    "batch_size": 4,
    "rl_steps": 3,
    "cql_samples": 2,
    "trajectories_per_group": 2,
    "max_horizon": 3,
    "top_fraction": 0.5,
    "solver_steps": 5,
}


def problem(simplex=True, context=True):
    x = torch.linspace(0.1, 0.9, 6).repeat(2)
    return OfflineProblem(
        torch.stack([x, 1 - x], 1),
        torch.tensor([[20.0, 10.0]] * 6 + [[60.0, 10.0]] * 6)
        if context
        else torch.empty(12, 0),
        x - 2.0,
        torch.tensor([1000.0, 19500.0]) if context else torch.empty(0),
        SimplexSpace(2)
        if simplex
        else BoxSpace(torch.tensor([[0.0, 1.0], [0.0, 1.0]])),
        ProblemMetadata(task_name="pgs-method-test"),
    )


def learner(**kwargs):
    settings = {
        "hidden_size": 8,
        "learning_rate": 0.001,
        "discount": 0.99,
        "target_rate": 0.1,
        "cql_weight": 5.0,
        "cql_samples": 2,
        "cql_temperature": 1.0,
        "initial_alpha": 1.0,
        "automatic_entropy": True,
        "backup_entropy": False,
    }
    settings.update(kwargs)
    context = RunContext(method_seed=38, candidate_budget=2, dtype=torch.float64)
    return ConservativeSAC(3, 2, context, context.make_generator(), **settings)


def batch():
    return SACBatch(
        torch.tensor([[0.1, 0.2, 1.0], [0.3, 0.4, 0.5]], dtype=torch.float64),
        torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float64),
        torch.tensor([0.2, -0.1], dtype=torch.float64),
        torch.tensor([[0.3, 0.4, 0.5], [0.1, 0.2, 0.0]], dtype=torch.float64),
        torch.tensor([False, True]),
    )


def test_squashed_gaussian_density_matches_distribution_and_preserves_rng():
    actor = GaussianActor(3, 2, 8).double()
    states = batch().states
    rng = torch.random.get_rng_state().clone()
    actions, density = actor.sample(states, torch.Generator().manual_seed(3))
    mean, log_std = actor.parameters_at(states)
    expected = (
        torch.distributions.Normal(mean, log_std.exp()).log_prob(torch.atanh(actions))
        - torch.log1p(-actions.square())
    ).sum(-1)
    assert torch.allclose(density, expected, atol=1e-10)
    assert (actions.abs() < 1).all()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(
        actions, actor.sample(states, torch.Generator().manual_seed(3))[0]
    )
    assert torch.equal(actor.mode(states), mean.tanh())
    (-density.mean()).backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in actor.parameters()
    )


def test_squashed_log_density_remains_finite_at_saturated_actions():
    actor = GaussianActor(3, 2, 8).double()
    with torch.no_grad():
        actor.network[-1].weight.zero_()
        actor.network[-1].bias[:2].fill_(1000)
    actions, density = actor.sample(batch().states, torch.Generator().manual_seed(4))
    assert actions.isfinite().all() and density.isfinite().all()


def test_bellman_terminal_mask_entropy_switch_and_stop_gradient():
    rewards = torch.tensor([1.0, 2.0], requires_grad=True)
    q = torch.tensor([4.0, float("nan")], requires_grad=True)
    logp = torch.tensor([-2.0, float("nan")])
    terminals = torch.tensor([False, True])
    plain = bellman_target(
        rewards, terminals, q, logp, discount=0.5, alpha=0.2, backup_entropy=False
    )
    entropy = bellman_target(
        rewards, terminals, q, logp, discount=0.5, alpha=0.2, backup_entropy=True
    )
    assert torch.allclose(plain, torch.tensor([3.0, 2.0]))
    assert torch.allclose(entropy, torch.tensor([3.2, 2.0]))
    assert not plain.requires_grad


def test_cql_importance_correction_and_gradient_direction():
    q = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    density = torch.tensor(
        [[-2 * math.log(2), -0.2, -0.4]], dtype=torch.float64, requires_grad=True
    )
    logged = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    value = conservative_penalty(q, density, logged, 0.7)
    expected = (
        0.7 * torch.logsumexp((q - density.detach()) / 0.7, 1).mean() - logged.mean()
    )
    assert torch.equal(value, expected)
    value.backward()
    assert (q.grad > 0).all() and logged.grad.item() == -1.0
    assert density.grad is None


def test_polyak_update_and_no_target_gradients():
    source, target = nn.Linear(2, 1).double(), nn.Linear(2, 1).double()
    before = [p.detach().clone() for p in target.parameters()]
    polyak_update(source, target, 0.2)
    for old, new, online in zip(before, target.parameters(), source.parameters()):
        assert torch.allclose(new, 0.8 * old + 0.2 * online)
        assert new.grad is None


def test_critic_loss_does_not_train_actor_alpha_or_targets():
    agent = learner()
    loss, metrics = agent.critic_loss(batch())
    assert all(math.isfinite(v) for v in metrics.values())
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in agent.critics.parameters()
    )
    assert all(p.grad is None for p in agent.actor.parameters())
    assert all(
        p.grad is None and not p.requires_grad for p in agent.targets.parameters()
    )
    assert agent.log_alpha.grad is None


def test_all_cql_proposals_are_evaluated_at_current_states():
    agent = learner()
    replay = batch()
    seen = []
    hook = agent.critics[0].register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    try:
        agent.critic_loss(replay)
    finally:
        hook.remove()
    assert len(seen) == 2
    assert torch.equal(seen[0], replay.states)
    assert torch.equal(
        seen[1], replay.states.repeat_interleave(3 * agent.cql_samples, 0)
    )


def test_updates_isolate_actor_critic_entropy_and_update_targets(monkeypatch):
    import llm_design_bench.optimizers.pgs_sac as module

    agent = learner()
    target_before = [p.detach().clone() for p in agent.targets.parameters()]
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    original = module.checked_step
    phases = []

    def checked(loss, optimizer, parameters):
        if optimizer is agent.actor_optimizer:
            phases.append("actor")
            before = [p.detach().clone() for p in agent.critics.parameters()]
            assert all(
                not p.requires_grad and p.grad is None
                for p in agent.critics.parameters()
            )
            value = original(loss, optimizer, parameters)
            assert all(
                torch.equal(old, new)
                for old, new in zip(before, agent.critics.parameters())
            )
            assert agent.log_alpha.grad is None
            return value
        phases.append("critic" if optimizer is agent.critic_optimizer else "alpha")
        return original(loss, optimizer, parameters)

    monkeypatch.setattr(module, "checked_step", checked)
    metrics = agent.update(batch())
    assert phases == ["critic", "actor", "alpha"]
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["alpha"] != 1.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(actor_before, agent.actor.parameters())
    )
    for old, target, current in zip(
        target_before, agent.targets.parameters(), agent.critics.parameters()
    ):
        assert torch.allclose(target, 0.9 * old + 0.1 * current)
        assert target.grad is None


def test_fixed_entropy_ablation_preserves_alpha_and_learner_rng():
    rng = torch.random.get_rng_state().clone()
    first, second = learner(automatic_entropy=False), learner(automatic_entropy=False)
    assert first.update(batch()) == second.update(batch())
    assert first.alpha == 1.0 and first.log_alpha.grad is None
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("simplex,context", [(True, True), (False, False)])
def test_method_reproducibility_boundaries_and_frozen_surrogate(
    dtype, simplex, context, monkeypatch
):
    import llm_design_bench.optimizers.pgs as module

    original = module.design_gradients
    snapshots = []

    def tracked(proxy, data, designs, contexts):
        assert all(not p.requires_grad and p.grad is None for p in proxy.parameters())
        snapshots.append([p.detach().clone() for p in proxy.parameters()])
        return original(proxy, data, designs, contexts)

    monkeypatch.setattr(module, "design_gradients", tracked)
    p = problem(simplex, context)
    before = [p.train_designs.clone(), p.train_context.clone(), p.train_utility.clone()]
    method = make_method("pgs", **CONFIG)
    ctx = RunContext(method_seed=38, candidate_budget=8, dtype=dtype)
    rng = torch.random.get_rng_state().clone()
    first, second = method.run(p, ctx), method.run(p, ctx)
    assert torch.equal(first.candidates, second.candidates)
    assert first.training_summary == second.training_summary
    assert torch.equal(rng, torch.random.get_rng_state())
    assert first.candidates.shape == (8, 2) and first.candidates.dtype == dtype
    assert (
        first.diagnostics["solver_steps_used"]
        <= first.diagnostics["maximum_retained_horizon"]
    )
    assert first.training_summary["rl_updates"] == CONFIG["rl_steps"]
    assert first.training_summary["reward_divisor"] == pytest.approx(
        float(p.train_utility.to(dtype).std(unbiased=False))
    )
    for old, new in zip(before, (p.train_designs, p.train_context, p.train_utility)):
        assert torch.equal(old, new)
    assert all(
        all(torch.equal(a, b) for a, b in zip(snapshots[0], snapshot))
        for snapshot in snapshots
    )


def test_policy_state_uses_same_context_and_horizon_units():
    p = problem()
    data = MentoringData.from_problem(p, 1e-6)
    state = policy_states(
        data, p.train_designs[:2], p.train_context[:2], torch.tensor([2, 0]), 4
    )
    assert torch.equal(state[:, :-1], data.features[:2])
    assert torch.equal(state[:, -1], torch.tensor([0.5, 0.0]))


def test_rollout_uses_saved_scale_and_fixed_target_only(monkeypatch):
    import llm_design_bench.optimizers.pgs as module

    original_step, original_state = module.projected_gradient_step, module.policy_states
    scales, contexts, horizons = [], [], []

    def tracked_step(x, gradient, action, scale, space):
        scales.append(scale.clone())
        return original_step(x, gradient, action, scale, space)

    def tracked_state(data, x, context, remaining, horizon):
        contexts.append(context.clone())
        horizons.append(remaining.clone())
        return original_state(data, x, context, remaining, horizon)

    monkeypatch.setattr(module, "projected_gradient_step", tracked_step)
    monkeypatch.setattr(module, "policy_states", tracked_state)
    p = problem()
    result = make_method("pgs", **CONFIG).run(
        p, RunContext(method_seed=38, candidate_budget=4)
    )
    count = result.diagnostics["solver_steps_used"]
    assert len(scales) == count
    for i, (scale, context, remaining) in enumerate(
        zip(scales, contexts[2:], horizons[2:])
    ):
        assert scale.tolist() == result.diagnostics["step_scale"]
        assert torch.equal(context, p.target_context.expand(4, -1))
        assert (remaining == count - i).all()


def test_oracle_runs_only_after_return_and_not_for_unsupported_data(
    tmp_path, monkeypatch
):
    cls = type(make_method("pgs"))
    original, completed = cls.optimize, []

    def tracked(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        completed.append(True)
        return result

    monkeypatch.setattr(cls, "optimize", tracked)

    class Evaluator:
        calls = 0

        def at_target_fidelity(self, designs):
            return CandidateBatch.at_fidelity(designs, 1000.0, 19500.0)

        def predict(self, candidates):
            assert completed
            self.calls += 1
            return -np.square(candidates.mixtures).sum(1)

    evaluator = Evaluator()

    def run(p):
        return run_method_seed_benchmark(
            evaluator,
            p,
            [MethodSpec("pgs", CONFIG)],
            reference_utility=np.array([-2.0, 0.0]),
            config=SeedBenchmarkConfig(
                seeds=(38,), candidate_budget=3, results_dir=tmp_path
            ),
            write_results=False,
        )

    result = run(problem())
    assert result.per_seed.status.tolist() == ["success"]
    p = problem()
    p.train_context[:, 0] = torch.arange(12.0)
    failed = run(p)
    assert failed.per_seed.status.tolist() == ["failed"]
    assert "no same-fidelity" in failed.per_seed.error_message.iloc[0]
    assert evaluator.calls == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"surrogate_epochs": True},
        {"batch_size": 0},
        {"rl_steps": 0},
        {"cql_samples": -1},
        {"cql_weight": 0},
        {"cql_temperature": float("nan")},
        {"target_rate": 0},
        {"discount": 1.1},
        {"initial_alpha": 0},
        {"automatic_entropy": 1},
        {"backup_entropy": "yes"},
        {"solver_steps": 0},
        {"top_fraction": 0},
        {"max_horizon": -1},
        {"rl_learning_rate": float("inf")},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        make_method("pgs", **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_method():
    result = make_method("pgs", **CONFIG).run(
        problem(), RunContext(method_seed=38, candidate_budget=4, device="cuda")
    )
    assert result.candidates.is_cuda and result.candidates.isfinite().all()
