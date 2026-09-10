import json
import math

import pytest
import torch

from llm_design_bench.optimizers import get_method_metadata, make_method
from llm_design_bench.optimizers.adaptive_generative import cbas_log_weights
from llm_design_bench.optimizers.diffusion_methods import probability_flow_log_density, reverse_kl_proxy_loss
from llm_design_bench.optimizers.gabo import LatentGP, expected_improvement
from llm_design_bench.optimizers.root import bridge_posterior, bridge_sample
from llm_design_bench.optimizers.spade import support_lcb, support_statistics
from llm_design_bench.optimizers.torch_components import Diffusion, classifier_free_condition
from llm_design_bench.optimizers.trajectory_methods import RegretTransformer, construct_trajectories, regret_to_go
from llm_design_bench.problem import OfflineProblem, ProblemMetadata, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace


TINY_CONFIG = {
    "cbas": {"population_size": 16, "adaptation_epochs": 1, "ensemble_size": 2},
    "mins": {"target_grid_size": 2},
    "ddom": {"diffusion_steps": 8},
    "rgd": {"diffusion_steps": 8, "refinement_rounds": 1, "refinement_candidates": 2,
            "likelihood_samples": 2, "likelihood_steps": 3},
    "demo": {"diffusion_steps": 8, "editing_epochs": 1, "pseudo_samples": 8},
    "bonet": {"trajectory_length": 3, "trajectory_count": 8},
    "gtg": {"diffusion_steps": 8, "trajectory_length": 3, "trajectory_count": 8},
    "gabo": {"initial_points": 4, "acquisition_steps": 2, "acquisition_restarts": 2, "critic_steps": 2},
    "root": {},
    "spade": {"diffusion_steps": 8, "sampling_steps": 3, "mc_samples": 2,
              "calibration_samples": 2, "calibration_steps": 3},
}


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def problem_for(space_name, dtype=torch.float32, context_dim=2):
    generator = torch.Generator().manual_seed(17)
    space = SimplexSpace(3) if space_name == "simplex" else BoxSpace(torch.tensor([[-1., 1.]] * 3))
    x = space.sample(16, generator=generator, dtype=dtype)
    c = torch.zeros((16, context_dim), dtype=dtype)
    if context_dim:
        c[:8] = 1
        c[8:] = 2
    y = -(x - 0.3).square().sum(-1)
    return OfflineProblem(train_designs=x, train_context=c, train_utility=y,
                          target_context=c[-1], design_space=space,
                          metadata=ProblemMetadata(task_name=f"additional-{space_name}"))


@pytest.mark.parametrize("method_id", list(TINY_CONFIG))
@pytest.mark.parametrize("space_name", ["box", "simplex"])
@pytest.mark.parametrize("dtype,context_dim", [(torch.float32, 2), (torch.float64, 0)])
def test_method_contract_and_seed_replay(method_id, space_name, dtype, context_dim):
    problem = problem_for(space_name, dtype, context_dim)
    originals = problem.train_designs.clone(), problem.train_utility.clone()
    config = dict(epochs=2, steps=2, batch_size=8, hidden_size=16, **TINY_CONFIG[method_id])
    context = RunContext(method_seed=38, split_seed=7, candidate_budget=4, dtype=dtype)
    torch.manual_seed(101)
    before_rng = torch.get_rng_state().clone()
    first = make_method(method_id, **config).run(problem, context)
    assert torch.equal(torch.get_rng_state(), before_rng)
    torch.rand(7)  # ambient randomness must not change replay
    second = make_method(method_id, **config).run(problem, context)
    assert torch.equal(first.candidates, second.candidates)
    assert first.candidates.shape == (4, 3)
    assert first.candidates.dtype == dtype
    assert torch.isfinite(first.candidates).all()
    problem.design_space.validate(first.candidates)
    assert torch.equal(problem.train_designs, originals[0])
    assert torch.equal(problem.train_utility, originals[1])
    assert first.training_summary["train_samples"] < 16
    assert first.training_summary["resolved_method_config"]["epochs"] == 2
    json.dumps(first.training_summary, allow_nan=False)
    json.dumps(first.diagnostics, allow_nan=False)
    metadata = get_method_metadata(method_id)
    assert len(metadata.source_commit) == 40
    assert metadata.implementation_kind.value == "lightweight_adaptation"


def test_cbas_density_ratio_and_normal_survival():
    # Equal densities and threshold at mean give weight 1/2; double prior gives 1.
    logweights = cbas_log_weights(torch.tensor([0., math.log(2)]), torch.zeros(2),
                                  torch.zeros(2, 1), torch.ones(2, 1), 0.)
    assert torch.allclose(logweights.exp(), torch.tensor([0.5, 1.]))
    extreme = cbas_log_weights(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.ones(1), 100.)
    assert torch.isfinite(extreme).all()


def test_diffusion_forward_and_perfect_reverse():
    model = Diffusion(2, 1, 8, 8)
    clean = torch.tensor([[0.2, -0.4]])
    epsilon = torch.tensor([[0.5, 0.25]])
    t = torch.tensor([7])
    noisy = model.q_sample(clean, t, epsilon)
    expected = model.alpha_bar[7].sqrt() * clean + (1 - model.alpha_bar[7]).sqrt() * epsilon
    assert torch.equal(noisy, expected)

    class PerfectDiffusion(Diffusion):
        def forward(self, x, t, condition):
            a = self.alpha_bar[t, None]
            return (x - a.sqrt() * clean) / (1 - a).sqrt()

    perfect = PerfectDiffusion(2, 1, 8, 8)
    result = perfect.sample(torch.zeros(1, 1), torch.Generator().manual_seed(0), steps=8, initial=noisy)
    assert torch.allclose(result, clean, atol=1e-5)
    y, c = torch.zeros(2, 1), torch.ones(2, 1)
    assert not torch.equal(classifier_free_condition(y, c), classifier_free_condition(y, c, unconditional=True))


def test_trajectory_fidelity_order_and_causality():
    x = torch.arange(24.).reshape(8, 3)
    y = torch.tensor([1., 2., 3., 4., 10., 11., 12., 13.])[:, None]
    c = torch.tensor([0.] * 4 + [1.] * 4)[:, None]
    for neighbors in (None, 2):
        _, ty, tc, ids = construct_trajectories(x, y, c, count=8, length=4,
                                               generator=torch.Generator().manual_seed(1), neighbors=neighbors)
        assert torch.all(ty[:, 1:] >= ty[:, :-1])
        assert torch.all(tc == tc[:, :1])
        assert torch.equal(ty, y[ids])
    assert torch.equal(regret_to_go(torch.tensor([[[1.], [2.], [3.]]]), 4.), torch.tensor([[[6.], [3.], [1.]]]))
    model = RegretTransformer(3, 1, 8, 4).eval()
    tokens, regrets, contexts = torch.randn(2, 4, 3), torch.ones(2, 4, 1), torch.zeros(2, 4, 1)
    first = model(tokens, regrets, contexts)[0]
    tokens[:, 2:] = 1000
    second = model(tokens, regrets, contexts)[0]
    assert torch.allclose(first[:, :2], second[:, :2])


def test_bridge_endpoints_and_posterior_variance():
    low, high = torch.ones(10000, 1), torch.zeros(10000, 1)
    noise = torch.randn(10000, 1, generator=torch.Generator().manual_seed(3))
    assert torch.equal(bridge_sample(high, low, torch.tensor(0.), torch.tensor(0.), noise)[0], high)
    assert torch.equal(bridge_sample(high, low, torch.tensor(1.), torch.tensor(0.), noise)[0], low)
    m, ms, v, vs = map(torch.tensor, [0.8, 0.4, 0.32, 0.48])
    x = (1 - m) * high + m * low
    output = bridge_posterior(x, high, low, m, ms, v, vs, noise)
    variance = (v - vs * (1 - m) ** 2 / (1 - ms) ** 2) * vs / v
    assert abs(float(output.mean()) - float(ms)) < 0.025
    assert abs(float(output.var()) - float(variance)) < 0.025


def test_probability_flow_stationary_gaussian_and_reverse_kl():
    class StationaryGaussian:
        timesteps = 100
        def __call__(self, x, t, condition):
            time = (t + 0.5) / self.timesteps
            ab = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square() / math.cos(0.008 / 1.008 * math.pi / 2) ** 2
            return (1 - ab).clamp_min(1e-6).sqrt()[:, None] * x
    x = torch.tensor([[0.2, 0.3], [-0.5, 1.]], dtype=torch.float64)
    logp = probability_flow_log_density(StationaryGaussian(), x, torch.zeros(2, 1), torch.Generator().manual_seed(1), steps=12)
    assert torch.allclose(logp, -0.5 * (x.square() + math.log(2 * math.pi)).sum(-1), atol=1e-8)
    logq = torch.tensor([-1., -2.], requires_grad=True)
    target = torch.tensor([-2., -1.])
    reverse_kl_proxy_loss(logq, target).backward()
    assert torch.allclose(logq.grad, torch.tensor([1., 0.]))


def test_gp_and_support_conservatism():
    x, y = torch.tensor([[-1.], [0.], [1.]]), torch.tensor([0., 1., 0.])
    gp = LatentGP(x, y)
    mean, std = gp.predict(x)
    assert mean[1] > mean[0]
    assert torch.all(expected_improvement(mean, std, y.max()) >= 0)
    _, radius = support_statistics(x, x, y[:, None], 1, exclude_ids=torch.arange(3))
    assert torch.all(radius == 1)
    draws = torch.tensor([[1., 1.], [1.1, 1.1]])
    scores = support_lcb(draws, torch.tensor([1., math.e]), beta=1, mean_penalty=.1, sigma_base=.1, sigma_distance=.1)
    assert scores[1] < scores[0]


def test_float32_decoding_respects_original_float64_box_limits():
    from llm_design_bench.tasks.synthetic_functions import SyntheticFunctionTask
    task = SyntheticFunctionTask("ackley", logged_samples=16, seed=3)
    problem = OfflineProblem.from_task(task)
    space = problem.design_space
    assert space.bounds.dtype == torch.float64
    extremes = torch.tensor([[-100., -100.], [100., 100.]])
    for designs in (space.decode_from_model(extremes), space.from_unconstrained(extremes)):
        task.validate(task.at_target_fidelity(designs.numpy()))
