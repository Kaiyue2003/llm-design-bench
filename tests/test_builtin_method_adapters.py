import numpy as np
import pytest
import torch

from llm_design_bench.evaluation.seed_runner import (
    MethodSpec,
    SeedBenchmarkConfig,
    run_method_seed_benchmark,
)
from llm_design_bench.optimizers import make_method, method_names
from llm_design_bench.problem import (
    OfflineProblem,
    ProblemMetadata,
    RunContext,
)
from llm_design_bench.spaces import BoxSpace, SimplexSpace
from llm_design_bench.types import CandidateBatch


def _simplex_problem() -> OfflineProblem:
    return OfflineProblem(
        train_designs=torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.7, 0.2, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.2, 0.7],
            ]
        ),
        train_context=torch.tensor(
            [
                [20.0, 1_000.0],
                [60.0, 2_000.0],
                [150.0, 5_000.0],
                [1000.0, 19_500.0],
            ]
        ),
        train_utility=torch.tensor([-0.1, -0.05, -1.0, -2.0]),
        target_context=torch.tensor([1000.0, 19_500.0]),
        design_space=SimplexSpace(3),
        metadata=ProblemMetadata(task_name="adapter-simplex"),
    )


def _box_problem() -> OfflineProblem:
    return OfflineProblem(
        train_designs=torch.tensor(
            [
                [-0.8, 1.2],
                [-0.1, 1.8],
                [0.4, 2.4],
                [0.9, 2.8],
            ]
        ),
        train_context=torch.empty((4, 0)),
        train_utility=torch.tensor([-1.0, -0.2, -0.4, -1.5]),
        target_context=torch.empty(0),
        design_space=BoxSpace(torch.tensor([[-1.0, 1.0], [1.0, 3.0]])),
        metadata=ProblemMetadata(task_name="adapter-box"),
    )


_METHOD_CONFIGS = (
    ("best_logged", {}),
    ("random_search", {}),
    ("sobol", {}),
    (
        "offline_mlp",
        {
            "hidden_size": 16,
            "epochs": 2,
            "batch_size": 2,
            "particle_steps": 2,
        },
    ),
)


@pytest.mark.parametrize("method_id,kwargs", _METHOD_CONFIGS)
@pytest.mark.parametrize("problem_factory", [_simplex_problem, _box_problem])
def test_builtin_method_returns_valid_candidate_batch(
    method_id,
    kwargs,
    problem_factory,
) -> None:
    problem = problem_factory()
    result = make_method(method_id, **kwargs).run(
        problem,
        RunContext(method_seed=38, candidate_budget=6),
    )

    assert result.candidates.shape == (6, problem.design_dim)
    problem.design_space.validate(result.candidates)
    assert result.candidates.dtype == torch.float32


def test_builtin_methods_are_registered() -> None:
    assert {"best_logged", "random_search", "sobol", "offline_mlp"}.issubset(
        method_names()
    )


def test_best_logged_preserves_utility_order_and_removes_duplicates() -> None:
    result = make_method("best_logged").run(
        _simplex_problem(),
        RunContext(method_seed=38, candidate_budget=5),
    )

    expected = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
        ]
    )
    assert torch.equal(result.candidates, expected)
    assert result.training_summary["unique_logged_designs_used"] == 3


@pytest.mark.parametrize("method_id", ["random_search", "sobol", "offline_mlp"])
def test_stochastic_methods_are_reproducible(method_id) -> None:
    kwargs = dict(_METHOD_CONFIGS)[method_id]
    context = RunContext(method_seed=41, candidate_budget=7)

    first = make_method(method_id, **kwargs).run(_simplex_problem(), context)
    second = make_method(method_id, **kwargs).run(_simplex_problem(), context)

    assert torch.equal(first.candidates, second.candidates)


@pytest.mark.parametrize("method_id", ["random_search", "sobol"])
def test_sampling_seed_changes_candidates(method_id) -> None:
    first = make_method(method_id).run(
        _simplex_problem(),
        RunContext(method_seed=38, candidate_budget=7),
    )
    second = make_method(method_id).run(
        _simplex_problem(),
        RunContext(method_seed=39, candidate_budget=7),
    )

    assert not torch.equal(first.candidates, second.candidates)


def test_offline_mlp_does_not_mutate_global_torch_rng() -> None:
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)

    make_method(
        "offline_mlp",
        hidden_size=8,
        epochs=1,
        batch_size=2,
        particle_steps=1,
    ).run(
        _simplex_problem(),
        RunContext(method_seed=38, candidate_budget=5),
    )

    assert torch.equal(torch.rand(4), expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 0},
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"particle_steps": 0},
        {"particle_learning_rate": float("nan")},
    ],
)
def test_offline_mlp_rejects_invalid_hyperparameters(kwargs) -> None:
    with pytest.raises(ValueError):
        make_method("offline_mlp", **kwargs)


class _CountingEvaluator:
    target_model_scale = 1000.0
    target_training_steps = 19_500.0

    def __init__(self) -> None:
        self.predict_calls = 0

    def at_target_fidelity(self, designs: np.ndarray) -> CandidateBatch:
        return CandidateBatch.at_fidelity(
            designs,
            self.target_model_scale,
            self.target_training_steps,
        )

    def predict(self, batch: CandidateBatch) -> np.ndarray:
        self.predict_calls += 1
        target = np.array([0.7, 0.2, 0.1])
        return -np.square(batch.mixtures - target).sum(axis=1)


def test_seed_runner_evaluates_builtin_candidates_only_after_return(tmp_path) -> None:
    evaluator = _CountingEvaluator()
    methods = [
        MethodSpec("best_logged"),
        MethodSpec("random_search"),
        MethodSpec("sobol"),
        MethodSpec(
            "offline_mlp",
            {
                "hidden_size": 8,
                "epochs": 1,
                "batch_size": 2,
                "particle_steps": 1,
            },
        ),
    ]
    result = run_method_seed_benchmark(
        evaluator,
        _simplex_problem(),
        methods,
        reference_utility=np.array([-2.0, 0.0]),
        config=SeedBenchmarkConfig(
            seeds=(38, 39),
            candidate_budget=5,
            results_dir=tmp_path,
        ),
    )

    assert evaluator.predict_calls == len(methods) * 2
    assert set(result.per_seed["status"]) == {"success"}
    assert set(result.per_seed["method_id"]) == {
        "best_logged",
        "random_search",
        "sobol",
        "offline_mlp",
    }
