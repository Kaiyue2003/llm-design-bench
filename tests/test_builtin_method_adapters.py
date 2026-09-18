"""Forward-method regressions; the separate SPADE implementation is out of scope."""

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
    (
        "pgs",
        {
            "hidden_size": 16,
            "surrogate_epochs": 2,
            "batch_size": 2,
            "rl_steps": 2,
            "cql_samples": 2,
            "trajectories_per_group": 2,
            "top_fraction": 1.0,
            "solver_steps": 2,
        },
    ),
    (
        "match_opt",
        {"embedding_dim": 2, "surrogate_epochs": 2, "batch_size": 2, "solver_steps": 2},
    ),
    (
        "ltr",
        {
            "hidden_size": 8,
            "surrogate_epochs": 2,
            "batch_size": 2,
            "list_length": 3,
            "lists_per_epoch": 3,
            "validation_lists": 2,
            "solver_steps": 2,
        },
    ),
    (
        "roma",
        {
            "hidden_size": 8,
            "surrogate_epochs": 2,
            "batch_size": 2,
            "weight_perturbation_steps": 2,
            "adaptation_steps": 2,
            "solver_steps": 2,
        },
    ),
    (
        "ict",
        {
            "hidden_size": 8,
            "surrogate_epochs": 2,
            "batch_size": 2,
            "adaptation_steps": 2,
            "solver_steps": 2,
            "neighbor_samples": 4,
            "remember_count": 2,
            "surrogate_learning_rate": 0.01,
        },
    ),
    (
        "tri_mentoring",
        {
            "hidden_size": 8,
            "surrogate_epochs": 2,
            "batch_size": 2,
            "solver_steps": 2,
            "neighbor_samples": 4,
        },
    ),
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
    (
        "standard_ga",
        {
            "hidden_size": 16,
            "surrogate_epochs": 2,
            "batch_size": 2,
            "solver_steps": 2,
        },
    ),
    (
        "cma_es",
        {
            "generations": 2,
            "population_size": 4,
            "ensemble_size": 2,
            "hidden_size": 16,
            "surrogate_epochs": 2,
            "batch_size": 2,
        },
    ),
    (
        "reinforce",
        {
            "iterations": 2,
            "reinforce_batch_size": 8,
            "ensemble_size": 2,
            "hidden_size": 16,
            "surrogate_epochs": 2,
            "batch_size": 2,
        },
    ),
    (
        "bo_qei",
        {
            "gp_training_steps": 2,
            "acquisition_steps": 2,
            "mc_samples": 8,
        },
    ),
    (
        "ga_on_gp",
        {
            "gp_training_steps": 2,
            "solver_steps": 2,
        },
    ),
    (
        "mc_dropout",
        {
            "hidden_size": 16,
            "epochs": 2,
            "batch_size": 2,
            "mc_samples": 4,
            "particle_steps": 2,
        },
    ),
    (
        "coms",
        {
            "hidden_size": 16,
            "epochs": 2,
            "batch_size": 2,
            "adversarial_steps": 2,
            "particle_steps": 2,
        },
    ),
    (
        "bdi",
        {
            "steps": 2,
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
    if method_id in {"match_opt", "pgs"} and problem_factory is _simplex_problem:
        # Keep an eligible same-fidelity pair; other contexts remain distinct.
        problem.train_context[2].copy_(problem.train_context[3])
    result = make_method(method_id, **kwargs).run(
        problem,
        RunContext(method_seed=38, candidate_budget=6),
    )

    assert result.candidates.shape == (6, problem.design_dim)
    problem.design_space.validate(result.candidates)
    assert result.candidates.dtype == torch.float32


def test_builtin_methods_are_registered() -> None:
    assert {
        "pgs",
        "match_opt",
        "ltr",
        "roma",
        "ict",
        "tri_mentoring",
        "best_logged",
        "random_search",
        "sobol",
        "offline_mlp",
        "standard_ga",
        "cma_es",
        "reinforce",
        "bo_qei",
        "ga_on_gp",
        "mc_dropout",
        "coms",
        "bdi",
    }.issubset(method_names())


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


@pytest.mark.parametrize(
    "method_id",
    [
        "random_search",
        "sobol",
        "offline_mlp",
        "standard_ga",
        "cma_es",
        "reinforce",
        "bo_qei",
        "ga_on_gp",
        "mc_dropout",
        "coms",
        "bdi",
    ],
)
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


@pytest.mark.parametrize("method_id", ["bo_qei", "ga_on_gp", "mc_dropout"])
def test_gp_and_uncertainty_seed_changes_candidates(method_id) -> None:
    kwargs = dict(_METHOD_CONFIGS)[method_id]
    first = make_method(method_id, **kwargs).run(
        _simplex_problem(),
        RunContext(method_seed=38, candidate_budget=6),
    )
    second = make_method(method_id, **kwargs).run(
        _simplex_problem(),
        RunContext(method_seed=39, candidate_budget=6),
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


def test_coms_does_not_mutate_global_torch_rng() -> None:
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)

    make_method(
        "coms",
        hidden_size=8,
        epochs=1,
        batch_size=2,
        adversarial_steps=1,
        particle_steps=1,
    ).run(
        _simplex_problem(),
        RunContext(method_seed=38, candidate_budget=5),
    )

    assert torch.equal(torch.rand(4), expected)


@pytest.mark.parametrize("method_id", ["standard_ga", "cma_es", "reinforce"])
def test_probabilistic_baselines_do_not_mutate_global_torch_rng(method_id) -> None:
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)

    make_method(method_id, **dict(_METHOD_CONFIGS)[method_id]).run(
        _simplex_problem(),
        RunContext(method_seed=38, candidate_budget=5),
    )

    assert torch.equal(torch.rand(4), expected)


@pytest.mark.parametrize("method_id", ["bo_qei", "ga_on_gp", "mc_dropout"])
def test_gp_and_uncertainty_methods_do_not_mutate_global_torch_rng(
    method_id,
) -> None:
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)

    make_method(method_id, **dict(_METHOD_CONFIGS)[method_id]).run(
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


@pytest.mark.parametrize(
    "method_id,kwargs",
    [
        ("coms", {"alpha": 0.0}),
        ("coms", {"adversarial_steps": 0}),
        ("coms", {"overestimation_limit": float("nan")}),
        ("bdi", {"steps": 0}),
        ("bdi", {"lengthscale": 0.0}),
        ("bdi", {"ridge": 0.0}),
        ("bdi", {"forward_weight": -1.0}),
        ("bdi", {"forward_weight": 0.0, "distillation_weight": 0.0}),
    ],
)
def test_adaptations_reject_invalid_hyperparameters(method_id, kwargs) -> None:
    with pytest.raises(ValueError):
        make_method(method_id, **kwargs)


@pytest.mark.parametrize(
    "method_id,kwargs",
    [
        ("standard_ga", {"surrogate_epochs": 0}),
        ("standard_ga", {"solver_steps": 0}),
        ("standard_ga", {"solver_learning_rate": 0.0}),
        ("standard_ga", {"initial_min_std": 0.2, "initial_max_std": 0.1}),
        ("cma_es", {"generations": 0}),
        ("cma_es", {"population_size": 1}),
        ("cma_es", {"sigma": 0.0}),
        ("cma_es", {"initial_min_std": 0.2, "initial_max_std": 0.1}),
        ("reinforce", {"iterations": 0}),
        ("reinforce", {"reinforce_batch_size": 0}),
        ("reinforce", {"exploration_std": float("nan")}),
        ("reinforce", {"initial_min_std": 0.2, "initial_max_std": 0.1}),
    ],
)
def test_standard_baselines_reject_invalid_hyperparameters(
    method_id,
    kwargs,
) -> None:
    with pytest.raises(ValueError):
        make_method(method_id, **kwargs)


@pytest.mark.parametrize(
    "method_id,kwargs",
    [
        ("bo_qei", {"gp_training_steps": 0}),
        ("bo_qei", {"acquisition_steps": 0}),
        ("bo_qei", {"mc_samples": 0}),
        ("bo_qei", {"random_start_fraction": 1.0}),
        ("bo_qei", {"jitter": 0.0}),
        ("ga_on_gp", {"gp_training_steps": 0}),
        ("ga_on_gp", {"solver_steps": 0}),
        ("ga_on_gp", {"random_start_fraction": float("nan")}),
        ("mc_dropout", {"dropout_probability": 0.0}),
        ("mc_dropout", {"dropout_probability": 1.0}),
        ("mc_dropout", {"mc_samples": 0}),
        ("mc_dropout", {"uncertainty_weight": -1.0}),
        ("mc_dropout", {"uncertainty_weight": float("nan")}),
    ],
)
def test_gp_and_uncertainty_methods_reject_invalid_hyperparameters(
    method_id,
    kwargs,
) -> None:
    with pytest.raises(ValueError):
        make_method(method_id, **kwargs)


def test_adaptation_metadata_is_explicit() -> None:
    coms = make_method("coms").metadata
    bdi = make_method("bdi").metadata

    assert coms.display_name == "COMs adaptation"
    assert coms.implementation_kind.value == "lightweight_adaptation"
    assert "compact_pytorch_reimplementation" in coms.adaptations
    assert bdi.display_name == "BDI adaptation"
    assert bdi.implementation_kind.value == "lightweight_adaptation"
    assert "rbf_kernel_replaces_infinite_width_ntk" in bdi.adaptations

    standard_ga = make_method("standard_ga").metadata
    cma_es = make_method("cma_es").metadata
    reinforce = make_method("reinforce").metadata
    assert standard_ga.display_name == "Standard GA adaptation"
    assert cma_es.source_commit == reinforce.source_commit
    assert cma_es.display_name == "CMA-ES adaptation"
    assert reinforce.display_name == "REINFORCE adaptation"
    assert {
        standard_ga.implementation_kind.value,
        cma_es.implementation_kind.value,
        reinforce.implementation_kind.value,
    } == {"multi_fidelity_adaptation"}

    bo_qei = make_method("bo_qei").metadata
    ga_on_gp = make_method("ga_on_gp").metadata
    mc_dropout = make_method("mc_dropout").metadata
    assert bo_qei.display_name == "BO-qEI adaptation"
    assert "gpytorch_exact_gp" in bo_qei.adaptations
    assert "native_pytorch_joint_qei" in bo_qei.adaptations
    assert ga_on_gp.display_name == "GA on GP adaptation"
    assert "gpytorch_exact_gp" in ga_on_gp.adaptations
    assert ga_on_gp.source_commit
    assert mc_dropout.display_name == "MC-Dropout adaptation"
    assert {
        bo_qei.implementation_kind.value,
        ga_on_gp.implementation_kind.value,
        mc_dropout.implementation_kind.value,
    } == {"multi_fidelity_adaptation"}


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
        MethodSpec("ltr", dict(_METHOD_CONFIGS)["ltr"]),
        MethodSpec("roma", dict(_METHOD_CONFIGS)["roma"]),
        MethodSpec("ict", dict(_METHOD_CONFIGS)["ict"]),
        MethodSpec("tri_mentoring", dict(_METHOD_CONFIGS)["tri_mentoring"]),
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
        MethodSpec(
            "standard_ga",
            {
                "hidden_size": 8,
                "surrogate_epochs": 1,
                "batch_size": 2,
                "solver_steps": 1,
            },
        ),
        MethodSpec(
            "cma_es",
            {
                "generations": 1,
                "population_size": 4,
                "ensemble_size": 2,
                "hidden_size": 8,
                "surrogate_epochs": 1,
                "batch_size": 2,
            },
        ),
        MethodSpec(
            "reinforce",
            {
                "iterations": 1,
                "reinforce_batch_size": 8,
                "ensemble_size": 2,
                "hidden_size": 8,
                "surrogate_epochs": 1,
                "batch_size": 2,
            },
        ),
        MethodSpec(
            "bo_qei",
            {
                "gp_training_steps": 1,
                "acquisition_steps": 1,
                "mc_samples": 4,
            },
        ),
        MethodSpec(
            "ga_on_gp",
            {
                "gp_training_steps": 1,
                "solver_steps": 1,
            },
        ),
        MethodSpec(
            "mc_dropout",
            {
                "hidden_size": 8,
                "epochs": 1,
                "batch_size": 2,
                "mc_samples": 4,
                "particle_steps": 1,
            },
        ),
        MethodSpec(
            "coms",
            {
                "hidden_size": 8,
                "epochs": 1,
                "batch_size": 2,
                "adversarial_steps": 1,
                "particle_steps": 1,
            },
        ),
        MethodSpec("bdi", {"steps": 1}),
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
        "ltr",
        "roma",
        "ict",
        "tri_mentoring",
        "best_logged",
        "random_search",
        "sobol",
        "offline_mlp",
        "standard_ga",
        "cma_es",
        "reinforce",
        "bo_qei",
        "ga_on_gp",
        "mc_dropout",
        "coms",
        "bdi",
    }
