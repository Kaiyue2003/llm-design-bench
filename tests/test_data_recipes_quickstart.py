"""Exercise the runnable quickstart without loading upstream data or an oracle."""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import OfflineProblem, ProblemMetadata
from llm_design_bench.spaces import SimplexSpace
from llm_design_bench.types import CandidateBatch


EXAMPLE = Path(__file__).parents[1] / "examples" / "data_recipes_quickstart.py"


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "data_recipes_quickstart_test", EXAMPLE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example():
    return _load_example()


class _PublicEvaluator:
    """Expose only the public adapter API used by the example."""

    def __init__(self, events):
        self.events = events
        self.batches = []
        self.override_utility = None

    def at_target_fidelity(self, mixtures):
        assert self.events[-1] == "candidates_ready"
        assert isinstance(mixtures, np.ndarray)
        self.events.append("target_fidelity")
        return CandidateBatch.at_fidelity(mixtures, 1000.0, 19500.0)

    def predict(self, batch):
        assert self.events[-1] == "target_fidelity"
        self.events.append("oracle_evaluation")
        self.batches.append(batch)
        if self.override_utility is not None:
            return self.override_utility
        # Deliberately negative utilities make reversed objective direction visible.
        return -np.linspace(1.25, 3.0, len(batch))


@pytest.fixture
def harness(example, monkeypatch):
    import llm_design_bench.evaluation as evaluation
    import llm_design_bench.optimizers as optimizers

    events = []
    evaluator = _PublicEvaluator(events)
    problem = OfflineProblem(
        train_designs=torch.tensor(
            [[0.2] * 5, [0.4, 0.1, 0.2, 0.1, 0.2]], dtype=torch.float32
        ),
        train_context=torch.tensor([[150.0, 10000.0], [1000.0, 19500.0]]),
        train_utility=torch.tensor([-3.0, -2.0]),
        target_context=torch.tensor([1000.0, 19500.0]),
        design_space=SimplexSpace(5),
        metadata=ProblemMetadata(
            task_name="quickstart_fake",
            objective_name="StackExchange CrossEntropyLoss",
            extra={"utility_transform": "negative_loss"},
        ),
    )
    state = SimpleNamespace(
        events=events,
        evaluator=evaluator,
        problem=problem,
        builder_kwargs=[],
        trial_seeds=[],
        contexts=[],
        candidates=[],
    )

    def trial_factory(seed):
        events.append("load_visible_problem")
        state.trial_seeds.append(seed)
        return SimpleNamespace(problem=problem, evaluator_task=evaluator)

    def task_spec(**kwargs):
        events.append("build_task_spec")
        state.builder_kwargs.append(kwargs)
        return SimpleNamespace(trial_factory=trial_factory)

    def method_factory(name):
        assert name == "random_search"
        method = make_method(name)

        def run(method_problem, context):
            assert method_problem is problem
            assert events[-1] == "load_visible_problem"
            events.append("generate_candidates")
            state.contexts.append(context)
            result = method.run(method_problem, context)
            assert events[-1] == "generate_candidates"
            state.candidates.append(result.candidates.detach().clone())
            events.append("candidates_ready")
            return result

        return SimpleNamespace(run=run)

    monkeypatch.setattr(evaluation, "make_data_recipes_task_spec", task_spec)
    monkeypatch.setattr(optimizers, "make_method", method_factory)
    return state


def test_import_does_not_construct_tasks_or_run_methods(monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "torch", "llm_design_bench"}:
            pytest.fail(
                "importing an example must not initialize benchmark dependencies"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _load_example()
    assert capsys.readouterr().out == ""


def test_help_exits_without_loading_data(example, monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "torch", "llm_design_bench"}:
            pytest.fail("--help must not initialize benchmark dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as error:
        example.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--data-recipes-root" in output
    assert "--candidate-budget" in output
    assert "--seed" in output


@pytest.mark.parametrize(
    ("args", "root", "budget", "seed"),
    [
        ([], None, 128, 0),
        (
            [
                "--data-recipes-root",
                "upstream data",
                "--candidate-budget",
                "7",
                "--seed",
                "42",
            ],
            Path("upstream data"),
            7,
            42,
        ),
        (["--candidate-budget", "1", "--seed", "1"], None, 1, 1),
    ],
)
def test_quickstart_runs_offline_then_evaluates_exact_budget(
    example, harness, capsys, args, root, budget, seed
):
    example.main(args)

    assert harness.builder_kwargs == [
        {
            "data_recipes_root": root,
            "metric_index": 4,
            "train_min_percentile": 0,
            "train_max_percentile": 40,
        }
    ]
    assert harness.trial_seeds == [seed]
    context = harness.contexts[0]
    assert context.method_seed == seed
    assert context.candidate_budget == budget
    assert context.device == torch.device("cpu")
    assert context.dtype == torch.float32
    assert harness.events == [
        "build_task_spec",
        "load_visible_problem",
        "generate_candidates",
        "candidates_ready",
        "target_fidelity",
        "oracle_evaluation",
    ]
    assert len(harness.evaluator.batches) == 1
    batch = harness.evaluator.batches[0]
    assert batch.mixtures.shape == (budget, 5)
    assert np.isfinite(batch.mixtures).all()
    assert (batch.mixtures >= 0).all()
    np.testing.assert_allclose(batch.mixtures.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(batch.model_scales, np.full(budget, 1000.0))
    np.testing.assert_array_equal(batch.training_steps, np.full(budget, 19500.0))
    np.testing.assert_array_equal(batch.mixtures, harness.candidates[0].numpy())

    output = capsys.readouterr().out
    assert "demo only" in output.lower()
    assert f"Generated {budget} candidate mixtures." in output
    assert "Best utility: -1.250000" in output
    assert "Lowest StackExchange cross entropy loss: 1.250000" in output


def test_main_without_argv_uses_command_line(example, harness, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", [str(EXAMPLE), "--candidate-budget", "3", "--seed", "9"]
    )
    example.main()
    assert harness.contexts[0].candidate_budget == 3
    assert harness.contexts[0].method_seed == 9


@pytest.mark.parametrize(
    "args",
    [
        ["--candidate-budget", "0"],
        ["--candidate-budget", "-2"],
        ["--candidate-budget", "1.5"],
        ["--candidate-budget", "nan"],
        ["--seed", "-1"],
        ["--seed", "1.5"],
        ["--seed", "nan"],
        ["--unknown-option"],
    ],
)
def test_invalid_cli_fails_before_loading_adapter(example, harness, args):
    with pytest.raises(SystemExit) as error:
        example.main(args)
    assert error.value.code == 2
    assert harness.events == []


def test_sampling_is_reproducible_without_oracle_feedback(example, harness):
    for seed in (42, 42, 43):
        example.main(["--candidate-budget", "11", "--seed", str(seed)])
    torch.testing.assert_close(
        harness.candidates[0], harness.candidates[1], rtol=0, atol=0
    )
    assert not torch.equal(harness.candidates[0], harness.candidates[2])
    assert len(harness.evaluator.batches) == 3


@pytest.mark.parametrize(
    "bad_utility",
    [
        np.array(-2.0),
        np.array([]),
        np.full(4, -2.0),
        np.full((3, 1), -2.0),
        np.array([-2.0, np.nan, -1.0]),
        np.array([-2.0, np.inf, -1.0]),
        np.array([-2.0, -np.inf, -1.0]),
    ],
)
def test_invalid_oracle_output_is_rejected(example, harness, capsys, bad_utility):
    harness.evaluator.override_utility = bad_utility
    with pytest.raises(ValueError):
        example.main(["--candidate-budget", "3"])
    assert len(harness.evaluator.batches) == 1
    output = capsys.readouterr().out
    assert "Best utility:" not in output
    assert "Lowest StackExchange cross entropy loss:" not in output
