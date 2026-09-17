"""Check the synthetic example against real current methods and a local oracle."""

import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_design_bench.problem import OfflineProblem
from llm_design_bench.tasks.synthetic_functions import SyntheticFunctionTask


EXAMPLE = Path(__file__).parents[1] / "examples" / "synthetic_quickstart.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("synthetic_quickstart_test", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_import_does_not_load_dependencies_or_run_experiment(
    monkeypatch, capsys
):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "torch", "llm_design_bench"}:
            pytest.fail(f"example import must not load {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert callable(_load_example().main)
    assert capsys.readouterr().out == ""


def test_example_uses_logged_only_method_then_final_evaluation(
    monkeypatch, capsys, tmp_path
):
    import llm_design_bench
    import llm_design_bench.optimizers as optimizers

    real_factory = optimizers.make_method
    task = SyntheticFunctionTask("ackley", logged_samples=128, seed=38)
    real_predict = task.predict
    events = []
    evaluated = []
    generated = []

    def make_task(name, **kwargs):
        assert name == "synthetic-ackley"
        assert kwargs == {"logged_samples": 128, "seed": 38}
        return task

    def make_method(method_id, **kwargs):
        if method_id == "coms":
            assert kwargs == {"epochs": 20, "particle_steps": 20}
        else:
            assert method_id == "bdi"
            assert kwargs == {"steps": 20}
        method = real_factory(method_id, **kwargs)
        original_run = method.run

        def run(problem, context):
            assert isinstance(problem, OfflineProblem)
            assert problem.sample_count == 128
            assert context.method_seed == 38
            assert context.candidate_budget == 16
            assert context.device.type == "cpu"
            assert context.dtype == (
                torch.float64 if method_id == "bdi" else torch.float32
            )
            events.append(f"generate:{method_id}")
            result = original_run(problem, context)
            assert events[-1] == f"generate:{method_id}"
            generated.append(result.candidates.detach().cpu().numpy().copy())
            events.append(f"ready:{method_id}")
            return result

        monkeypatch.setattr(method, "run", run)
        return method

    def predict(batch):
        assert events[-1] in {"ready:coms", "ready:bdi"}
        np.testing.assert_array_equal(batch.mixtures, generated[-1])
        assert len(batch) == 16
        task.validate(batch)
        utility = real_predict(batch)
        evaluated.append(utility)
        events.append("evaluate")
        return utility

    monkeypatch.setattr(llm_design_bench, "make", make_task)
    monkeypatch.setattr(optimizers, "make_method", make_method)
    monkeypatch.setattr(task, "predict", predict)
    monkeypatch.chdir(tmp_path)
    _load_example().main()

    assert events == [
        "generate:coms",
        "ready:coms",
        "evaluate",
        "generate:bdi",
        "ready:bdi",
        "evaluate",
    ]
    output = capsys.readouterr().out
    assert "not a frozen pilot or formal benchmark run" in output
    for name, utility in zip(["COMs adaptation", "BDI adaptation"], evaluated):
        best = float(utility.max())
        assert f"{name}: best utility={best:.6f}, best objective={-best:.6f}" in output
    assert list(tmp_path.iterdir()) == []
