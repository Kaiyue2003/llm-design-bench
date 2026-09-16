"""Exercise the real adapter and random method with tiny local test data."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_design_bench import optimizers
from llm_design_bench.tasks.data_recipes import DataRecipesTask
from test_data_recipes_filter import _write_fake_data_recipes


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path_mode", ["explicit", "environment"])
def test_demo_uses_real_visible_split_and_adapter_without_checkpoint(
    tmp_path, monkeypatch, capsys, path_mode
):
    checkout = tmp_path / "trusted-upstream"
    checkout.mkdir()
    _write_fake_data_recipes(checkout)
    unrelated = tmp_path / "unrelated-working-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    spec = importlib.util.spec_from_file_location(
        "quickstart_integration", ROOT / "examples/data_recipes_quickstart.py"
    )
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)

    oracle_calls = []
    observed = []

    class FakeBenchmark:
        def _raw_func_with_model_scale(self, steps, scale, mixture, *, with_exp):
            assert with_exp is False
            oracle_calls.append((steps, scale, mixture.copy()))
            return 1.0 + mixture[0]  # Raw loss; the real adapter negates it once.

    def forbidden(*args, **kwargs):
        pytest.fail("the example test must not load an upstream model or checkpoint")

    monkeypatch.setattr(DataRecipesTask, "_load_upstream_benchmarks", forbidden)
    monkeypatch.setattr(DataRecipesTask, "_get_benchmark", lambda self: FakeBenchmark())
    original_make = optimizers.make_method

    def inspected_make(method_id):
        assert method_id == "random_search"
        method = original_make(method_id)
        original_run = method.run

        def run(problem, context):
            assert not oracle_calls
            assert problem.metadata.extra["utility_transform"] == "negative_loss"
            # The singleton 20M row and the worse-loss 1B row are visible;
            # the better-loss 1B row remains hidden from the method.
            assert problem.metadata.extra["visible_row_ids"] == [0, 2]
            torch.testing.assert_close(
                problem.train_utility, torch.tensor([-2.5, -1.75], dtype=torch.float64)
            )
            assert context.method_seed == 7 and context.candidate_budget == 4
            assert context.dtype == torch.float32 and context.device.type == "cpu"
            observed.append(problem)
            return original_run(problem, context)

        monkeypatch.setattr(method, "run", run)
        return method

    monkeypatch.setattr(optimizers, "make_method", inspected_make)
    args = ["--seed", "7", "--candidate-budget", "4"]
    if path_mode == "explicit":
        args.extend(["--data-recipes-root", str(checkout)])
    else:
        monkeypatch.setenv("DATA_RECIPES_ROOT", str(checkout))
    original_files = {
        path: path.read_bytes() for path in checkout.rglob("*") if path.is_file()
    }

    example.main(args)

    assert len(observed) == 1
    assert len(oracle_calls) == 4
    assert all(steps == 195.0 and scale == 100.0 for steps, scale, _ in oracle_calls)
    mixtures = np.array([mixture for _, _, mixture in oracle_calls])
    assert mixtures.shape == (4, 5) and (mixtures >= 0).all()
    np.testing.assert_allclose(mixtures.sum(axis=1), 1.0, atol=1e-6)
    best_loss = float(np.min(1.0 + mixtures[:, 0]))
    output = capsys.readouterr().out
    assert f"Best utility: {-best_loss:.6f}" in output
    assert f"Lowest StackExchange cross entropy loss: {best_loss:.6f}" in output
    assert not list(unrelated.iterdir())
    assert {
        path: path.read_bytes() for path in checkout.rglob("*") if path.is_file()
    } == original_files
