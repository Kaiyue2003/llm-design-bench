import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from llm_design_bench.evaluation.run_artifacts import (
    result_directory_lock,
    verify_successful_attempt,
)
from llm_design_bench.evaluation.seed_runner import (
    DEFAULT_METHOD_SEEDS,
    MethodSpec,
    SeedBenchmarkConfig,
    run_method_seed_benchmark,
)
from llm_design_bench.evaluation.unified_report import (
    BenchmarkTaskSpec,
    BenchmarkTrial,
    run_benchmark_suite,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata
from llm_design_bench.spaces import SimplexSpace


def _problem():
    return OfflineProblem(
        train_designs=torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float64),
        train_context=torch.tensor([[20, 1000], [1000, 19500]], dtype=torch.float64),
        train_utility=torch.tensor([-3.0, -2.0], dtype=torch.float64),
        target_context=torch.tensor([1000, 19500], dtype=torch.float64),
        design_space=SimplexSpace(2),
        metadata=ProblemMetadata(
            task_name="artifacts-toy",
            objective_name="cross_entropy",
            extra={"utility_transform": "negative_loss"},
        ),
    )


class Evaluator:
    def __init__(self, directory, error=None):
        self.directory = directory
        self.error = error
        self.calls = 0

    def at_target_fidelity(self, candidates):
        # This assertion establishes the persistence boundary before *any*
        # task/oracle operation, not merely before predict().
        saved = list(self.directory.rglob("candidates.npz"))
        assert saved
        assert any(
            np.array_equal(np.load(path)["candidates"], candidates) for path in saved
        )
        return candidates

    def predict(self, candidates):
        self.calls += 1
        if self.error:
            raise self.error
        return -2.0 - candidates[:, 0]


def _config(directory, **overrides):
    return SeedBenchmarkConfig(
        results_dir=directory,
        save_artifacts=True,
        seeds=(38,),
        required_seeds=DEFAULT_METHOD_SEEDS,
        candidate_budget=4,
        phase="formal",
        package_commit="fixture-commit",
        provenance={"plan_id": "fixture-plan"},
        **overrides,
    )


def _run(directory, *, task=None, config=None, methods=None):
    return run_method_seed_benchmark(
        task or Evaluator(directory),
        _problem(),
        methods or ["random_search"],
        reference_utility=np.array([-4.0, -1.0]),
        config=config or _config(directory),
    )


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_candidates_precede_oracle_and_scores_include_raw_loss(tmp_path):
    result = _run(tmp_path, methods=[MethodSpec("random_search", dtype=torch.float64)])
    candidate_path = next(tmp_path.rglob("candidates.npz"))
    candidates = np.load(candidate_path)["candidates"]
    assert candidates.shape == (4, 2)
    assert candidates.dtype == np.float64
    evaluation = np.load(candidate_path.with_name("evaluation.npz"))
    np.testing.assert_array_equal(evaluation["utility"], -2.0 - candidates[:, 0])
    np.testing.assert_array_equal(evaluation["raw_loss"], -evaluation["utility"])
    np.testing.assert_allclose(
        evaluation["refnorm_score"], (evaluation["utility"] + 4) / 3
    )
    manifest = _json(candidate_path.with_name("manifest.json"))
    assert manifest["environment"]["torch"] == torch.__version__
    assert manifest["logical_config"]["dtype"] == "torch.float64"
    assert (
        json.loads(manifest["logical_config"]["provenance_json"])["plan_id"]
        == "fixture-plan"
    )
    saved = _json(candidate_path.with_name("result.json"))
    assert saved["status"] == "success"
    assert saved["total_seconds"] > 0
    assert result.summary.iloc[0]["successful_runs"] == 1
    assert result.summary.iloc[0]["missing_runs"] == 7
    assert not result.summary.iloc[0]["rank_eligible"]


def test_explicit_resume_does_not_train_or_evaluate_twice(tmp_path, monkeypatch):
    task = Evaluator(tmp_path)
    first = _run(tmp_path, task=task)
    with pytest.raises(FileExistsError, match="resume"):
        _run(tmp_path, task=task)
    result = _run(tmp_path, task=task, config=_config(tmp_path, resume=True))
    assert task.calls == 1
    assert len(list(tmp_path.rglob("result.json"))) == 1
    assert (
        result.per_seed.iloc[0]["raw_max_utility"]
        == first.per_seed.iloc[0]["raw_max_utility"]
    )


@pytest.mark.parametrize("artifact", ["candidates.npz", "evaluation.npz"])
def test_resume_rejects_missing_or_corrupted_success_artifact(tmp_path, artifact):
    _run(tmp_path)
    path = next(tmp_path.rglob(artifact))
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupted"):
        _run(tmp_path, config=_config(tmp_path, resume=True))
    with pytest.raises(ValueError, match="corrupted"):
        verify_successful_attempt(path.parent)


def test_resume_is_portable_after_results_directory_is_copied(tmp_path):
    source, destination = tmp_path / "original", tmp_path / "downloaded"
    first = _run(source)
    shutil.copytree(source, destination)
    task = Evaluator(destination)
    resumed = _run(destination, task=task, config=_config(destination, resume=True))
    assert task.calls == 0
    assert str(destination) in resumed.per_seed.iloc[0]["artifact_dir"]
    assert (
        resumed.per_seed.iloc[0]["artifact_relative_dir"]
        == first.per_seed.iloc[0]["artifact_relative_dir"]
    )


@pytest.mark.parametrize(
    "error", [RuntimeError("oracle unavailable"), KeyboardInterrupt()]
)
def test_fail_fast_and_interrupt_preserve_candidates_and_failure(tmp_path, error):
    with pytest.raises(type(error)):
        _run(
            tmp_path,
            task=Evaluator(tmp_path, error),
            config=_config(tmp_path, fail_fast=True),
        )
    saved = _json(next(tmp_path.rglob("result.json")))
    assert saved["status"] == "failed"
    assert saved["failure_stage"] == "evaluation"
    assert saved["raw_max_utility"] is None
    assert list(tmp_path.rglob("candidates.npz"))
    assert not list(tmp_path.rglob("evaluation.npz"))
    csv = pd.read_csv(tmp_path / "method_seed_results.csv")
    assert csv.iloc[0]["status"] == "failed"


def test_failure_requires_explicit_retry_and_keeps_previous_attempt(tmp_path):
    _run(tmp_path, task=Evaluator(tmp_path, RuntimeError("connection interrupted")))
    with pytest.raises(FileExistsError, match="infrastructure retry"):
        _run(tmp_path, config=_config(tmp_path, resume=True))
    result = _run(
        tmp_path,
        config=_config(tmp_path, infrastructure_retry_reason="Colab disconnected"),
    )
    files = sorted(tmp_path.rglob("result.json"))
    assert [_json(path)["status"] for path in files] == ["failed", "success"]
    assert result.per_seed.iloc[0]["status"] == "success"
    assert len(pd.read_csv(tmp_path / "method_seed_results.csv")) == 1


def test_retry_rejects_changed_logical_configuration(tmp_path):
    _run(tmp_path, task=Evaluator(tmp_path, RuntimeError("interrupted")))
    changed = replace(
        _config(tmp_path), candidate_budget=5, infrastructure_retry_reason="retry"
    )
    with pytest.raises(ValueError, match="logical configuration changed"):
        _run(tmp_path, config=changed)
    assert len(list(tmp_path.rglob("manifest.json"))) == 1


def test_failure_after_scores_clears_success_scores(tmp_path, monkeypatch):
    import llm_design_bench.evaluation.seed_runner as runner

    def fail_diagnostics(*args):
        raise RuntimeError("diagnostics failed")

    monkeypatch.setattr(runner, "_candidate_diagnostics", fail_diagnostics)
    result = _run(tmp_path)
    assert result.per_seed.iloc[0]["status"] == "failed"
    assert np.isnan(result.per_seed.iloc[0]["raw_max_utility"])
    assert list(tmp_path.rglob("evaluation.npz"))


def _task(directory):
    def factory(seed):
        return BenchmarkTrial(Evaluator(directory), _problem(), np.array([-4.0, -1.0]))

    return BenchmarkTaskSpec("toy", "Toy", "toy", "toy", "Toy", factory)


def test_suite_merges_seed_shards_and_only_ranks_complete_set(tmp_path):
    first = run_benchmark_suite(
        [_task(tmp_path)], ["random_search"], config=_config(tmp_path)
    )
    assert np.isnan(first.summary.iloc[0]["task_rank"])
    assert "[1/8]" in (tmp_path / "benchmark_table.tex").read_text(encoding="utf-8")
    later = replace(_config(tmp_path), seeds=tuple(range(39, 46)))
    complete = run_benchmark_suite([_task(tmp_path)], ["random_search"], config=later)
    assert len(complete.per_seed) == 8
    assert complete.summary.iloc[0]["missing_runs"] == 0
    assert complete.summary.iloc[0]["task_rank"] == 1
    assert complete.ranks.iloc[0]["tasks_ranked"] == 1
    assert len(list(tmp_path.rglob("candidates.npz"))) == 8


def test_suite_keeps_previous_method_and_pilot_is_not_ranked(tmp_path):
    config = replace(_config(tmp_path), phase="pilot", seeds=(0,), required_seeds=(0,))
    run_benchmark_suite([_task(tmp_path)], ["random_search"], config=config)
    result = run_benchmark_suite([_task(tmp_path)], ["best_logged"], config=config)
    assert set(result.per_seed["method_id"]) == {"random_search", "best_logged"}
    assert result.summary["complete_seed_set"].all()
    assert not result.summary["rank_eligible"].any()
    assert result.ranks["tasks_ranked"].eq(0).all()
    assert "not formal eight-seed results" in (
        tmp_path / "benchmark_table.tex"
    ).read_text(encoding="utf-8")


def test_concurrent_suite_is_rejected_before_training(tmp_path):
    with (
        result_directory_lock(tmp_path),
        pytest.raises(RuntimeError, match="another worker"),
    ):
        run_benchmark_suite(
            [_task(tmp_path)], ["random_search"], config=_config(tmp_path)
        )
    assert not list(tmp_path.rglob("candidates.npz"))


def test_formal_seed_contract_rejects_pilot_or_different_expected_set():
    with pytest.raises(ValueError, match="38-45"):
        SeedBenchmarkConfig(phase="formal", seeds=(0,))
    with pytest.raises(ValueError, match="subset"):
        SeedBenchmarkConfig(
            phase="formal", seeds=(0,), required_seeds=DEFAULT_METHOD_SEEDS
        )
    with pytest.raises(ValueError, match="pilot"):
        SeedBenchmarkConfig(phase="pilot", seeds=(38,))
