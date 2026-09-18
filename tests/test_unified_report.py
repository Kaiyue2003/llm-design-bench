import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from llm_design_bench.evaluation import (
    LEGACY_PUBLICATION_SOURCE,
    RESULT_SCHEMA_VERSION,
    BenchmarkTaskSpec,
    BenchmarkTrial,
    SeedBenchmarkConfig,
    load_legacy_publication_results,
    make_synthetic_task_spec,
    run_benchmark_suite,
    write_unified_report,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata
from llm_design_bench.spaces import SimplexSpace
from llm_design_bench.types import CandidateBatch


ROOT = Path(__file__).resolve().parents[1]


class SuiteEvaluator:
    mixture_dim = 3
    target_model_scale = 1000.0
    target_training_steps = 19_500.0

    def __init__(self, target: np.ndarray) -> None:
        self.target = target
        self.predict_calls = 0

    def at_target_fidelity(self, designs: np.ndarray) -> CandidateBatch:
        return CandidateBatch.at_fidelity(
            designs,
            self.target_model_scale,
            self.target_training_steps,
        )

    def predict(self, batch: CandidateBatch) -> np.ndarray:
        self.predict_calls += 1
        return -np.square(batch.mixtures - self.target).sum(axis=1)


def _task_spec(
    task_id: str,
    target: np.ndarray,
    evaluators: list[SuiteEvaluator],
) -> BenchmarkTaskSpec:
    def trial_factory(seed: int) -> BenchmarkTrial:
        offset = (seed - 38) * 0.01
        designs = torch.tensor(
            [
                [0.6 - offset, 0.3 + offset, 0.1],
                [0.3, 0.4, 0.3],
                [0.1, 0.2, 0.7],
            ],
            dtype=torch.float32,
        )
        evaluator = SuiteEvaluator(target)
        evaluators.append(evaluator)
        logged = CandidateBatch.at_fidelity(designs.numpy(), 1000.0, 19_500.0)
        utility = evaluator.predict(logged)
        problem = OfflineProblem(
            train_designs=designs,
            train_context=torch.tensor(
                [[20.0, 1000.0], [150.0, 5000.0], [1000.0, 19_500.0]]
            ),
            train_utility=torch.tensor(utility, dtype=torch.float32),
            target_context=torch.tensor([1000.0, 19_500.0]),
            design_space=SimplexSpace(3),
            metadata=ProblemMetadata(task_name=task_id, objective_name="toy_loss"),
        )
        reference = np.concatenate([utility, np.array([-1.0, 0.0])])
        return BenchmarkTrial(
            evaluator_task=evaluator,
            problem=problem,
            reference_utility=reference,
            dataset_seed=seed,
            d_best_utility=float(utility.max()),
        )

    return BenchmarkTaskSpec(
        task_id=task_id,
        display_name=task_id.replace("_", " ").title(),
        suite="toy",
        category="continuous",
        category_display_name="Continuous",
        trial_factory=trial_factory,
        normalization_reference_id=f"{task_id}_full_reference",
    )


def test_suite_runner_writes_one_schema_for_multiple_tasks(tmp_path) -> None:
    evaluators: list[SuiteEvaluator] = []
    tasks = [
        _task_spec("toy_alpha", np.array([0.7, 0.2, 0.1]), evaluators),
        _task_spec("toy_beta", np.array([0.2, 0.6, 0.2]), evaluators),
    ]
    result = run_benchmark_suite(
        tasks,
        ["best_logged", "random_search"],
        config=SeedBenchmarkConfig(
            experiment_id="unified-test",
            seeds=(38, 39),
            candidate_budget=4,
            results_dir=tmp_path,
        ),
        metadata={"protocol": "test"},
    )

    assert len(result.per_seed) == 2 * 2 * 2
    assert len(result.summary) == 2 * 2
    assert len(result.ranks) == 2
    assert len(result.d_best) == 2
    assert set(result.per_seed["schema_version"]) == {RESULT_SCHEMA_VERSION}
    assert set(result.per_seed["result_source"]) == {"unified_runner"}
    assert set(result.per_seed["dataset_seed"]) == {38, 39}
    assert set(result.per_seed["status"]) == {"success"}
    assert (result.summary["successful_runs"] == 2).all()
    assert (result.summary["failed_runs"] == 0).all()
    assert np.isfinite(result.summary["refnorm_max_score_std"]).all()
    assert np.isfinite(result.summary["refnorm_max_score_se"]).all()
    assert all(evaluator.predict_calls == 3 for evaluator in evaluators)

    expected_files = {
        ".suite.lock",
        "README.md",
        "benchmark_table.tex",
        "d_best_summary.csv",
        "method_seed_results.csv",
        "method_seed_summary.csv",
        "rank_summary.csv",
        "run_metadata.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}
    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["table_uncertainty"] == "standard_error"
    assert metadata["uncertainty_columns"]["std"].startswith("sample standard")
    assert metadata["failed_runs_retained"] is True
    assert metadata["protocol"] == "test"

    markdown = (tmp_path / "README.md").read_text(encoding="utf-8")
    latex = (tmp_path / "benchmark_table.tex").read_text(encoding="utf-8")
    assert "mean +/- standard error" in markdown
    assert "D(best)" in markdown
    assert "Method provenance" in markdown
    assert "\\pm" in latex
    assert "\\mathcal{D}" in latex


def test_synthetic_task_spec_pairs_dataset_and_method_seed() -> None:
    spec = make_synthetic_task_spec("ackley", logged_samples=8)
    trial = spec.trial_factory(41)

    assert spec.task_id == "ackley"
    assert spec.suite == "synthetic"
    assert trial.dataset_seed == 41
    assert trial.problem.sample_count == 8
    assert trial.problem.metadata.extra["utility_transform"] == "negative_objective"
    np.testing.assert_allclose(
        trial.problem.train_utility.numpy(),
        trial.reference_utility,
    )


def test_legacy_publication_results_convert_without_inventing_metrics(tmp_path) -> None:
    legacy_dir = ROOT / "reference_results" / "publication"
    converted = load_legacy_publication_results(legacy_dir)

    assert len(converted) == 240
    assert set(converted["result_source"]) == {LEGACY_PUBLICATION_SOURCE}
    assert set(converted["status"]) == {"success"}
    assert set(converted["method_id"]) == {"best_logged", "coms", "bdi"}
    assert converted["candidate_diversity"].isna().all()
    assert converted["method_seconds"].isna().all()
    assert set(converted["implementation_kind"]) == {
        "native_baseline",
        "lightweight_adaptation",
    }

    report = write_unified_report(
        converted,
        tmp_path,
        metadata={"legacy_source": str(legacy_dir)},
    )
    assert len(report.summary) == 30
    assert len(report.ranks) == 3
    assert len(report.d_best) == 10
    assert report.summary["candidate_diversity_n"].eq(0).all()
    assert report.summary["method_seconds_n"].eq(0).all()
    assert (tmp_path / "method_seed_results.csv").is_file()
    assert (tmp_path / "benchmark_table.tex").is_file()


def test_unified_report_rejects_duplicate_task_method_seed_rows(tmp_path) -> None:
    legacy_dir = ROOT / "reference_results" / "publication"
    converted = load_legacy_publication_results(legacy_dir).iloc[[0]].copy()
    duplicates = pd.concat([converted, converted], ignore_index=True)

    try:
        write_unified_report(duplicates, tmp_path)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate rows should be rejected")


def test_unified_report_rejects_reserved_metadata_override(tmp_path) -> None:
    legacy_dir = ROOT / "reference_results" / "publication"
    converted = load_legacy_publication_results(legacy_dir).iloc[[0]].copy()

    try:
        write_unified_report(
            converted,
            tmp_path,
            metadata={"schema_version": 999},
        )
    except ValueError as exc:
        assert "reserved field" in str(exc)
    else:
        raise AssertionError("reserved report metadata should not be overridden")


def test_failed_only_method_remains_in_rank_and_markdown_outputs(tmp_path) -> None:
    legacy_dir = ROOT / "reference_results" / "publication"
    converted = load_legacy_publication_results(legacy_dir).iloc[[0]].copy()
    converted["status"] = "failed"
    for column in (
        "raw_max_utility",
        "raw_median_utility",
        "raw_mean_utility",
        "refnorm_max_score",
        "refnorm_median_score",
        "refnorm_mean_score",
    ):
        converted[column] = np.nan

    report = write_unified_report(converted, tmp_path)

    assert len(report.ranks) == 1
    assert report.ranks.iloc[0]["tasks_ranked"] == 0
    assert report.ranks.iloc[0]["failed_runs"] == 1
    assert np.isnan(report.ranks.iloc[0]["mean_rank"])
    markdown = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "| Best Logged | -- | -- |" in markdown
