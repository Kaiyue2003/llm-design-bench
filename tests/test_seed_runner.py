import json

import numpy as np
import pytest
import torch

from llm_design_bench.evaluation.seed_runner import (
    MethodSpec,
    SeedBenchmarkConfig,
    run_method_seed_benchmark,
)
from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.problem import (
    MethodResult,
    OfflineProblem,
    ProblemMetadata,
)
from llm_design_bench.spaces import SimplexSpace
from llm_design_bench.types import CandidateBatch


class SeedEvaluatorTask:
    mixture_dim = 3
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


def _problem() -> OfflineProblem:
    return OfflineProblem(
        train_designs=torch.tensor(
            [
                [0.4, 0.4, 0.2],
                [0.2, 0.5, 0.3],
                [0.1, 0.2, 0.7],
            ]
        ),
        train_context=torch.tensor(
            [
                [20.0, 1_000.0],
                [150.0, 5_000.0],
                [1000.0, 19_500.0],
            ]
        ),
        train_utility=torch.tensor([-2.0, -1.5, -1.0]),
        target_context=torch.tensor([1000.0, 19_500.0]),
        design_space=SimplexSpace(3),
        metadata=ProblemMetadata(task_name="seed-runner-toy"),
    )


@register_method("seed_runner_random")
class SeedRunnerRandomMethod(OfflineBBOMethod):
    metadata = MethodMetadata(
        method_id="seed_runner_random",
        display_name="Seed Runner Random",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
        adaptations=("multi_fidelity",),
    )
    capabilities = MethodCapabilities(
        supports_simplex=True,
        supports_context=True,
        stochastic=True,
    )
    instances = 0

    def __init__(self, tag: str = "default") -> None:
        type(self).instances += 1
        self.tag = tag

    def optimize(self, problem, *, context, generator):
        return MethodResult(
            candidates=problem.design_space.sample(
                context.candidate_budget,
                generator=generator,
                device=context.device,
                dtype=context.dtype,
            ),
            training_summary={"tag": self.tag},
        )


@register_method("seed_runner_failure")
class SeedRunnerFailureMethod(OfflineBBOMethod):
    metadata = MethodMetadata(
        method_id="seed_runner_failure",
        display_name="Seed Runner Failure",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
    )

    def optimize(self, problem, *, context, generator):
        if context.method_seed == 39:
            raise RuntimeError("intentional seed failure")
        return MethodResult(
            candidates=problem.design_space.sample(
                context.candidate_budget,
                generator=generator,
                device=context.device,
                dtype=context.dtype,
            )
        )


@register_method("seed_runner_no_context")
class SeedRunnerNoContextMethod(SeedRunnerRandomMethod):
    metadata = MethodMetadata(
        method_id="seed_runner_no_context",
        display_name="Seed Runner No Context",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
    )
    capabilities = MethodCapabilities(supports_context=False)


def test_seed_runner_writes_per_seed_and_summary_results(tmp_path) -> None:
    SeedRunnerRandomMethod.instances = 0
    task = SeedEvaluatorTask()
    result = run_method_seed_benchmark(
        task,
        _problem(),
        [MethodSpec("seed_runner_random", {"tag": "created"})],
        reference_utility=np.array([-2.0, -1.0, 0.0]),
        config=SeedBenchmarkConfig(
            experiment_id="contract-test",
            normalization_reference_id="toy-target-reference",
            seeds=(38, 39),
            candidate_budget=4,
            results_dir=tmp_path,
        ),
    )

    assert SeedRunnerRandomMethod.instances == 2
    assert task.predict_calls == 2
    assert len(result.per_seed) == 2
    assert set(result.per_seed["status"]) == {"success"}
    assert result.summary.loc[0, "successful_runs"] == 2
    assert result.summary.loc[0, "failed_runs"] == 0
    assert result.summary.loc[0, "raw_max_utility_n"] == 2
    assert np.isfinite(result.summary.loc[0, "raw_max_utility_se"])
    assert np.isfinite(result.summary.loc[0, "raw_max_utility_ci95_low"])
    assert np.isfinite(result.summary.loc[0, "raw_max_utility_ci95_high"])
    assert (
        result.summary.loc[0, "raw_max_utility_min"]
        <= result.summary.loc[0, "raw_max_utility_max"]
    )
    assert result.summary.loc[0, "experiment_id"] == "contract-test"
    assert set(result.per_seed["normalization_reference_id"]) == {
        "toy-target-reference"
    }
    assert (tmp_path / "method_seed_results.csv").is_file()
    assert (tmp_path / "method_seed_summary.csv").is_file()
    assert (tmp_path / "method_seed_table.md").is_file()
    assert (tmp_path / "method_seed_table.tex").is_file()
    summary = json.loads(result.per_seed.loc[0, "training_summary_json"])
    assert summary["tag"] == "created"
    assert json.loads(result.per_seed.loc[0, "method_config_json"]) == {
        "tag": "created"
    }
    assert (
        result.per_seed.loc[0, "resolved_method_config_json"]
        == result.per_seed.loc[0, "method_config_json"]
    )
    for field in (
        "paper_url",
        "original_framework",
        "implementation_framework",
        "optional_dependencies_json",
        "config_schema_version",
    ):
        assert field in result.per_seed.columns

    row = result.per_seed.iloc[0]
    expected = (row["raw_max_utility"] + 2.0) / 2.0
    assert np.isclose(row["refnorm_max_score"], expected)


def test_seed_runner_retains_failed_runs_without_oracle_calls(tmp_path) -> None:
    task = SeedEvaluatorTask()
    result = run_method_seed_benchmark(
        task,
        _problem(),
        ["seed_runner_failure"],
        reference_utility=np.array([-2.0, 0.0]),
        config=SeedBenchmarkConfig(
            seeds=(38, 39, 40),
            candidate_budget=4,
            results_dir=tmp_path,
        ),
    )

    assert task.predict_calls == 2
    assert list(result.per_seed["status"]).count("failed") == 1
    failed = result.per_seed[result.per_seed["status"] == "failed"].iloc[0]
    assert failed["method_seed"] == 39
    assert failed["error_type"] == "RuntimeError"
    assert result.summary.loc[0, "successful_runs"] == 2
    assert result.summary.loc[0, "failed_runs"] == 1


def test_seed_runner_rejects_incompatible_context_method(tmp_path) -> None:
    task = SeedEvaluatorTask()

    with pytest.raises(ValueError, match="varying fidelity context"):
        run_method_seed_benchmark(
            task,
            _problem(),
            ["seed_runner_no_context"],
            reference_utility=np.array([-2.0, 0.0]),
            config=SeedBenchmarkConfig(
                seeds=(38,),
                candidate_budget=4,
                results_dir=tmp_path,
            ),
        )

    assert task.predict_calls == 0


def test_seed_config_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        SeedBenchmarkConfig(seeds=(38, 38))
