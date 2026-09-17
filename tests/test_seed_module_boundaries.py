"""Compatibility checks for the seed runner's typed internal module boundaries."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from llm_design_bench.evaluation import seed_runner
from llm_design_bench.evaluation import seed_persistence, seed_statistics, seed_types
from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodFamily,
    MethodMetadata,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata
from llm_design_bench.spaces import SimplexSpace


@pytest.mark.parametrize(
    ("name", "owner"),
    [
        ("MethodSpec", seed_types),
        ("SeedBenchmarkConfig", seed_types),
        ("SeedBenchmarkResult", seed_types),
        ("DEFAULT_SEED_BENCHMARK_CONFIG", seed_types),
        ("summarize_seed_results", seed_statistics),
        ("reference_normalize", seed_statistics),
        ("merge_result_rows", seed_persistence),
        ("validate_existing_result_config", seed_persistence),
    ],
)
def test_existing_public_imports_are_the_same_objects(name, owner):
    assert getattr(seed_runner, name) is getattr(owner, name)


def _fixture_row():
    problem = OfflineProblem(
        train_designs=torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float64),
        train_context=torch.tensor(
            [[20.0, 1000.0], [1000.0, 19500.0]], dtype=torch.float64
        ),
        train_utility=torch.tensor([-3.0, -2.0], dtype=torch.float64),
        target_context=torch.tensor([1000.0, 19500.0], dtype=torch.float64),
        design_space=SimplexSpace(2),
        metadata=ProblemMetadata(
            task_name="boundary-toy",
            objective_name="cross_entropy",
            extra={"utility_transform": "negative_loss"},
        ),
    )
    config = seed_types.SeedBenchmarkConfig(
        experiment_id="boundary-contract",
        seeds=(38, 39),
        candidate_budget=4,
        dtype=torch.float64,
        package_commit="fixture",
    )
    metadata = MethodMetadata(
        method_id="random_search",
        display_name="Random Search",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
    )
    row = seed_persistence._base_row(
        spec=seed_types.MethodSpec("random_search"),
        metadata=metadata,
        problem=problem,
        config=config,
        seed=38,
        reference_low=-4.0,
        reference_high=-1.0,
    )
    return row, problem


def test_base_row_json_matches_pre_refactor_contract():
    row, _ = _fixture_row()
    digest = hashlib.sha256(seed_types._json_dumps(row).encode()).hexdigest()
    # Captured from the pre-split runner with this exact, local-only fixture.
    assert digest == "67fef2cad5bb26a73810f845d82e1fda01e6149c1da3ce7420a4233e7a943516"


def test_base_row_keeps_csv_column_order_and_declares_every_field():
    row, _ = _fixture_row()
    expected = (
        "schema_version result_source run_id experiment_id method_id "
        "method_display_name display_name family implementation_kind adaptations_json "
        "source_url source_commit requested_method_config_json method_config_json "
        "package_commit provenance_json phase required_seeds_json task_id task_name "
        "objective_name problem_metadata_json target_context_json method_seed "
        "dataset_seed split_seed candidate_budget train_size reference_min_utility "
        "reference_max_utility normalization_reference_id d_best_utility device dtype "
        "status error_type error_message method_seconds evaluation_seconds "
        "total_seconds training_summary_json diagnostics_json environment_json "
        "artifact_dir artifact_relative_dir peak_gpu_memory_bytes logical_fingerprint "
        "infrastructure_retry_reason failure_stage raw_min_loss raw_median_loss "
        "raw_mean_loss unique_candidate_count unique_candidate_fraction "
        "candidate_diversity candidate_novelty raw_max_utility raw_median_utility "
        "raw_mean_utility refnorm_max_score refnorm_median_score refnorm_mean_score "
        "refnorm_d_best_score"
    ).split()
    assert list(row) == expected
    assert set(row) <= seed_types.SeedResultRow.__annotations__.keys()
    assert seed_types.SeedResultRow.__required_keys__ <= row.keys()


def test_logical_identity_matches_pre_refactor_contract():
    row, problem = _fixture_row()
    identity = seed_persistence._logical_config(row, problem, np.array([-4.0, -1.0]))
    digest = hashlib.sha256(seed_types._json_dumps(identity).encode()).hexdigest()
    assert digest == "baece8414434c5d5e07c60074c0c9d9d8c55495ef0b28f4fb1f1182bfd0e9b58"


def test_typed_score_update_preserves_order_and_other_fields():
    row, _ = _fixture_row()
    original_keys = list(row)
    seed_types._update_scores(
        row,
        seed_statistics._candidate_scores(
            np.array([-3.0, -2.0]), reference_low=-4.0, reference_high=-1.0
        ),
    )
    assert row["raw_max_utility"] == -2.0
    seed_types._update_scores(row, seed_statistics._empty_candidate_scores())
    assert list(row) == original_keys
    assert row["method_id"] == "random_search"
    assert all(np.isnan(row[name]) for name in seed_statistics._SCORE_COLUMNS)


def test_relocated_configuration_keeps_normalization_and_validation():
    config = seed_types.SeedBenchmarkConfig(seeds=(38,), device="cpu")
    assert config.device == torch.device("cpu")
    assert config.required_seeds == (38,)
    with pytest.raises(ValueError, match="formal required_seeds must be exactly"):
        replace(config, phase="formal")
    with pytest.raises(TypeError, match="floating point"):
        seed_types.MethodSpec("random_search", dtype=torch.int64)


def test_json_serialization_options_remain_stable():
    assert seed_types._json_dumps({"z": torch.float64, "a": [1, float("nan")]}) == (
        '{"a":[1,NaN],"z":"torch.float64"}'
    )
