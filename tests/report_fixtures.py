"""Invented current-schema report rows, independent of archived experiments."""

import pandas as pd

from llm_design_bench.evaluation.seed_types import RESULT_SCHEMA_VERSION
from llm_design_bench.evaluation.seed_statistics import reference_normalize


def report_rows() -> pd.DataFrame:
    """Return a fresh valid one-seed exploratory row without running an oracle."""
    row = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_source": "unified_runner",
        "experiment_id": "invented-report-fixture",
        "suite": "data_mixture",
        "task_id": "fixture-mixture",
        "task_display_name": "Fixture Mixture",
        "category": "data_mixture",
        "category_display_name": "Data Mixture",
        "run_id": "best_logged",
        "method_id": "best_logged",
        "method_display_name": "Best Logged",
        "display_name": "Best Logged",
        "family": "reference",
        "implementation_kind": "native_baseline",
        "adaptations_json": "[]",
        "source_url": None,
        "source_commit": None,
        "requested_method_config_json": "{}",
        "method_config_json": "{}",
        "method_seed": 38,
        "dataset_seed": 0,
        "split_seed": 0,
        "phase": "exploratory",
        "required_seeds_json": "[38]",
        "status": "success",
        "candidate_budget": 128,
        "train_size": 12,
        "dtype": "torch.float64",
        "device": "cpu",
        "reference_min_utility": -4.0,
        "reference_max_utility": -1.0,
        "normalization_reference_id": "invented-reference",
    }
    for raw, normalized, value in (
        ("raw_max_utility", "refnorm_max_score", -1.6),
        ("raw_median_utility", "refnorm_median_score", -2.2),
        ("raw_mean_utility", "refnorm_mean_score", -2.6),
        ("d_best_utility", "refnorm_d_best_score", -2.5),
    ):
        row[raw] = value
        row[normalized] = reference_normalize(
            value, reference_low=-4.0, reference_high=-1.0
        )
    return pd.DataFrame([row])
