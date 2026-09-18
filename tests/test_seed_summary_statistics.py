"""Retain aggregation regression coverage on the current result schema."""

import json

import numpy as np
import pandas as pd
import pytest

from llm_design_bench.evaluation.seed_runner import summarize_seed_results


@pytest.mark.parametrize("values", [(1.0, 3.0), (1.0, 1.0), (2.0,)])
def test_seed_summary_reports_sample_standard_deviation_and_standard_error(values):
    frame = pd.DataFrame(
        [
            {
                "experiment_id": "aggregation-contract",
                "result_source": "unified_runner",
                "phase": "exploratory",
                "required_seeds_json": json.dumps(list(range(38, 38 + len(values)))),
                "run_id": "coms",
                "method_id": "coms",
                "method_seed": seed,
                "status": "success",
                "raw_max_utility": value,
                "refnorm_max_score": value,
            }
            for seed, value in enumerate(values, start=38)
        ]
    )

    summary = summarize_seed_results(frame).iloc[0]
    for metric in ("raw_max_utility", "refnorm_max_score"):
        assert summary[f"{metric}_n"] == len(values)
        assert summary[f"{metric}_mean"] == np.mean(values)
        if len(values) > 1:
            expected_std = np.std(values, ddof=1)
            assert summary[f"{metric}_std"] == expected_std
            assert summary[f"{metric}_se"] == expected_std / np.sqrt(len(values))
        else:
            assert np.isnan(summary[f"{metric}_std"])
            assert np.isnan(summary[f"{metric}_se"])
