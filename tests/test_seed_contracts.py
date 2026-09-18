"""Modern seed coverage cannot acquire legacy ranking rules by omitting fields."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_design_bench.evaluation.seed_contracts import (
    LEGACY_PUBLICATION_SOURCE,
    seed_group_contract,
    validate_seed_contracts,
    validate_seed_group,
)
from llm_design_bench.evaluation.seed_statistics import summarize_seed_results
from llm_design_bench.evaluation.unified_report import load_legacy_publication_results


def _rows(seeds=(38,), *, phase="formal", required=tuple(range(38, 46))):
    return pd.DataFrame(
        [
            {
                "experiment_id": "seed-contract-test",
                "suite": "data_mixture",
                "task_id": "task-a",
                "run_id": "method-a",
                "method_id": "method-a",
                "result_source": "unified_runner",
                "phase": phase,
                "required_seeds_json": json.dumps(required),
                "method_seed": seed,
                "status": "success",
                "raw_max_utility": float(seed),
                "refnorm_max_score": float(seed) / 100,
            }
            for seed in seeds
        ]
    )


@pytest.mark.parametrize(
    ("seeds", "failed", "rank", "missing"),
    [
        (tuple(range(38, 46)), False, True, 0),
        (tuple(range(38, 45)), False, False, 1),
        (tuple(range(38, 46)), True, False, 0),
        ((38,), False, False, 7),
    ],
)
def test_formal_requires_all_eight_successes_without_rejecting_partial_runs(
    seeds, failed, rank, missing
):
    frame = _rows(seeds)
    if failed:
        frame.loc[frame.index[-1], "status"] = "failed"
    result = summarize_seed_results(frame).iloc[0]
    assert bool(result["rank_eligible"]) is rank
    assert bool(result["complete_seed_set"]) is rank
    assert result["requested_runs"] == 8
    assert result["attempted_runs"] == len(seeds)
    assert result["missing_runs"] == missing
    assert result["failed_runs"] == int(failed)
    assert result["successful_runs"] == len(seeds) - int(failed)


def test_required_seed_set_order_is_not_a_configuration_difference():
    frame = _rows(tuple(range(38, 46)))
    frame.loc[1, "required_seeds_json"] = json.dumps(list(reversed(range(38, 46))))
    result = summarize_seed_results(frame).iloc[0]
    assert bool(result["rank_eligible"])
    assert result["required_seeds_json"] == "[38,39,40,41,42,43,44,45]"


@pytest.mark.parametrize(
    ("phase", "seeds", "required", "complete", "rank"),
    [
        ("pilot", (0,), (0,), True, False),
        ("exploratory", (2, 4), (2, 4), True, True),
        ("exploratory", (2,), (2, 4), False, False),
    ],
)
def test_pilot_and_exploratory_keep_their_distinct_ranking_rules(
    phase, seeds, required, complete, rank
):
    result = summarize_seed_results(_rows(seeds, phase=phase, required=required)).iloc[
        0
    ]
    assert bool(result["complete_seed_set"]) is complete
    assert bool(result["rank_eligible"]) is rank


@pytest.mark.parametrize("column", ["phase", "required_seeds_json"])
@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, "", "drop-column"])
def test_modern_rows_cannot_fall_back_to_legacy_when_metadata_is_missing(
    column, missing
):
    frame = _rows()
    if isinstance(missing, str) and missing == "drop-column":
        frame = frame.drop(columns=column)
    else:
        frame[column] = pd.Series([missing], dtype=object)
    with pytest.raises(ValueError, match="phase|required_seeds_json"):
        summarize_seed_results(frame)


@pytest.mark.parametrize("source", [None, np.nan, pd.NA, "", "unknown_source"])
def test_unknown_or_missing_source_cannot_fall_back_to_legacy(source):
    frame = _rows()
    frame["result_source"] = pd.Series([source], dtype=object)
    with pytest.raises(ValueError, match="result_source"):
        summarize_seed_results(frame)


def test_source_column_is_required_even_for_minimal_summary_inputs():
    with pytest.raises(KeyError, match="result_source"):
        summarize_seed_results(_rows().drop(columns="result_source"))


@pytest.mark.parametrize("phase", ["legacy", "unknown", "Formal"])
def test_modern_unknown_or_legacy_phase_is_rejected(phase):
    frame = _rows(phase=phase)
    with pytest.raises(ValueError, match="phase"):
        summarize_seed_results(frame)


@pytest.mark.parametrize(
    "required_json",
    [
        "null",
        "{}",
        "[]",
        '"38"',
        "[38,38]",
        "[true]",
        "[38.0]",
        '["38"]',
        "[38,null]",
        "[38,NaN]",
        "[-1]",
        "not-json",
    ],
)
def test_required_json_must_contain_only_unique_non_negative_integer_literals(
    required_json,
):
    frame = _rows(phase="exploratory", required=(38,))
    frame["required_seeds_json"] = required_json
    with pytest.raises(ValueError, match="required_seeds_json"):
        summarize_seed_results(frame)


@pytest.mark.parametrize(
    ("phase", "seed", "required"),
    [("formal", 38, (38,)), ("pilot", 38, (38,)), ("pilot", 0, (0, 1))],
)
def test_phase_cannot_redefine_its_required_seed_set(phase, seed, required):
    with pytest.raises(ValueError, match=f"{phase} required_seeds"):
        summarize_seed_results(_rows((seed,), phase=phase, required=required))


@pytest.mark.parametrize(
    "seed",
    [
        True,
        False,
        np.bool_(True),
        38.9,
        np.nan,
        np.inf,
        -np.inf,
        -1,
        "38.9",
        "38.000000000000000000000001",
        "NaN",
        "True",
        "",
        None,
        pd.NA,
    ],
)
def test_observed_seeds_are_never_truncated_or_coerced_from_booleans(seed):
    frame = _rows()
    frame["method_seed"] = pd.Series([seed], dtype=object)
    with pytest.raises(ValueError, match="method_seed"):
        summarize_seed_results(frame)


@pytest.mark.parametrize(
    "seed", [38.0, "38", "38.0", "3.8e1", " 38 ", np.int64(38), np.float32(38)]
)
def test_lossless_csv_seed_representations_remain_valid(seed):
    frame = _rows()
    frame["method_seed"] = pd.Series([seed], dtype=object)
    assert seed_group_contract(frame).observed_seeds == {38}
    assert summarize_seed_results(frame).iloc[0]["missing_runs"] == 7


@pytest.mark.parametrize("duplicate", [38, 38.0, "38", "38.0"])
def test_duplicate_observed_seed_aliases_are_rejected(duplicate):
    frame = _rows((38, 39))
    frame["method_seed"] = pd.Series([38, duplicate], dtype=object)
    with pytest.raises(ValueError, match="duplicate method/task/seed"):
        summarize_seed_results(frame)


def test_observed_seed_must_belong_to_declared_set():
    with pytest.raises(ValueError, match="observed seed is not in required_seeds"):
        summarize_seed_results(_rows((37,)))


def test_contracts_are_isolated_by_task_and_method():
    complete = _rows(tuple(range(38, 46)))
    other_method = _rows()
    other_method["run_id"] = "method-b"
    other_method["method_id"] = "method-b"
    other_task = _rows((0,), phase="pilot", required=(0,))
    other_task["task_id"] = "task-b"
    frame = pd.concat([complete, other_method, other_task], ignore_index=True)
    validate_seed_contracts(frame)
    result = summarize_seed_results(frame)
    assert result["rank_eligible"].tolist() == [True, False, False]
    assert result["phase"].tolist() == ["formal", "formal", "pilot"]


def test_display_metadata_cannot_hide_duplicate_identity_rows():
    frame = pd.concat([_rows(), _rows()], ignore_index=True)
    frame["task_display_name"] = ["Original task name", "Altered task name"]
    with pytest.raises(ValueError, match="duplicate method/task/seed"):
        summarize_seed_results(frame)


@pytest.mark.parametrize("field", ["phase", "required_seeds_json"])
def test_modern_contract_cannot_change_within_one_group(field):
    frame = _rows((38, 39), phase="exploratory", required=(38, 39))
    frame.loc[1, field] = "formal" if field == "phase" else "[38,39,40]"
    with pytest.raises(ValueError, match="consistent within a task/method"):
        summarize_seed_results(frame)


def test_explicit_legacy_preserves_historical_missing_field_behavior():
    frame = _rows((38, 39)).drop(columns=["phase", "required_seeds_json"])
    frame["result_source"] = LEGACY_PUBLICATION_SOURCE
    frame.loc[1, "status"] = "failed"
    result = summarize_seed_results(frame).iloc[0]
    assert result["phase"] == "legacy"
    assert result["requested_runs"] == 2
    assert not bool(result["complete_seed_set"])
    assert bool(result["rank_eligible"])  # The old explicit-legacy rule is unchanged.


def test_actual_legacy_adapter_output_still_aggregates_without_phase():
    source = Path(__file__).resolve().parents[1] / "reference_results" / "publication"
    frame = load_legacy_publication_results(source).iloc[:2]
    assert "phase" not in frame
    summary = summarize_seed_results(frame)
    assert set(summary["phase"]) == {"legacy"}
    assert summary["rank_eligible"].all()


def test_explicit_legacy_with_modern_phase_uses_the_declared_required_set():
    frame = _rows(phase="exploratory", required=(38, 39))
    frame["result_source"] = LEGACY_PUBLICATION_SOURCE
    result = summarize_seed_results(frame).iloc[0]
    assert result["missing_runs"] == 1
    assert not bool(result["rank_eligible"])


def test_legacy_and_modern_cannot_mix_within_one_task_method():
    frame = _rows((38, 39))
    frame.loc[0, "result_source"] = LEGACY_PUBLICATION_SOURCE
    with pytest.raises(ValueError, match="result_source must be consistent"):
        summarize_seed_results(frame)


@pytest.mark.parametrize("status", ["complete", "", None, pd.NA])
def test_unknown_status_cannot_affect_success_coverage(status):
    frame = _rows()
    frame["status"] = pd.Series([status], dtype=object)
    with pytest.raises(ValueError, match="status must be success or failed"):
        summarize_seed_results(frame)


def test_empty_group_has_no_implicit_contract():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_seed_group([])


def test_invalid_group_error_identifies_task_method_and_observed_seeds():
    frame = _rows((38, 39))
    frame["method_seed"] = pd.Series([38, 39.5], dtype=object)
    with pytest.raises(ValueError) as caught:
        summarize_seed_results(frame)
    message = str(caught.value)
    assert "task_id='task-a'" in message
    assert "run_id='method-a'" in message
    assert "method_seeds=[38, 39.5]" in message
