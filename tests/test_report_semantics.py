"""Report inputs must preserve the meaning of saved normalized utilities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from llm_design_bench.evaluation.seed_statistics import reference_normalize
from llm_design_bench.evaluation.unified_report import (
    write_unified_report,
)


from report_fixtures import report_rows

PAIRS = (
    ("raw_max_utility", "refnorm_max_score"),
    ("raw_median_utility", "refnorm_median_score"),
    ("raw_mean_utility", "refnorm_mean_score"),
    ("d_best_utility", "refnorm_d_best_score"),
)
CANDIDATE_COLUMNS = tuple(column for pair in PAIRS[:3] for column in pair)


@pytest.fixture
def rows():
    frame = report_rows()
    frame["reference_min_utility"] = -4.0
    frame["reference_max_utility"] = -1.0
    for (raw, normalized), value in zip(PAIRS, (-1.6, -2.2, -2.6, -2.5), strict=True):
        frame[raw] = value
        frame[normalized] = reference_normalize(
            value, reference_low=-4.0, reference_high=-1.0
        )
    return frame


def _files(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("normalized", [pair[1] for pair in PAIRS])
@pytest.mark.parametrize(
    "existing", [False, True], ids=["new-output", "existing-output"]
)
def test_changed_normalized_value_is_rejected_before_any_output(
    rows, tmp_path, normalized, existing
):
    destination = tmp_path / "report"
    if existing:
        write_unified_report(rows, destination)
    before_files = _files(destination)
    rows.loc[0, normalized] = 999.0
    before_rows = rows.copy(deep=True)

    with pytest.raises((ValueError, TypeError)):
        write_unified_report(rows, destination)

    assert _files(destination) == before_files
    assert destination.exists() is existing
    pd.testing.assert_frame_equal(rows, before_rows, check_exact=True)


@pytest.mark.parametrize("raw,normalized", PAIRS)
def test_changed_raw_value_without_matching_normalization_is_rejected(
    rows, tmp_path, raw, normalized
):
    del normalized
    rows.loc[0, raw] += 0.5
    destination = tmp_path / "report"
    with pytest.raises((ValueError, TypeError)):
        write_unified_report(rows, destination)
    assert not destination.exists()


@pytest.mark.parametrize("normalized", [pair[1] for pair in PAIRS])
def test_clearly_wrong_but_finite_normalization_is_rejected(rows, tmp_path, normalized):
    rows.loc[0, normalized] += 0.1
    with pytest.raises((ValueError, TypeError)):
        write_unified_report(rows, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    "value", [-7.0, 2.0], ids=["below-reference", "above-reference"]
)
def test_legitimate_scores_outside_zero_one_are_preserved(rows, tmp_path, value):
    for raw, normalized in PAIRS[:3]:
        rows[raw] = value
        rows[normalized] = reference_normalize(
            value, reference_low=-4.0, reference_high=-1.0
        )
    expected = rows.loc[0, "refnorm_max_score"]
    assert expected < 0 or expected > 1
    result = write_unified_report(rows, tmp_path / "report")
    assert result.per_seed.loc[0, "refnorm_max_score"] == expected
    assert result.summary.loc[0, "refnorm_max_score_mean"] == expected


@pytest.mark.parametrize("width", [0.0, 5e-13], ids=["degenerate", "near-zero"])
def test_degenerate_reference_uses_existing_zero_normalization(rows, tmp_path, width):
    low, high = 1.0, 1.0 + width
    rows["reference_min_utility"] = float(low)
    rows["reference_max_utility"] = float(high)
    for index, (raw, normalized) in enumerate(PAIRS):
        rows[raw] = low if index == 3 else high + (3 - index)
        rows[normalized] = reference_normalize(
            rows.loc[0, raw], reference_low=low, reference_high=high
        )
        assert rows.loc[0, normalized] == 0.0
    result = write_unified_report(rows, tmp_path / "report")
    assert result.per_seed[[pair[1] for pair in PAIRS]].eq(0.0).all().all()


@pytest.mark.parametrize("normalized", [pair[1] for pair in PAIRS])
def test_degenerate_reference_does_not_allow_arbitrary_normalized_scores(
    rows, tmp_path, normalized
):
    rows["reference_min_utility"] = rows["reference_max_utility"] = -2.5
    rows["d_best_utility"] = -2.5
    for _, column in PAIRS:
        rows[column] = 0.0
    rows[normalized] = 1.0
    with pytest.raises((ValueError, TypeError)):
        write_unified_report(rows, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    "precision", [np.float32, np.float64], ids=["float32", "float64"]
)
def test_floating_point_rows_and_csv_round_trip_keep_valid_semantics(
    rows, tmp_path, precision
):
    low, high = precision(-4.23456789), precision(-1.12345678)
    rows["dtype"] = "torch." + np.dtype(precision).name
    rows["reference_min_utility"] = float(low)
    rows["reference_max_utility"] = float(high)
    for (raw, normalized), value in zip(
        PAIRS, (-1.61234567, -2.23456789, -2.64567891, -2.57891234), strict=True
    ):
        # Runner aggregates are Python floats even when their original source
        # was float32; do not add independent float32 CSV quantization here.
        rows[raw] = float(precision(value))
        rows[normalized] = reference_normalize(
            float(precision(value)),
            reference_low=float(low),
            reference_high=float(high),
        )
    before = rows.copy(deep=True)
    original_report = write_unified_report(rows, tmp_path / "direct")
    pd.testing.assert_frame_equal(rows, before, check_exact=True)

    source = tmp_path / "input.csv"
    rows.to_csv(source, index=False)
    before_bytes = source.read_bytes()
    loaded = pd.read_csv(source)
    loaded_before = loaded.copy(deep=True)
    round_trip_report = write_unified_report(loaded, tmp_path / "round-trip")
    assert source.read_bytes() == before_bytes
    pd.testing.assert_frame_equal(loaded, loaded_before, check_exact=True)
    np.testing.assert_allclose(
        round_trip_report.summary["refnorm_max_score_mean"],
        original_report.summary["refnorm_max_score_mean"],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "high",
    [np.nextafter(1.0 + 1e-12, 1.0), 1.0 + 1e-12, 1.0 + 2e-12],
    ids=["below-threshold", "above-threshold", "narrow-nondegenerate"],
)
def test_valid_near_threshold_csv_uses_input_precision_not_tiny_score_tolerance(
    rows, tmp_path, high
):
    # Cancellation magnifies normal CSV input rounding. A fixed 1e-12 absolute
    # score tolerance would reject legitimate rows near this threshold.
    low = 1.0
    rows["dtype"] = "torch.float64"
    rows["reference_min_utility"] = low
    rows["reference_max_utility"] = high
    for (raw, normalized), fraction in zip(PAIRS, (0.8, 0.55, 0.45, 0.3), strict=True):
        value = low + fraction * (high - low)
        rows[raw] = value
        rows[normalized] = reference_normalize(
            value, reference_low=low, reference_high=high
        )
    source = tmp_path / "near-threshold.csv"
    rows.to_csv(source, index=False)
    original = source.read_bytes()
    loaded = pd.read_csv(source)
    report = write_unified_report(loaded, tmp_path / "report")
    assert report.summary.loc[0, "successful_runs"] == 1
    assert source.read_bytes() == original
    for _, normalized in PAIRS:
        corrupted = loaded.copy(deep=True)
        corrupted[normalized] = 999.0
        output = tmp_path / f"corrupted-{normalized}"
        with pytest.raises((ValueError, TypeError)):
            write_unified_report(corrupted, output)
        assert not output.exists()


def test_failed_rows_keep_null_scores_and_success_rows_remain_rankable(rows, tmp_path):
    rows["required_seeds_json"] = "[38,39]"
    failed = rows.copy()
    failed["method_seed"] = int(rows.loc[0, "method_seed"]) + 1
    failed["status"] = "failed"
    failed["error_type"] = "RuntimeError"
    failed["error_message"] = "retained failure"
    failed[list(CANDIDATE_COLUMNS)] = np.nan
    combined = pd.concat([rows, failed], ignore_index=True)
    before = combined.copy(deep=True)
    result = write_unified_report(combined, tmp_path / "report")
    assert len(result.per_seed) == 2
    assert result.summary.loc[0, "successful_runs"] == 1
    assert result.summary.loc[0, "failed_runs"] == 1
    saved_failure = result.per_seed[result.per_seed["status"] == "failed"]
    assert saved_failure[list(CANDIDATE_COLUMNS)].isna().all().all()
    pd.testing.assert_frame_equal(combined, before, check_exact=True)


def test_valid_modern_partial_formal_report_remains_unranked(rows, tmp_path):
    rows["result_source"] = "unified_runner"
    rows["phase"] = "formal"
    rows["required_seeds_json"] = json.dumps(list(range(38, 46)))
    rows["method_seed"] = 38
    result = write_unified_report(rows, tmp_path / "report")
    assert result.summary.loc[0, "successful_runs"] == 1
    assert result.summary.loc[0, "missing_runs"] == 7
    assert not result.summary.loc[0, "rank_eligible"]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "invalid",
    [
        "missing-phase",
        "missing-required",
        "short-formal",
        "bool-seed",
        "fractional-seed",
    ],
)
def test_invalid_modern_seed_contract_is_rejected_before_report_writes(
    rows, tmp_path, existing, invalid
):
    rows["result_source"] = "unified_runner"
    rows["phase"] = "formal"
    rows["required_seeds_json"] = json.dumps(list(range(38, 46)))
    rows["method_seed"] = 38
    destination = tmp_path / "report"
    if existing:
        write_unified_report(rows, destination)
    before_files = _files(destination)
    if invalid == "missing-phase":
        rows = rows.drop(columns="phase")
    elif invalid == "missing-required":
        rows = rows.drop(columns="required_seeds_json")
    elif invalid == "short-formal":
        rows["required_seeds_json"] = "[38]"
    else:
        rows["method_seed"] = True if invalid == "bool-seed" else 38.9
    before_rows = rows.copy(deep=True)
    with pytest.raises(ValueError):
        write_unified_report(rows, destination)
    assert _files(destination) == before_files
    assert destination.exists() is existing
    pd.testing.assert_frame_equal(rows, before_rows, check_exact=True)


def test_numeric_error_identifies_task_run_seed_and_field(rows, tmp_path):
    rows["refnorm_max_score"] = 999.0
    with pytest.raises(ValueError) as caught:
        write_unified_report(rows, tmp_path / "report")
    message = str(caught.value)
    for field in ("task_id", "run_id", "method_seed"):
        assert str(rows.loc[0, field]) in message
    assert "refnorm_max_score" in message


@pytest.mark.parametrize(
    "column",
    [
        *(column for pair in PAIRS for column in pair),
        "reference_min_utility",
        "reference_max_utility",
    ],
)
def test_missing_semantic_column_is_rejected_before_output(rows, tmp_path, column):
    with pytest.raises(KeyError):
        write_unified_report(rows.drop(columns=column), tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    "column",
    [
        "raw_max_utility",
        "refnorm_mean_score",
        "d_best_utility",
        "reference_min_utility",
    ],
)
@pytest.mark.parametrize("value", [True, "not-numeric", np.inf, -np.inf, np.nan])
def test_invalid_semantic_numbers_are_rejected(rows, tmp_path, column, value):
    # Object dtype avoids pandas rejecting the fixture assignment before the
    # public report validator has a chance to see the invalid value.
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = value
    with pytest.raises((ValueError, TypeError)):
        write_unified_report(rows, tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    "column",
    [
        "raw_max_utility",
        "refnorm_mean_score",
        "d_best_utility",
        "reference_min_utility",
    ],
)
@pytest.mark.parametrize("value", [False, np.bool_(False)], ids=["bool", "numpy-bool"])
def test_boolean_is_not_accepted_even_when_its_numeric_value_would_match(
    rows, tmp_path, column, value
):
    rows["reference_min_utility"] = 0.0
    rows["reference_max_utility"] = 2.0
    for raw, normalized in PAIRS:
        rows[raw] = 0.0
        rows[normalized] = 0.0
    rows[column] = rows[column].astype(object)
    rows.loc[0, column] = value
    with pytest.raises((ValueError, TypeError)):
        write_unified_report(rows, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_reporting_does_not_construct_a_method_or_query_an_oracle(
    rows, tmp_path, monkeypatch
):
    from llm_design_bench.optimizers import registry
    from llm_design_bench.tasks.data_recipes import DataRecipesTask

    def forbidden(*args, **kwargs):
        pytest.fail("report regeneration must not run methods or query the oracle")

    monkeypatch.setattr(registry, "make_method", forbidden)
    monkeypatch.setattr(DataRecipesTask, "predict", forbidden)
    before = rows.copy(deep=True)
    result = write_unified_report(rows, tmp_path / "report")
    assert result.summary.loc[0, "successful_runs"] == 1
    pd.testing.assert_frame_equal(rows, before, check_exact=True)
