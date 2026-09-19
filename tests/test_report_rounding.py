"""CSV normalization checks tolerate rounding, not changed experiment scores."""

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from llm_design_bench.evaluation.report_scores import (
    _normalization_matches,
    validate_report_scores,
)
from llm_design_bench.evaluation.seed_statistics import reference_normalize


PAIRS = (
    ("raw_max_utility", "refnorm_max_score"),
    ("raw_median_utility", "refnorm_median_score"),
    ("raw_mean_utility", "refnorm_mean_score"),
    ("d_best_utility", "refnorm_d_best_score"),
)


def _frame(raw, score, low, high):
    frame = pd.DataFrame(
        {
            "task_id": "rounding-probe",
            "run_id": "method",
            "method_seed": np.arange(len(raw)),
            "status": "success",
            "reference_min_utility": low,
            "reference_max_utility": high,
        }
    )
    for raw_field, score_field in PAIRS:
        frame[raw_field] = raw
        frame[score_field] = score
    return frame


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8, 1e100])
@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 2.0, 1e6])
def test_csv_rounding_across_scales_and_reference_widths(scale, factor):
    rng = np.random.default_rng(42)
    low = rng.uniform(-3.0, 3.0, 200) * scale
    high = low + factor * 1e-12 * np.maximum(1.0, np.abs(low))
    if factor == 1.0:
        high[::3] = np.nextafter(high[::3], np.inf)
        high[1::3] = np.nextafter(high[1::3], -np.inf)
    raw = low + (high - low) * rng.uniform(-2.0, 4.0, len(low))
    score = [
        reference_normalize(
            float(value), reference_low=float(lo), reference_high=float(hi)
        )
        for value, lo, hi in zip(raw, low, high, strict=True)
    ]
    loaded = pd.read_csv(StringIO(_frame(raw, score, low, high).to_csv(index=False)))
    before = loaded.copy(deep=True)

    validate_report_scores(loaded)

    pd.testing.assert_frame_equal(loaded, before, check_exact=True)
    # Test each input independently: one rejected row must not hide a looser
    # acceptance policy for other near-threshold intervals in the same frame.
    for row in loaded.to_dict(orient="records"):
        assert not _normalization_matches(
            row["raw_max_utility"],
            999.0,
            row["reference_min_utility"],
            row["reference_max_utility"],
        )


@pytest.mark.parametrize(
    "low,high,raw,direction",
    [
        (1.7163858316617233, 1.7163858316634395, 1.7163858316683753, -np.inf),
        (2.5605899330916113, 2.560589933094172, 2.560589933094313, np.inf),
    ],
    ids=["degenerate-to-nondegenerate", "nondegenerate-to-degenerate"],
)
def test_one_ulp_endpoint_shift_across_threshold_keeps_the_original_score(
    low, high, raw, direction
):
    score = reference_normalize(raw, reference_low=low, reference_high=high)
    rounded_low = float(np.nextafter(low, direction))
    rounded_score = reference_normalize(
        raw, reference_low=rounded_low, reference_high=high
    )
    assert (score == 0.0) != (rounded_score == 0.0)
    frame = _frame([raw], [score], [rounded_low], [high])
    before = frame.copy(deep=True)

    validate_report_scores(frame)

    pd.testing.assert_frame_equal(frame, before, check_exact=True)


@pytest.mark.parametrize(
    "raw,score,low,high",
    [
        (0.0, 0.0, -1e308, 1e308),  # Reference width overflows.
        (1e308, 1e308, 0.0, 1e308),  # Both residual and tolerance overflow.
        (1e308, 1.0, -1e308, 0.0),  # The raw offset overflows.
    ],
)
def test_finite_numbers_with_overflowing_consistency_arithmetic_fail_closed(
    raw, score, low, high
):
    frame = _frame([raw], [score], [low], [high])
    with pytest.raises(ValueError):
        validate_report_scores(frame)


@pytest.mark.parametrize("missing", [None, pd.NA, np.nan, np.float32(np.nan)])
def test_failed_scores_accept_missing_values_without_changing_them(missing):
    frame = _frame([0.0], [0.0], [0.0], [1.0])
    frame["status"] = "failed"
    for raw_field, score_field in PAIRS[:3]:
        frame[raw_field] = pd.Series([missing], dtype=object)
        frame[score_field] = pd.Series([missing], dtype=object)
    before = frame.copy(deep=True)

    validate_report_scores(frame)

    pd.testing.assert_frame_equal(frame, before, check_exact=True)


@pytest.mark.parametrize("value", [False, np.bool_(False), 0.0, "nan", "", np.inf])
def test_failed_scores_do_not_treat_zero_boolean_or_strings_as_missing(value):
    frame = _frame([0.0], [0.0], [0.0], [1.0])
    frame["status"] = "failed"
    for raw_field, score_field in PAIRS[:3]:
        frame[raw_field] = None
        frame[score_field] = None
    frame["refnorm_mean_score"] = pd.Series([value], dtype=object)

    with pytest.raises(ValueError, match="failed rows must not contain utility scores"):
        validate_report_scores(frame)
