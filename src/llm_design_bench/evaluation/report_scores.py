"""Read-only scalar consistency checks for imported report rows.

CSV rounding is not evidence of a different result. Validate the normalization
relation in utility space, propagating the small parsing errors of each input,
instead of dividing those errors by a possibly tiny reference interval. This
does not authenticate a CSV or replace the per-attempt artifact verifier.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import cast

import numpy as np
import pandas as pd


_SCORE_PAIRS = (
    ("raw_max_utility", "refnorm_max_score"),
    ("raw_median_utility", "refnorm_median_score"),
    ("raw_mean_utility", "refnorm_mean_score"),
)
_NORMALIZATION_TOLERANCE = 1e-12


def validate_report_scores(per_seed: pd.DataFrame) -> None:
    """Reject inconsistent scores without altering values or requiring NPZs."""
    rows = cast(list[dict[str, object]], per_seed.to_dict(orient="records"))
    for row in rows:
        low = _finite_number(row, "reference_min_utility")
        high = _finite_number(row, "reference_max_utility")
        if low > high or not math.isfinite(high - low):
            raise ValueError(
                f"{_identity(row)}: reference_min_utility must not exceed "
                "reference_max_utility and reference width must be finite"
            )
        _validate_pair(row, "d_best_utility", "refnorm_d_best_score", low, high)
        for raw_field, score_field in _SCORE_PAIRS:
            if row["status"] == "success":
                _validate_pair(row, raw_field, score_field, low, high)
            else:
                for field in (raw_field, score_field):
                    if not _missing_score(row[field]):
                        raise ValueError(
                            f"{_identity(row)}: failed rows must not contain "
                            f"utility scores ({field})"
                        )


def _identity(row: Mapping[str, object]) -> str:
    return (
        f"task={row.get('task_id')!r}, run={row.get('run_id')!r}, "
        f"seed={row.get('method_seed')!r}"
    )


def _finite_number(row: Mapping[str, object], field: str) -> float:
    value = row[field]
    if not isinstance(value, (bool, np.bool_)) and isinstance(value, (Real, str)):
        try:
            number = float(value)
        except (ValueError, OverflowError):
            pass
        else:
            if math.isfinite(number):
                return number
    raise ValueError(f"{_identity(row)}: {field} must be a finite non-boolean number")


def _missing_score(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or isinstance(value, (float, np.floating))
        and math.isnan(float(value))
    )


def _validate_pair(
    row: Mapping[str, object],
    raw_field: str,
    score_field: str,
    low: float,
    high: float,
) -> None:
    raw = _finite_number(row, raw_field)
    score = _finite_number(row, score_field)
    if not _normalization_matches(raw, score, low, high):
        raise ValueError(
            f"{_identity(row)}: {score_field} is inconsistent with {raw_field} "
            "and the row's normalization reference"
        )


def _csv_rounding_bound(value: float) -> float:
    # Pandas' default decimal parser can lose a few absolute ULPs even near
    # zero. The unit floor also covers decimal cancellation around zero.
    return 4.0 * math.ulp(max(1.0, abs(value)))


def _normalization_matches(raw: float, score: float, low: float, high: float) -> bool:
    width = high - low
    # Same threshold as seed_statistics.reference_normalize; no new formula.
    threshold = _NORMALIZATION_TOLERANCE * max(1.0, abs(low), abs(high))
    low_error = _csv_rounding_bound(low)
    high_error = _csv_rounding_bound(high)
    boundary_error = (
        low_error
        + high_error
        + math.ulp(width)
        + _NORMALIZATION_TOLERANCE * max(low_error, high_error)
        + math.ulp(threshold)
    )
    boundary_distance = width - threshold
    # CSV rounding may move an interval across the existing zero-width cutoff.
    # Permit either interpretation only inside this narrow round-off band.
    if boundary_distance <= boundary_error and abs(score) <= _NORMALIZATION_TOLERANCE:
        return True
    if boundary_distance < -boundary_error:
        return False

    offset = raw - low
    scaled_score = score * width
    residual = offset - scaled_score
    error_bound = (
        _csv_rounding_bound(raw)
        + abs(1.0 - score) * low_error
        + abs(score) * high_error
        + width * _NORMALIZATION_TOLERANCE * (1.0 + abs(score))
        + math.ulp(offset)
        + math.ulp(scaled_score)
        + math.ulp(residual)
    )
    # Overflow must fail closed; inf <= inf is not a valid consistency check.
    return (
        all(
            math.isfinite(value)
            for value in (offset, scaled_score, residual, error_bound)
        )
        and abs(residual) <= error_bound
    )
