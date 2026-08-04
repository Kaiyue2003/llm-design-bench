import numpy as np
import pandas as pd


def usefulness_summary(
    values: np.ndarray,
    reference_values: np.ndarray,
    oracle_best: float,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    if values.size == 0 or reference_values.size == 0:
        raise ValueError("utility arrays must not be empty")
    low = float(reference_values.min())
    high = float(reference_values.max())
    width = high - low
    min_width = 1e-12 * max(1.0, abs(low), abs(high))

    def reference_normalized(value: float) -> float:
        return 0.0 if width <= min_width else (value - low) / width

    best = float(values.max())
    median = float(np.median(values))
    mean = float(values.mean())
    return {
        "raw_max_utility": best,
        "raw_median_utility": median,
        "raw_mean_utility": mean,
        "refnorm_max_score": reference_normalized(best),
        "refnorm_median_score": reference_normalized(median),
        "refnorm_mean_score": reference_normalized(mean),
        "regret": float(oracle_best - best),
    }


def add_reference_normalized_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with reference-normalized score columns.

    The ``refnorm_*`` columns are computed against the reference dataset passed
    to ``usefulness_summary``. This function is intentionally a light schema
    guard for report generation rather than a second normalization pass:
    report-local min-max normalization can make a method's max score appear
    lower than its median score, even though the raw max utility is larger.
    """
    frame = frame.copy()
    required = {
        "refnorm_max_score",
        "refnorm_median_score",
        "refnorm_mean_score",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"missing reference-normalized score columns: {missing}")
    return frame
