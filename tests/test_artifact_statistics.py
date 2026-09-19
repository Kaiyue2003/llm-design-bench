"""Read-only checks of reported statistics against synthetic raw artifacts."""

import json

import numpy as np
import pytest

from llm_design_bench.evaluation.run_artifacts import (
    file_sha256,
    stable_fingerprint,
    verify_successful_attempt,
)


STATISTIC_FIELDS = (
    "raw_min_loss",
    "raw_median_loss",
    "raw_mean_loss",
    "unique_candidate_count",
    "unique_candidate_fraction",
)


def _success(directory, *, dtype=np.float64, layout="unique", negative_loss=True):
    """Write a complete attempt without invoking any method, oracle, or runner."""
    directory.mkdir()
    candidates = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.25, 0.25, 0.5]], dtype=dtype
    )
    unique_count = 4
    if layout == "duplicate":
        candidates[:] = [0.25, 0.25, 0.5]
        unique_count = 1
    elif layout == "rounding":
        # All four are distinct in float64, but the first pair collides at
        # 12 decimal places. In float32, all four collapse during conversion.
        candidates = np.array(
            [
                [0.25 + delta, 0.25 - delta, 0.5]
                for delta in (0.0, 2e-13, 2e-12, -2e-12)
            ],
            dtype=dtype,
        )
        unique_count = 3 if dtype == np.float64 else 1
    elif layout == "float32_rounding":
        candidates = np.array(
            [[delta, 0.5, 0.5] for delta in (1.5e-12, 2e-12, 1.5e-12, 2e-12)],
            dtype=np.float32,
        )
        unique_count = 1
    target = np.array([1000, 19500], dtype=dtype)
    utility = np.array([-1.25, -2.5, -4.0, -5.25])
    score = (utility + 6.0) / 6.0
    logical = {
        "candidate_budget": len(candidates),
        "design_space": {"dimension": candidates.shape[1]},
        "dtype": f"torch.{np.dtype(dtype).name}",
        "problem_metadata_json": json.dumps(
            {"utility_transform": "negative_loss" if negative_loss else "identity"}
        ),
    }
    fingerprint = stable_fingerprint(logical)
    manifest = {"logical_config": logical, "logical_fingerprint": fingerprint}
    np.savez(directory / "candidates.npz", candidates=candidates, target_context=target)
    evaluation = {"utility": utility, "refnorm_score": score}
    if negative_loss:
        evaluation["raw_loss"] = -utility
    np.savez(directory / "evaluation.npz", **evaluation)
    row = {
        **logical,
        "status": "success",
        "logical_fingerprint": fingerprint,
        "artifact_sha256": {
            name: file_sha256(directory / name)
            for name in ("candidates.npz", "evaluation.npz")
        },
        "target_context_json": json.dumps(target.tolist()),
        "reference_min_utility": -6.0,
        "reference_max_utility": 0.0,
        "unique_candidate_count": unique_count,
        "unique_candidate_fraction": unique_count / len(candidates),
    }
    for prefix, values, suffix in (
        ("raw", utility, "utility"),
        ("refnorm", score, "score"),
    ):
        for name, reduce in (("max", np.max), ("median", np.median), ("mean", np.mean)):
            row[f"{prefix}_{name}_{suffix}"] = float(reduce(values))
    if negative_loss:
        row.update(raw_min_loss=1.25, raw_median_loss=3.25, raw_mean_loss=3.25)
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_row(directory, row)
    return manifest, row


def _write_row(directory, row):
    (directory / "result.json").write_text(json.dumps(row), encoding="utf-8")


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["unique", "duplicate", "rounding"])
def test_correct_statistics_are_accepted_without_writes(tmp_path, dtype, layout):
    directory = tmp_path / "attempt"
    expected = _success(directory, dtype=dtype, layout=layout)
    before = _snapshot(directory)

    assert verify_successful_attempt(directory) == expected
    assert _snapshot(directory) == before


@pytest.mark.parametrize("field", STATISTIC_FIELDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_changed_statistics_are_rejected_without_writes(tmp_path, field, dtype):
    directory = tmp_path / "attempt"
    _, row = _success(directory, dtype=dtype)
    row[field] += 1
    _write_row(directory, row)
    before = _snapshot(directory)

    with pytest.raises(ValueError, match=field):
        verify_successful_attempt(directory)
    assert _snapshot(directory) == before


@pytest.mark.parametrize("field", STATISTIC_FIELDS)
@pytest.mark.parametrize(
    "invalid",
    [None, float("nan"), float("inf"), float("-inf"), True, False, "numeric-string"],
    ids=["null", "nan", "positive-inf", "negative-inf", "true", "false", "string"],
)
def test_invalid_statistic_types_are_rejected(tmp_path, field, invalid):
    directory = tmp_path / "attempt"
    _, row = _success(directory)
    row[field] = str(row[field]) if invalid == "numeric-string" else invalid
    _write_row(directory, row)

    with pytest.raises(ValueError, match=field):
        verify_successful_attempt(directory)


@pytest.mark.parametrize("field", STATISTIC_FIELDS)
def test_missing_statistics_are_rejected_with_the_field_name(tmp_path, field):
    directory = tmp_path / "attempt"
    _, row = _success(directory)
    del row[field]
    _write_row(directory, row)

    with pytest.raises(ValueError, match=field):
        verify_successful_attempt(directory)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_non_loss_task_does_not_require_loss_statistics(tmp_path, dtype):
    directory = tmp_path / "attempt"
    expected = _success(directory, dtype=dtype, negative_loss=False)
    assert not any(
        key.startswith("raw_") and key.endswith("loss") for key in expected[1]
    )

    assert verify_successful_attempt(directory) == expected


@pytest.mark.parametrize(
    "field", ["unique_candidate_count", "unique_candidate_fraction"]
)
def test_non_loss_task_still_requires_candidate_statistics(tmp_path, field):
    directory = tmp_path / "attempt"
    _, row = _success(directory, negative_loss=False)
    del row[field]
    _write_row(directory, row)

    with pytest.raises(ValueError, match=field):
        verify_successful_attempt(directory)


@pytest.mark.parametrize(
    "field", ["unique_candidate_count", "unique_candidate_fraction"]
)
def test_candidate_statistics_use_twelve_decimal_rounding(tmp_path, field):
    directory = tmp_path / "attempt"
    _, row = _success(directory, layout="rounding")
    with np.load(directory / "candidates.npz", allow_pickle=False) as archive:
        assert len(np.unique(archive["candidates"], axis=0)) == 4
    # Exact-value uniqueness would incorrectly report four candidates / 100%.
    row[field] = 4 if field == "unique_candidate_count" else 1.0
    _write_row(directory, row)

    with pytest.raises(ValueError, match=field):
        verify_successful_attempt(directory)


def test_candidate_rounding_preserves_the_saved_float32_dtype(tmp_path):
    directory = tmp_path / "attempt"
    expected = _success(directory, dtype=np.float32, layout="float32_rounding")
    with np.load(directory / "candidates.npz", allow_pickle=False) as archive:
        candidates = archive["candidates"]
    # Casting before rounding changes the tie handling and would count two.
    assert len(np.unique(np.round(candidates, 12), axis=0)) == 1
    assert len(np.unique(np.round(candidates.astype(np.float64), 12), axis=0)) == 2

    assert verify_successful_attempt(directory) == expected


@pytest.mark.parametrize("invalid", [4.5, 4.0 - 5e-13, 4.0 + 5e-13])
def test_candidate_count_requires_an_exact_integer_value(tmp_path, invalid):
    directory = tmp_path / "attempt"
    _, row = _success(directory)
    row["unique_candidate_count"] = invalid
    _write_row(directory, row)

    with pytest.raises(ValueError, match="unique_candidate_count"):
        verify_successful_attempt(directory)


def test_integer_valued_float_candidate_count_is_accepted(tmp_path):
    directory = tmp_path / "attempt"
    manifest, row = _success(directory)
    row["unique_candidate_count"] = 4.0
    _write_row(directory, row)

    assert verify_successful_attempt(directory) == (manifest, row)


def test_candidate_fraction_cannot_exceed_one_within_float_tolerance(tmp_path):
    directory = tmp_path / "attempt"
    _, row = _success(directory)
    row["unique_candidate_fraction"] = 1.0 + 5e-13
    _write_row(directory, row)

    with pytest.raises(ValueError, match="unique_candidate_fraction"):
        verify_successful_attempt(directory)
