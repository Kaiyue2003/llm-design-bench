"""Offline integrity checks for immutable published experiment inputs."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "experiments" / "llmdm_forward_v1"
RELEASE = json.loads((ASSETS / "release.json").read_text(encoding="utf-8"))


def _digest(value):
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("relative", RELEASE["artifact_sha256"])
def test_release_artifacts_preserve_exact_published_bytes(relative):
    path = ASSETS / relative
    assert path.is_file()
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == RELEASE["artifact_sha256"][relative]
    )


def test_release_plan_is_self_consistent_and_complete():
    plan = json.loads((ASSETS / "plan.json").read_text(encoding="utf-8"))
    envelope = json.loads((ASSETS / "data/manifest.json").read_text(encoding="utf-8"))
    assert plan.pop("plan_id") == RELEASE["plan_id"] == _digest(plan)
    assert (
        plan["data_manifest_id"]
        == envelope["manifest_id"]
        == RELEASE["data_manifest_id"]
    )
    assert plan["experiment_id"] == RELEASE["experiment_id"]
    assert plan["package_source"]["git_dirty"] is False
    assert plan["package_source"]["git_commit"] == RELEASE["code_commit"]
    assert plan["package_source"]["sha256"] == RELEASE["package_source_sha256"]
    assert envelope["metadata"]["upstream_revision"] == RELEASE["upstream_commit"]
    for key in ("code_commit", "upstream_commit"):
        assert re.fullmatch("[0-9a-f]{40}", RELEASE[key])
    for filename, checksum in envelope["artifacts"].items():
        assert RELEASE["artifact_sha256"][f"data/{filename}"] == checksum
    settings = plan["shared_settings"]
    assert settings["protocol_id"] == RELEASE["protocol_id"]
    assert settings["candidate_budget"] == 128
    assert settings["pilot_seeds"] == [0]
    assert settings["formal_seeds"] == list(range(38, 46))
    assert settings["utility_transform"] == "negative_loss"
    assert settings["utility_percentiles"] == [0.0, 40.0]
    assert settings["mixed_precision"] is False
    requested = json.loads(
        (ROOT / "configs/llmdm_methods.forward_v1.json").read_text(encoding="utf-8")
    )
    assert len(plan["methods"]) == RELEASE["counts"]["methods"] == 19
    assert [entry["method_id"] for entry in plan["methods"]] == [
        entry["method_id"] for entry in requested
    ]
    for entry, original in zip(plan["methods"], requested, strict=True):
        assert entry["requested_kwargs"] == original["kwargs"]
        for name, value in original["kwargs"].items():
            assert entry["kwargs"][name] == value
        expected_dtype = (
            "float64"
            if entry["method_id"] in {"bdi", "bo_qei", "ga_on_gp"}
            else "float32"
        )
        assert entry["dtype"] == expected_dtype


def test_release_visible_rows_are_exact_low_utility_subset():
    with np.load(ASSETS / "data/visible.npz", allow_pickle=False) as archive:
        visible = dict(archive)
    with np.load(ASSETS / "data/reference.npz", allow_pickle=False) as archive:
        reference = dict(archive)
    assert len(reference["row_ids"]) == RELEASE["counts"]["usable_logged"] == 454
    assert len(visible["row_ids"]) == RELEASE["counts"]["main_visible"] == 184
    mask = np.zeros(454, dtype=bool)
    for scale in np.unique(reference["context"][:, 0]):
        group = reference["context"][:, 0] == scale
        utility = reference["utility"]
        lower, upper = np.percentile(utility[group], [0, 40], method="linear")
        mask |= group & (utility >= lower) & (utility <= upper)
    for key in ("row_ids", "mixtures", "context", "utility"):
        np.testing.assert_array_equal(visible[key], reference[key][mask])
    for arrays in (visible, reference):
        n = len(arrays["row_ids"])
        assert arrays["row_ids"].dtype == np.dtype("int64")
        assert np.all(np.diff(arrays["row_ids"]) > 0)
        assert arrays["mixtures"].shape == (n, 5)
        assert arrays["context"].shape == (n, 2)
        for key in ("mixtures", "context", "utility"):
            assert arrays[key].dtype == np.dtype("float64")
            assert np.isfinite(arrays[key]).all()
        assert (arrays["mixtures"] >= 0).all()
        np.testing.assert_allclose(arrays["mixtures"].sum(axis=1), 1, atol=1e-8)
        assert (arrays["utility"] < 0).all()
    one_b = visible["context"][:, 0] == 1000
    assert int(one_b.sum()) == RELEASE["counts"]["fixed_1b_visible"] == 26
    np.testing.assert_array_equal(
        visible["row_ids"][one_b],
        [
            54,
            61,
            82,
            83,
            87,
            88,
            117,
            122,
            192,
            210,
            211,
            212,
            349,
            362,
            363,
            365,
            380,
            400,
            402,
            403,
            424,
            425,
            426,
            429,
            437,
            438,
        ],
    )
