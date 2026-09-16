"""Resume must verify the exported CSV against its verified result artifact."""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from test_colab_batch import (
    _job,
    _save_job,
    batch,
    plan as plan,
    runner as runner,
)


def _csv_path(runner):
    return runner.state / "pilot" / "method_seed_results.csv"


def _read_csv(runner):
    with _csv_path(runner).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames), rows


def _write_csv(runner, fields, rows):
    with _csv_path(runner).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _snapshot(runner):
    """Include files and directories in both local state and backup storage."""
    return {
        path.relative_to(runner.state.parent).as_posix(): (
            path.read_bytes() if path.is_file() else None
        )
        for path in runner.state.parent.rglob("*")
    }


def _assert_blocked_without_writes(runner, monkeypatch, field, *, run_id="offline_mlp"):
    def forbidden(*args, **kwargs):
        pytest.fail("CSV disagreement must never dispatch or retrain")

    monkeypatch.setattr(batch, "run_job", forbidden)
    before = _snapshot(runner)
    preview = runner.preview(methods=[run_id])
    assert preview[0]["status"] == "blocked"
    assert "CSV" in preview[0]["error"]
    assert field in preview[0]["error"]
    report = runner.pilot_report(methods=[run_id])
    assert report[0]["status"] == "blocked"
    assert "CSV" in report[0]["error"]
    assert field in report[0]["error"]
    with pytest.raises(RuntimeError, match="CSV"):
        runner.run(methods=[run_id])
    assert _snapshot(runner) == before


def _save_extended_result(runner, *, run_id="offline_mlp"):
    """Add nullable metadata and a relocatable absolute artifact path."""
    directory, row = _save_job(runner, _job(runner, run_id=run_id))
    additions = {
        "artifact_dir": "/content/prior-runtime/formal/runs/old-absolute-path",
        "dataset_seed": None,
        "split_seed": None,
        "error_message": None,
    }
    row.update(additions)
    (directory / "result.json").write_text(
        json.dumps(row, sort_keys=True), encoding="utf-8"
    )
    fields, rows = _read_csv(runner)
    fields.extend(additions)
    rows[0].update(additions)
    _write_csv(runner, fields, rows)
    return directory, row


def _assert_resumes_without_writes(runner, monkeypatch, *, run_id="offline_mlp"):
    def forbidden(*args, **kwargs):
        pytest.fail("matching completed results must not be dispatched again")

    monkeypatch.setattr(batch, "run_job", forbidden)
    before = _snapshot(runner)
    assert runner.preview(methods=[run_id])[0]["status"] == "complete"
    assert runner.pilot_report(methods=[run_id])[0]["status"] == "verified"
    assert runner.run(methods=[run_id])[0]["status"] == "skipped"
    assert _snapshot(runner) == before


def test_changed_csv_score_blocks_resume_without_changing_artifacts(
    runner, monkeypatch
):
    _save_job(runner, _job(runner))
    fields, rows = _read_csv(runner)
    rows[0]["refnorm_max_score"] = "12345"
    _write_csv(runner, fields, rows)
    _assert_blocked_without_writes(runner, monkeypatch, "refnorm_max_score")


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("raw_mean_utility", "-2000"),
        ("raw_min_loss", "0.123"),
        ("candidate_budget", "64"),
        ("train_size", "26"),
        ("unique_candidate_count", "99999"),
        ("unique_candidate_fraction", "2"),
        ("method_config_json", '{"epochs": 1}'),
        ("problem_metadata_json", '{"utility_transform": "identity"}'),
        ("provenance_json", '{"plan_id": "wrong-plan"}'),
        ("training_summary_json", '{"final_standardized_mse": 123.0}'),
        ("diagnostics_json", '{"changed": true}'),
        ("target_context_json", "[1000.0, 1.0]"),
        ("method_seconds", "20000.0"),
        ("evaluation_seconds", "0"),
        ("total_seconds", "999"),
        ("peak_gpu_memory_bytes", "1025"),
        ("reference_min_utility", "-3000"),
        ("logical_fingerprint", "0" * 64),
        ("device", "cpu"),
        ("dtype", "torch.float64"),
        ("dataset_seed", "0"),
        ("error_message", "invented error"),
        ("artifact_sha256", "{}"),
        ("design_space", "{'dimension': 999}"),
    ],
)
def test_changed_csv_values_are_rejected(runner, monkeypatch, field, changed):
    _save_extended_result(runner)
    fields, rows = _read_csv(runner)
    rows[0][field] = changed
    _write_csv(runner, fields, rows)
    _assert_blocked_without_writes(runner, monkeypatch, field)


@pytest.mark.parametrize(
    "changed", ["NaN", "Infinity", "-Infinity", "", "None", "not-a-number"]
)
def test_invalid_numeric_csv_values_are_rejected(runner, monkeypatch, changed):
    _save_job(runner, _job(runner))
    fields, rows = _read_csv(runner)
    rows[0]["refnorm_max_score"] = changed
    _write_csv(runner, fields, rows)
    _assert_blocked_without_writes(runner, monkeypatch, "refnorm_max_score")


def test_original_csv_serialization_resumes_without_dispatch(runner, monkeypatch):
    _save_extended_result(runner)
    _assert_resumes_without_writes(runner, monkeypatch)


def test_semantically_equal_csv_json_is_accepted(runner, monkeypatch):
    _, artifact_row = _save_extended_result(runner)
    fields, rows = _read_csv(runner)
    for field, value in artifact_row.items():
        if field.endswith("_json"):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                parsed = dict(reversed(list(parsed.items())))
            rows[0][field] = json.dumps(parsed, indent=2)
    _write_csv(runner, fields, rows)
    _assert_resumes_without_writes(runner, monkeypatch)


@pytest.mark.parametrize("serializer", [repr, json.dumps], ids=["legacy-repr", "json"])
def test_artifact_hash_dictionary_formats_are_accepted(runner, monkeypatch, serializer):
    _, artifact_row = _save_extended_result(runner)
    fields, rows = _read_csv(runner)
    rows[0]["artifact_sha256"] = serializer(
        dict(reversed(list(artifact_row["artifact_sha256"].items())))
    )
    _write_csv(runner, fields, rows)
    _assert_resumes_without_writes(runner, monkeypatch)


def test_equivalent_numeric_csv_formats_are_accepted(runner, monkeypatch):
    _save_extended_result(runner)
    fields, rows = _read_csv(runner)
    rows[0].update(
        {
            "candidate_budget": "128.0",
            "train_size": "1.84e2",
            "refnorm_max_score": "5e-1",
            "raw_mean_utility": "-2.0000000000000000",
            "method_seconds": "2",
            "unique_candidate_count": "1.0",
            "peak_gpu_memory_bytes": "1.024e3",
        }
    )
    _write_csv(runner, fields, rows)
    _assert_resumes_without_writes(runner, monkeypatch)


@pytest.mark.parametrize("seed", ["0.0", "0e0"])
def test_equivalent_integer_seed_format_is_accepted(runner, monkeypatch, seed):
    _save_job(runner, _job(runner))
    fields, rows = _read_csv(runner)
    rows[0]["method_seed"] = seed
    _write_csv(runner, fields, rows)
    _assert_resumes_without_writes(runner, monkeypatch)


def test_null_values_exported_as_empty_fields_are_accepted(runner, monkeypatch):
    _save_extended_result(runner, run_id="bdi")
    _, rows = _read_csv(runner)
    for field in (
        "dataset_seed",
        "split_seed",
        "error_message",
        "peak_gpu_memory_bytes",
    ):
        assert rows[0][field] == ""
    _assert_resumes_without_writes(runner, monkeypatch, run_id="bdi")


def test_restored_absolute_artifact_location_does_not_invalidate_csv(
    runner, monkeypatch
):
    _save_extended_result(runner)
    fields, rows = _read_csv(runner)
    rows[0]["artifact_dir"] = str(runner.state / "restored-from-drive")
    _write_csv(runner, fields, rows)
    _assert_resumes_without_writes(runner, monkeypatch)


@pytest.mark.parametrize(
    "field", ["method_config_json", "refnorm_max_score", "artifact_sha256"]
)
def test_missing_csv_column_blocks_resume(runner, monkeypatch, field):
    _save_job(runner, _job(runner))
    fields, rows = _read_csv(runner)
    fields.remove(field)
    rows[0].pop(field)
    _write_csv(runner, fields, rows)
    _assert_blocked_without_writes(runner, monkeypatch, field)


@pytest.mark.parametrize("kind", ["missing-cell", "duplicate-header", "extra-cell"])
def test_malformed_csv_records_are_rejected(runner, monkeypatch, kind):
    _save_job(runner, _job(runner))
    with _csv_path(runner).open(newline="", encoding="utf-8") as stream:
        records = list(csv.reader(stream))
    if kind == "missing-cell":
        field = records[0][-1]
        records[1].pop()
    elif kind == "duplicate-header":
        field = "refnorm_max_score"
        records[0].append(field)
        records[1].append("0.5")
    else:
        field = "CSV"
        records[1].append("unexpected-extra-cell")
    with _csv_path(runner).open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(records)
    _assert_blocked_without_writes(runner, monkeypatch, field)


def test_legacy_dictionary_is_not_executed(runner, monkeypatch, tmp_path):
    _save_job(runner, _job(runner))
    fields, rows = _read_csv(runner)
    sentinel = tmp_path / "must-not-be-created"
    rows[0]["artifact_sha256"] = (
        f"__import__('pathlib').Path({str(sentinel)!r}).touch()"
    )
    _write_csv(runner, fields, rows)
    _assert_blocked_without_writes(runner, monkeypatch, "artifact_sha256")
    assert not sentinel.exists()


def test_numeric_and_boolean_json_values_are_not_interchangeable(runner, monkeypatch):
    directory, row = _save_extended_result(runner)
    row["diagnostics_json"] = '{"converged": true, "nested": [false]}'
    (directory / "result.json").write_text(
        json.dumps(row, sort_keys=True), encoding="utf-8"
    )
    fields, rows = _read_csv(runner)
    rows[0]["diagnostics_json"] = '{"converged": 1, "nested": [0]}'
    _write_csv(runner, fields, rows)
    _assert_blocked_without_writes(runner, monkeypatch, "diagnostics_json")


def test_extra_report_columns_are_accepted(runner, monkeypatch):
    _save_extended_result(runner)
    fields, rows = _read_csv(runner)
    fields.append("report_note")
    rows[0]["report_note"] = "This annotation is not a saved run-result field."
    _write_csv(runner, fields, rows)
    _assert_resumes_without_writes(runner, monkeypatch)


def test_pandas_export_with_nullable_integer_columns_is_accepted(runner, monkeypatch):
    directory, row = _save_extended_result(runner)
    row["method_seconds"] = 0.12345678912345678
    (directory / "result.json").write_text(
        json.dumps(row, sort_keys=True), encoding="utf-8"
    )
    unrelated = {
        **row,
        "run_id": "another-method",
        "candidate_budget": None,
        "peak_gpu_memory_bytes": None,
        "unique_candidate_count": None,
    }
    exported = pd.DataFrame([row, unrelated])
    assert exported["candidate_budget"].dtype.kind == "f"
    exported.to_csv(_csv_path(runner), index=False)
    _assert_resumes_without_writes(runner, monkeypatch)


def test_malformed_csv_quoting_blocks_resume(runner, monkeypatch):
    _save_job(runner, _job(runner))
    fields, _ = _read_csv(runner)
    with _csv_path(runner).open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(fields)
        stream.write('"unterminated quoted CSV cell')
    _assert_blocked_without_writes(runner, monkeypatch, "CSV")


def test_equivalent_seed_formats_cannot_hide_duplicate_rows(runner, monkeypatch):
    _save_job(runner, _job(runner))
    fields, rows = _read_csv(runner)
    rows.append({**rows[0], "method_seed": "0.0"})
    _write_csv(runner, fields, rows)
    _assert_blocked_without_writes(runner, monkeypatch, "method_seed")
