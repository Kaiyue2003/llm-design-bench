"""Read-only CSV and artifact checks for the current Colab queue.

This module cannot dispatch jobs, grant pilot approval, or repair results. The
queue owns journal validation and supplies its expected frozen configuration
only after the saved attempt has passed the lower-level artifact checks.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from llm_design_bench.evaluation.run_artifacts import (
    _component,
    verify_successful_attempt,
)

if TYPE_CHECKING:
    from colab_types import DispatchCompletion, JobIdentity

    from llm_design_bench.evaluation.seed_types import SeedResultRow


def _finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _same_json_value(actual: object, expected: object) -> bool:
    """Compare JSON meaning without treating true as the number 1."""
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, (int, float)):
        return (
            type(actual) in (int, float)
            and _finite(actual)
            and _finite(expected)
            and actual == expected
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(
                _same_json_value(actual[key], value) for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _same_json_value(left, right) for left, right in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _csv_integer(value: str) -> Decimal:
    """Accept integer-valued CSV notation without rounding through float."""
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("not an integer") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("not a finite integer")
    return number


def _csv_value_matches(value: str, expected: object, *, json_field: bool) -> bool:
    if expected is None:
        return value == ""
    if json_field and isinstance(expected, str):
        return _same_json_value(json.loads(value), json.loads(expected))
    if isinstance(expected, (dict, list)):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            # pandas exports artifact_sha256 as a Python dict repr in existing
            # releases. Parse literals only; never execute CSV cell contents.
            parsed = ast.literal_eval(value)
        return _same_json_value(parsed, expected)
    if isinstance(expected, bool):
        return value in ({"True", "true"} if expected else {"False", "false"})
    if isinstance(expected, int):
        return _csv_integer(value) == expected
    if isinstance(expected, float):
        actual = float(value)
        return (
            math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    return isinstance(expected, str) and value == expected


def _verify_csv_result(
    csv_row: Mapping[str, str], result: Mapping[str, object]
) -> None:
    """Check every saved result field; allow extra report-only CSV columns."""
    for field, expected in result.items():
        if field not in csv_row:
            raise ValueError(f"result CSV is missing column: {field}")
        if field == "artifact_dir":
            # Restoring a snapshot relocates this absolute display path. The
            # relative path is checked against the latest attempt separately.
            continue
        try:
            matches = _csv_value_matches(
                csv_row[field], expected, json_field=field.endswith("_json")
            )
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError(
                f"invalid result CSV value: {field}; inspect before resuming"
            ) from exc
        if not matches:
            raise ValueError(
                f"result CSV differs from verified result.json: {field}; "
                "inspect before resuming (no automatic repair or retraining)"
            )


def read_result_records(
    path: Path, *, task_id: str, run_id: str, method_seed: int
) -> list[dict[str, str]]:
    """Select one logical seed without weakening validation of other CSV rows."""
    records: list[dict[str, str]] = []
    if not path.exists():
        return records
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, strict=True)
        required = {
            "task_id",
            "run_id",
            "method_seed",
            "phase",
            "status",
            "artifact_relative_dir",
        }
        try:
            fields = reader.fieldnames or []
            missing = required.difference(fields)
            if missing:
                raise ValueError(
                    f"result CSV lacks required columns: {sorted(missing)}"
                )
            duplicates = {field for field in fields if fields.count(field) > 1}
            if duplicates or "" in fields:
                raise ValueError(
                    f"result CSV has duplicate/empty headers: {sorted(duplicates)}"
                )
            for record in reader:
                if None in record:
                    raise ValueError(
                        f"result CSV has extra cells at line {reader.line_num}"
                    )
                absent = [key for key, value in record.items() if value is None]
                if absent:
                    raise ValueError(f"result CSV has missing cells: {absent}")
                if record["task_id"] != task_id or record["run_id"] != run_id:
                    continue
                try:
                    seed = _csv_integer(record["method_seed"])
                except ValueError as exc:
                    raise ValueError("invalid result CSV method_seed") from exc
                if seed == method_seed:
                    records.append(record)
        except csv.Error as exc:
            raise ValueError(f"malformed result CSV at line {reader.line_num}") from exc
    return records


def verified_result_row(
    *,
    state: Path,
    experiment_id: str,
    task_id: str,
    job: JobIdentity,
    completion: DispatchCompletion | None,
    expected_logical: Callable[[], Mapping[str, object]],
) -> SeedResultRow | None:
    """Verify the CSV, latest attempt, configuration, and finite diagnostics."""
    root = state / job["phase"]
    trial_dir = (
        root
        / "runs"
        / _component(f"{experiment_id}_{job['phase']}")
        / _component(task_id)
        / _component(job["run_id"])
        / f"seed-{job['seed']}"
    )
    attempts = sorted(trial_dir.glob("attempt-[0-9][0-9][0-9][0-9]"))
    records = read_result_records(
        root / "method_seed_results.csv",
        task_id=task_id,
        run_id=job["run_id"],
        method_seed=job["seed"],
    )
    if not records:
        if completion is not None or attempts:
            raise RuntimeError(
                "prior attempt exists without a result row in CSV; inspect/restore"
            )
        return None
    if len(records) != 1:
        raise ValueError("duplicate result CSV rows for task_id/run_id/method_seed")
    if records[0]["phase"] != job["phase"]:
        raise ValueError("result CSV phase differs from the requested job")
    if records[0]["status"] != "success":
        raise RuntimeError("failed attempt requires inspection; no automatic retry")
    if completion is None or completion.get("status") != "success":
        raise RuntimeError("successful artifact lacks its verified completion journal")
    if not (state / f"environment-{job['run_id']}.json").is_file():
        raise RuntimeError(
            "successful result is missing its original environment contract"
        )
    relative = Path(records[0]["artifact_relative_dir"])
    artifact = (root / relative).resolve()
    if (
        relative.is_absolute()
        or not artifact.is_relative_to(root.resolve())
        or not attempts
        or artifact != attempts[-1].resolve()
    ):
        raise ValueError(
            "artifact must be the latest attempt inside its trial directory"
        )
    manifest, row = verify_successful_attempt(artifact)
    if manifest["logical_config"] != expected_logical():
        raise ValueError(
            "saved logical configuration differs from the frozen task/plan"
        )
    with np.load(artifact / "candidates.npz", allow_pickle=False) as archive:
        candidates = archive["candidates"]
        if (candidates < -1e-6).any() or not np.allclose(
            candidates.sum(axis=1), 1.0, rtol=0, atol=1e-6
        ):
            raise ValueError("saved candidates violate the simplex")
    for key in ("training_summary_json", "diagnostics_json"):
        if not _finite(json.loads(row[key])):
            raise ValueError(f"non-finite diagnostic: {key}")
    for key in ("method_seconds", "evaluation_seconds", "total_seconds"):
        value = row.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"missing or non-finite runtime: {key}")
        if value < 0:
            raise ValueError(f"negative runtime: {key}")
    peak = row.get("peak_gpu_memory_bytes")
    if (job["device"] == "cuda" or peak is not None) and (
        not isinstance(peak, (float, int)) or not math.isfinite(peak) or peak < 0
    ):
        raise ValueError("missing or invalid peak GPU allocation")
    _verify_csv_result(records[0], row)
    return cast("SeedResultRow", row)
