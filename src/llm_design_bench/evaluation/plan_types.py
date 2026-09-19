"""Typed, JSON-preserving contracts for frozen experiment plans.

Method kwargs are intentionally extensible. Plan identities, entries and shared
settings are not: their named fields are checked when reading external JSON.
These types never add defaults, coerce values, or change serialized content.
"""

from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict, cast


MethodDtype = Literal["float32", "float64"]


class PackageSourceIdentity(TypedDict):
    sha256: str
    git_commit: str | None
    git_dirty: bool | None


class SharedSettings(TypedDict):
    protocol_id: str
    metric_index: int
    objective: str
    utility_transform: str
    split: str
    utility_percentiles: list[float]
    percentile_method: str
    threshold_ties: str
    target_model_scale: int
    target_training_steps: int
    fixed_1b: str
    candidate_budget: int
    duplicates: str
    pilot_seeds: list[int]
    formal_seeds: list[int]
    reference: str
    training_normalization: str
    primary_score: str
    uncertainty: str
    mixed_precision: bool
    float64_methods: list[str]
    other_methods_dtype: MethodDtype


class MethodRequest(TypedDict):
    method_id: str
    kwargs: Mapping[str, Any]
    run_id: NotRequired[str]


class MethodPlanEntry(TypedDict):
    method_id: str
    run_id: str
    requested_kwargs: dict[str, Any]
    kwargs: dict[str, Any]
    dtype: MethodDtype


class MethodPlanPayload(TypedDict):
    schema_version: int
    experiment_id: str
    data_manifest_id: str
    shared_settings: SharedSettings
    package_source: PackageSourceIdentity
    methods: list[MethodPlanEntry]


class FrozenMethodPlan(MethodPlanPayload):
    plan_id: str


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise TypeError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _field(
    record: Mapping[str, object], key: str, types: tuple[type, ...], label: str
) -> None:
    if key not in record or type(record[key]) not in types:
        raise TypeError(f"{label}.{key} has a missing or invalid type")


def _list_field(
    record: Mapping[str, object], key: str, types: tuple[type, ...], label: str
) -> None:
    values = record.get(key)
    if not isinstance(values, list) or any(type(v) not in types for v in values):
        raise TypeError(f"{label}.{key} must be a list of the expected value type")


def read_plan_shape(value: object) -> FrozenMethodPlan:
    """Validate field types only; callers must still verify hashes and semantics.

    Extra fields are retained, not dropped from the identity being verified.
    The returned dictionaries are the originals, with no coercion or defaults.
    """
    plan = _object(value, "plan")
    for key in ("plan_id", "experiment_id", "data_manifest_id"):
        _field(plan, key, (str,), "plan")
    _field(plan, "schema_version", (int,), "plan")
    source = _object(plan.get("package_source"), "package_source")
    _field(source, "sha256", (str,), "package_source")
    _field(source, "git_commit", (str, type(None)), "package_source")
    _field(source, "git_dirty", (bool, type(None)), "package_source")
    settings = _object(plan.get("shared_settings"), "shared_settings")
    for key in (
        "protocol_id",
        "objective",
        "utility_transform",
        "split",
        "percentile_method",
        "threshold_ties",
        "fixed_1b",
        "duplicates",
        "reference",
        "training_normalization",
        "primary_score",
        "uncertainty",
        "other_methods_dtype",
    ):
        _field(settings, key, (str,), "shared_settings")
    for key in (
        "metric_index",
        "target_model_scale",
        "target_training_steps",
        "candidate_budget",
    ):
        _field(settings, key, (int,), "shared_settings")
    _field(settings, "mixed_precision", (bool,), "shared_settings")
    _list_field(settings, "utility_percentiles", (float, int), "shared_settings")
    _list_field(settings, "float64_methods", (str,), "shared_settings")
    for key in ("pilot_seeds", "formal_seeds"):
        _list_field(settings, key, (int,), "shared_settings")
    if settings["other_methods_dtype"] not in {"float32", "float64"}:
        raise ValueError("invalid shared method dtype")
    entries = plan.get("methods")
    if not isinstance(entries, list):
        raise TypeError("plan.methods must be a list")
    for index, raw_entry in enumerate(entries):
        label = f"methods[{index}]"
        entry = _object(raw_entry, label)
        for key in ("method_id", "run_id", "dtype"):
            _field(entry, key, (str,), label)
        if entry["dtype"] not in {"float32", "float64"}:
            raise ValueError("invalid method dtype")
        for key in ("requested_kwargs", "kwargs"):
            _object(entry.get(key), f"{label}.{key}")
    return cast(FrozenMethodPlan, value)
