"""Shared, side-effect-free phase and seed validation for result aggregation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Literal, cast

import pandas as pd

from llm_design_bench.evaluation.seed_types import DEFAULT_METHOD_SEEDS

LEGACY_PUBLICATION_SOURCE = "legacy_publication_v1"
UNIFIED_RUNNER_SOURCE = "unified_runner"
SeedPhase = Literal["exploratory", "pilot", "formal", "legacy"]


@dataclass(frozen=True)
class SeedGroupContract:
    phase: SeedPhase
    required_seeds: frozenset[int]
    observed_seeds: frozenset[int]
    complete: bool
    rank_eligible: bool


def _missing(value: object) -> bool:
    return (
        value is None
        or value is pd.NA
        or (isinstance(value, float) and math.isnan(value))
    )


def _observed_seed(value: object) -> int:
    """Accept lossless CSV numeric representations, never truncate a seed."""
    if isinstance(value, bool) or not isinstance(value, (str, Real)):
        raise ValueError("method_seed must be a non-negative integer")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("method_seed must be a non-negative integer") from exc
    if not parsed.is_finite() or parsed < 0 or parsed != parsed.to_integral_value():
        raise ValueError("method_seed must be a non-negative integer")
    return int(parsed)


def _required_seeds(value: object) -> frozenset[int]:
    if not isinstance(value, str):
        raise ValueError("required_seeds_json must be a JSON array of integer seeds")
    try:
        decoded: object = json.loads(value)
    except (ValueError, RecursionError) as exc:
        raise ValueError(
            "required_seeds_json must be a JSON array of integer seeds"
        ) from exc
    if (
        not isinstance(decoded, list)
        or not decoded
        or any(type(seed) is not int or seed < 0 for seed in decoded)
    ):
        raise ValueError(
            "required_seeds_json must be a non-empty array of non-negative integers"
        )
    result = frozenset(cast(list[int], decoded))
    if len(result) != len(decoded):
        raise ValueError("required_seeds_json must not contain duplicate seeds")
    return result


def validate_seed_group(rows: Sequence[Mapping[str, object]]) -> SeedGroupContract:
    """Validate one task/method's rows and derive its coverage/ranking contract.

    Only explicitly marked publication-v1 rows can omit phase/required seeds.
    A modern row must never acquire legacy ranking rules by losing metadata.
    """
    if not rows:
        raise ValueError("a seed group must not be empty")
    source = rows[0].get("result_source")
    if not isinstance(source, str) or source not in (
        UNIFIED_RUNNER_SOURCE,
        LEGACY_PUBLICATION_SOURCE,
    ):
        raise ValueError(
            "result_source must be unified_runner or legacy_publication_v1"
        )
    if any(
        not isinstance(row.get("result_source"), str)
        or row.get("result_source") != source
        for row in rows
    ):
        raise ValueError("result_source must be consistent within a task/method")
    legacy = source == LEGACY_PUBLICATION_SOURCE

    phases = [row.get("phase") for row in rows]
    if legacy:
        phases = ["legacy" if _missing(phase) else phase for phase in phases]
    valid_phases = (
        ("exploratory", "pilot", "formal", "legacy")
        if legacy
        else ("exploratory", "pilot", "formal")
    )
    if any(not isinstance(phase, str) or phase not in valid_phases for phase in phases):
        raise ValueError("phase is missing or invalid for result_source")
    if any(phase != phases[0] for phase in phases[1:]):
        raise ValueError("phase must be consistent within a task/method")
    phase = cast(SeedPhase, phases[0])

    observed = [_observed_seed(row.get("method_seed")) for row in rows]
    observed_seeds = frozenset(observed)
    if len(observed_seeds) != len(rows):
        raise ValueError("duplicate method/task/seed rows are not allowed")

    required_values = [row.get("required_seeds_json") for row in rows]
    missing_required = [_missing(value) for value in required_values]
    historical_defaults = legacy and phase == "legacy" and all(missing_required)
    if historical_defaults:
        required_seeds = observed_seeds
    else:
        if any(missing_required):
            raise ValueError(
                "required_seeds_json is required for every task/method row"
            )
        required_groups = [_required_seeds(value) for value in required_values]
        required_seeds = required_groups[0]
        if any(seeds != required_seeds for seeds in required_groups[1:]):
            raise ValueError("required seeds must be consistent within a task/method")
    if phase == "formal" and required_seeds != frozenset(DEFAULT_METHOD_SEEDS):
        raise ValueError("formal required_seeds must be exactly 38-45")
    if phase == "pilot" and required_seeds != frozenset({0}):
        raise ValueError("pilot required_seeds must be exactly [0]")
    if not observed_seeds.issubset(required_seeds):
        raise ValueError("observed seed is not in required_seeds")

    statuses = [row.get("status") for row in rows]
    if any(
        not isinstance(status, str) or status not in ("success", "failed")
        for status in statuses
    ):
        raise ValueError("status must be success or failed")
    successes = statuses.count("success")
    complete = successes == len(required_seeds)
    return SeedGroupContract(
        phase=phase,
        required_seeds=required_seeds,
        observed_seeds=observed_seeds,
        complete=complete,
        rank_eligible=phase != "pilot"
        and (successes > 0 if historical_defaults else complete),
    )


def seed_group_contract(group: pd.DataFrame) -> SeedGroupContract:
    """Adapt a DataFrame group without coercing or modifying its stored values."""
    fields = ("result_source", "phase", "required_seeds_json", "method_seed", "status")
    rows = [{field: row.get(field) for field in fields} for _, row in group.iterrows()]
    try:
        return validate_seed_group(rows)
    except ValueError as exc:
        identity = (
            ", ".join(
                f"{field}={group.iloc[0][field]!r}"
                for field in ("experiment_id", "suite", "task_id", "run_id")
                if field in group
            )
            if not group.empty
            else "empty group"
        )
        seeds = group["method_seed"].tolist() if "method_seed" in group else []
        raise ValueError(f"{exc} ({identity}, method_seeds={seeds!r})") from exc


def validate_seed_contracts(per_seed: pd.DataFrame) -> None:
    """Check task/method identities independently of display metadata columns."""
    required = {"experiment_id", "run_id", "method_seed", "status", "result_source"}
    missing = sorted(required.difference(per_seed.columns))
    if missing:
        raise KeyError(f"missing seed-contract columns: {missing}")
    keys = ["experiment_id"]
    keys.extend(column for column in ("suite", "task_id") if column in per_seed)
    keys.append("run_id")
    for _, group in per_seed.groupby(keys, sort=False, dropna=False):
        seed_group_contract(group)
