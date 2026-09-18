"""Freeze and run the agreed LLM-DM protocol, without selecting method budgets.

Preparation is read-only with respect to the oracle. A frozen plan expands every
method constructor default, so subsequent code defaults cannot silently change a
run. Only the evaluator receives reference utilities and the simulator.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch

from llm_design_bench.evaluation.data_manifest import (
    FrozenDataManifest,
    make_frozen_data_recipes_task_spec,
)
from llm_design_bench.evaluation.run_artifacts import verify_successful_attempt
from llm_design_bench.evaluation.formal_methods import require_resolved_method
from llm_design_bench.evaluation.plan_types import (
    FrozenMethodPlan,
    MethodPlanEntry,
    MethodPlanPayload,
    MethodRequest,
    PackageSourceIdentity,
    SharedSettings,
    read_plan_shape,
)
from llm_design_bench.evaluation.seed_runner import (
    DEFAULT_METHOD_SEEDS,
    MethodSpec,
    SeedBenchmarkConfig,
    resolved_method_config,
)
from llm_design_bench.evaluation.unified_report import (
    BenchmarkTaskSpec,
    UnifiedBenchmarkResult,
    run_benchmark_suite,
)
from llm_design_bench.optimizers.registry import make_method

PROTOCOL_ID = "llmdm_scale_stratified_v1"
DOUBLE_METHODS = frozenset({"bo_qei", "ga_on_gp", "bdi"})


def shared_settings() -> SharedSettings:
    """Return a fresh JSON-compatible copy of the group-wide settings."""
    return {
        "protocol_id": PROTOCOL_ID,
        "metric_index": 4,
        "objective": "eval/RedPajamaStackExchange/CrossEntropyLoss",
        "utility_transform": "negative_loss",
        "split": "model_scale_grouped_utility_percentile",
        "utility_percentiles": [0.0, 40.0],
        "percentile_method": "linear",
        "threshold_ties": "include_all",
        "target_model_scale": 1000,
        "target_training_steps": 19500,
        "fixed_1b": "subset_of_main_visible_without_resplitting",
        "candidate_budget": 128,
        "duplicates": "allowed_and_counted_after_rounding_to_12_decimals",
        "pilot_seeds": [0],
        "formal_seeds": list(DEFAULT_METHOD_SEEDS),
        "reference": "full_multiscale_logged_utility_evaluator_only",
        "training_normalization": "visible_data_only",
        "primary_score": "refnorm_max_score",
        "uncertainty": "sample_sd_ddof1_and_se",
        "mixed_precision": False,
        "float64_methods": sorted(DOUBLE_METHODS),
        "other_methods_dtype": "float32",
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def package_source_identity() -> PackageSourceIdentity:
    """Hash installed Python sources; commit alone cannot identify dirty code."""
    package = Path(__file__).resolve().parents[1]
    sources = {
        path.relative_to(package).as_posix(): hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    checkout = package.parent.parent
    commit = None
    dirty = None
    if (checkout / ".git").exists():
        try:
            prefix = ["git", "-c", f"safe.directory={checkout.as_posix()}"]
            commit = subprocess.check_output(
                [*prefix, "rev-parse", "HEAD"],
                cwd=checkout,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    [*prefix, "status", "--porcelain"],
                    cwd=checkout,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                ).strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return {"sha256": _digest(sources), "git_commit": commit, "git_dirty": dirty}


def _validate_name(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError(
            f"{name} must use letters, digits, underscores, dots or hyphens"
        )


def freeze_method_plan(
    bundle: FrozenDataManifest,
    methods: Sequence[MethodRequest],
    *,
    experiment_id: str,
) -> FrozenMethodPlan:
    """Expand explicitly selected methods; do not train, search, or query oracle.

    An empty kwargs mapping explicitly selects that method's current defaults.
    The protocol does not impose one member's budgets on another's methods.
    """
    _validate_name(experiment_id, "experiment_id")
    if not methods:
        raise ValueError("select at least one method with explicit kwargs")
    entries: list[MethodPlanEntry] = []
    for entry in methods:
        if not isinstance(entry, Mapping):
            raise TypeError("each method must be an object")
        if set(entry) - {"method_id", "kwargs", "run_id"}:
            raise ValueError("method entries allow only method_id, kwargs and run_id")
        if "method_id" not in entry or "kwargs" not in entry:
            raise ValueError("each method requires method_id and explicit kwargs")
        method_id = entry["method_id"]
        run_id = entry.get("run_id", method_id)
        _validate_name(method_id, "method_id")
        require_resolved_method(method_id)
        _validate_name(run_id, "run_id")
        if not isinstance(entry["kwargs"], Mapping):
            raise TypeError("method kwargs must be an object")
        requested = json.loads(_canonical(dict(entry["kwargs"])))
        method = make_method(method_id, **requested)
        resolved = json.loads(_canonical(resolved_method_config(method, requested)))
        # Ensure the saved JSON is executable without implicit constructor defaults.
        replay = make_method(method_id, **resolved)
        if _canonical(resolved_method_config(replay, resolved)) != _canonical(resolved):
            raise ValueError(f"method {method_id!r} does not round-trip its config")
        entries.append(
            {
                "method_id": method_id,
                "run_id": run_id,
                "requested_kwargs": requested,
                "kwargs": resolved,
                "dtype": "float64" if method_id in DOUBLE_METHODS else "float32",
            }
        )
    if len({entry["run_id"] for entry in entries}) != len(entries):
        raise ValueError("method run_id values must be unique")
    payload: MethodPlanPayload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "data_manifest_id": bundle.manifest_id,
        "shared_settings": shared_settings(),
        "package_source": package_source_identity(),
        "methods": entries,
    }
    return {**payload, "plan_id": _digest(payload)}


def save_method_plan(plan: Mapping[str, object], path: str | Path) -> Path:
    """Write a new plan; never overwrite an existing frozen configuration."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return output


def validate_method_plan(
    plan: Mapping[str, object], bundle: FrozenDataManifest
) -> None:
    payload = dict(plan)
    plan_id = payload.pop("plan_id", None)
    if plan_id != _digest(payload):
        raise ValueError("frozen method plan hash mismatch")
    if (
        payload.get("schema_version") != 1
        or payload.get("shared_settings") != shared_settings()
    ):
        raise ValueError("frozen plan does not match the agreed LLM-DM protocol")
    if payload.get("data_manifest_id") != bundle.manifest_id:
        raise ValueError("frozen plan references a different data manifest")
    # Keep the public Mapping input contract; the shape reader describes JSON dicts.
    checked = read_plan_shape(dict(plan))
    if checked["package_source"]["sha256"] != package_source_identity()["sha256"]:
        raise ValueError(
            "package source changed; create a new plan and rerun the pilot"
        )
    _validate_name(checked["experiment_id"], "experiment_id")
    if not checked["methods"]:
        raise ValueError("frozen plan contains no methods")
    run_ids = []
    for entry in checked["methods"]:
        require_resolved_method(entry["method_id"])
        _validate_name(entry["run_id"], "run_id")
        run_ids.append(entry["run_id"])
        expected = "float64" if entry["method_id"] in DOUBLE_METHODS else "float32"
        if entry["dtype"] != expected:
            raise ValueError("method dtype does not match the shared protocol")
        method = make_method(entry["method_id"], **entry["kwargs"])
        if _canonical(resolved_method_config(method, entry["kwargs"])) != _canonical(
            entry["kwargs"]
        ):
            raise ValueError(
                "frozen method config is incomplete or no longer compatible"
            )
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("frozen plan contains duplicate run_id values")


def load_method_plan(path: str | Path, bundle: FrozenDataManifest) -> FrozenMethodPlan:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise TypeError("frozen plan must be a JSON object")
    validate_method_plan(plan, bundle)
    # Structural and semantic validation above precede this JSON-boundary cast.
    return cast(FrozenMethodPlan, plan)


def _require_pilots(
    results_dir: str | Path | None,
    plan: FrozenMethodPlan,
    tasks: Sequence[BenchmarkTaskSpec],
    specs: Sequence[MethodSpec],
) -> None:
    if results_dir is None:
        raise ValueError("formal runs require pilot_results from the same frozen plan")
    path = Path(results_dir) / "method_seed_results.csv"
    if not path.is_file():
        raise ValueError("pilot method_seed_results.csv is missing")
    frame = pd.read_csv(path)
    required = {
        "task_id",
        "run_id",
        "method_seed",
        "phase",
        "status",
        "provenance_json",
        "artifact_relative_dir",
    }
    if not required.issubset(frame.columns):
        raise ValueError("pilot results do not have the frozen-protocol schema")
    for task in tasks:
        for spec in specs:
            rows = frame[
                (frame["task_id"] == task.task_id)
                & (frame["run_id"] == spec.run_id)
                & (frame["method_seed"] == 0)
                & (frame["phase"] == "pilot")
            ]
            if len(rows) != 1 or rows.iloc[0]["status"] != "success":
                raise ValueError(
                    f"successful seed-0 pilot missing: {task.task_id}/{spec.run_id}"
                )
            provenance = json.loads(rows.iloc[0]["provenance_json"])
            if provenance.get("plan_id") != plan["plan_id"]:
                raise ValueError("pilot was run under a different frozen plan")
            root = Path(results_dir).resolve()
            relative = rows.iloc[0]["artifact_relative_dir"]
            if not isinstance(relative, str):
                raise TypeError("pilot is missing its artifact location")
            artifact = (root / relative).resolve()
            if not artifact.is_relative_to(root):
                raise ValueError(
                    "pilot artifact path must stay inside its results directory"
                )
            manifest, saved = verify_successful_attempt(artifact)
            logical = manifest["logical_config"]
            expected = {
                "phase": "pilot",
                "method_seed": 0,
                "candidate_budget": 128,
                "task_id": task.task_id,
                "run_id": spec.run_id,
                "method_id": spec.method_id,
                "dtype": str(spec.dtype),
            }
            if any(logical.get(key) != value for key, value in expected.items()):
                raise ValueError(
                    "pilot artifact configuration does not match the frozen plan"
                )
            if json.loads(logical["method_config_json"]) != spec.kwargs:
                raise ValueError("pilot method budget does not match the frozen plan")
            if json.loads(logical["provenance_json"]).get("plan_id") != plan["plan_id"]:
                raise ValueError("pilot artifact belongs to a different frozen plan")
            if saved["status"] != "success":
                raise ValueError(
                    "pilot artifacts do not contain a complete successful result"
                )


def run_frozen_experiment(
    bundle: FrozenDataManifest,
    plan: FrozenMethodPlan,
    *,
    data_recipes_root: str | Path,
    results_dir: str | Path,
    phase: str,
    setting: str = "multi_scale",
    run_ids: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    device: str = "cpu",
    oracle_device: str = "cpu",
    pilot_results: str | Path | None = None,
    resume: bool = False,
    infrastructure_retry_reason: str | None = None,
) -> UnifiedBenchmarkResult:
    """Explicit execution entry point; never called by preparation or freezing."""
    validate_method_plan(plan, bundle)
    if phase not in {"pilot", "formal"}:
        raise ValueError("phase must be pilot or formal")
    if setting not in {"multi_scale", "fixed_1b", "both"}:
        raise ValueError("setting must be multi_scale, fixed_1b or both")
    expected_seeds = (0,) if phase == "pilot" else DEFAULT_METHOD_SEEDS
    selected_seeds = tuple(expected_seeds if seeds is None else seeds)
    if not selected_seeds or any(seed not in expected_seeds for seed in selected_seeds):
        raise ValueError(
            f"{phase} seeds must be a non-empty subset of {expected_seeds}"
        )
    available = {entry["run_id"]: entry for entry in plan["methods"]}
    chosen = list(available) if run_ids is None else list(run_ids)
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(available):
        raise ValueError("select unique run IDs present in the frozen plan")
    specs = [
        MethodSpec(
            available[name]["method_id"],
            available[name]["kwargs"],
            run_id=name,
            dtype=getattr(torch, available[name]["dtype"]),
        )
        for name in chosen
    ]
    scales = (
        [None, 1000.0]
        if setting == "both"
        else [1000.0 if setting == "fixed_1b" else None]
    )
    tasks = [
        make_frozen_data_recipes_task_spec(
            bundle,
            data_recipes_root=data_recipes_root,
            logged_model_scale=scale,
            device=oracle_device,
        )
        for scale in scales
    ]
    if phase == "formal":
        _require_pilots(pilot_results, plan, tasks, specs)
    provenance = {
        "protocol_id": PROTOCOL_ID,
        "plan_id": plan["plan_id"],
        "data_manifest_id": bundle.manifest_id,
        "package_source_sha256": plan["package_source"]["sha256"],
        "oracle_device": oracle_device,
    }
    return run_benchmark_suite(
        tasks,
        specs,
        config=SeedBenchmarkConfig(
            experiment_id=f"{plan['experiment_id']}_{phase}",
            seeds=selected_seeds,
            required_seeds=expected_seeds,
            phase=phase,
            candidate_budget=128,
            device=device,
            results_dir=Path(results_dir),
            save_artifacts=True,
            resume=resume,
            infrastructure_retry_reason=infrastructure_retry_reason,
            package_commit=plan["package_source"]["git_commit"],
            provenance=provenance,
        ),
        metadata={"frozen_plan": dict(plan), "protocol_id": PROTOCOL_ID},
    )
