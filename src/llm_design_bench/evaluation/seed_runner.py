from __future__ import annotations

import inspect
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from llm_design_bench.evaluation.run_artifacts import (
    RunAttempt,
    array_fingerprint,
    atomic_bytes,
    atomic_npz,
    capture_environment,
    file_sha256,
    result_directory_lock,
)
from llm_design_bench.metrics.diversity import pairwise_diversity
from llm_design_bench.metrics.novelty import candidate_novelty
from llm_design_bench.optimizers.registry import (
    get_method_capabilities,
    get_method_metadata,
    make_method,
)
from llm_design_bench.problem import OfflineProblem, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace

DEFAULT_METHOD_SEEDS = tuple(range(38, 46))
RESULT_SCHEMA_VERSION = 1

_SCORE_COLUMNS = (
    "raw_max_utility",
    "raw_median_utility",
    "raw_mean_utility",
    "refnorm_max_score",
    "refnorm_median_score",
    "refnorm_mean_score",
)

_SUMMARY_METRICS = _SCORE_COLUMNS + (
    "raw_min_loss",
    "raw_median_loss",
    "raw_mean_loss",
    "method_seconds",
    "evaluation_seconds",
    "total_seconds",
    "unique_candidate_count",
    "unique_candidate_fraction",
    "candidate_diversity",
    "candidate_novelty",
    "mean_mixture_entropy",
    "mean_active_domain_count",
)


def _validate_dtype(dtype: torch.dtype) -> None:
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if not self.method_id:
            raise ValueError("method_id must not be empty")
        resolved_run_id = self.run_id or self.method_id
        if not resolved_run_id:
            raise ValueError("run_id must not be empty")
        object.__setattr__(self, "run_id", resolved_run_id)
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        if self.dtype is not None:
            _validate_dtype(self.dtype)


@dataclass(frozen=True)
class SeedBenchmarkConfig:
    experiment_id: str = "default"
    normalization_reference_id: str = "provided_reference"
    seeds: tuple[int, ...] = DEFAULT_METHOD_SEEDS
    candidate_budget: int = 128
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    dataset_seed: int | None = None
    split_seed: int | None = None
    results_dir: Path = Path("results")
    fail_fast: bool = False
    package_commit: str | None = None
    required_seeds: tuple[int, ...] | None = None
    phase: str = "exploratory"
    save_artifacts: bool = False
    resume: bool = False
    infrastructure_retry_reason: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    task_id: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.normalization_reference_id:
            raise ValueError("normalization_reference_id must not be empty")
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds:
            raise ValueError("seeds must not be empty")
        if any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        _validate_dtype(self.dtype)
        if self.phase not in {"exploratory", "pilot", "formal"}:
            raise ValueError("phase must be exploratory, pilot, or formal")
        required = tuple(
            int(seed)
            for seed in (seeds if self.required_seeds is None else self.required_seeds)
        )
        if not required or len(set(required)) != len(required) or min(required) < 0:
            raise ValueError(
                "required_seeds must be non-empty, unique, and non-negative"
            )
        if not set(seeds).issubset(required):
            raise ValueError("seeds must be a subset of required_seeds")
        if self.phase == "formal" and set(required) != set(DEFAULT_METHOD_SEEDS):
            raise ValueError("formal required_seeds must be exactly 38-45")
        if self.phase == "pilot" and required != (0,):
            raise ValueError("pilot required_seeds must be (0,)")
        if (
            self.infrastructure_retry_reason is not None
            and not self.infrastructure_retry_reason.strip()
        ):
            raise ValueError("infrastructure_retry_reason must not be blank")
        if (
            self.resume or self.infrastructure_retry_reason
        ) and not self.save_artifacts:
            raise ValueError("resume and infrastructure retries require save_artifacts")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "required_seeds", required)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "results_dir", Path(self.results_dir))


@dataclass(frozen=True)
class SeedBenchmarkResult:
    per_seed: pd.DataFrame
    summary: pd.DataFrame


DEFAULT_SEED_BENCHMARK_CONFIG = SeedBenchmarkConfig()


def run_method_seed_benchmark(
    evaluator_task,
    problem: OfflineProblem,
    methods: Sequence[MethodSpec | str],
    *,
    reference_utility: np.ndarray,
    config: SeedBenchmarkConfig = DEFAULT_SEED_BENCHMARK_CONFIG,
    write_results: bool = True,
) -> SeedBenchmarkResult:
    """Run paired method seeds with an exclusive artifact-backed output directory."""
    if config.save_artifacts and write_results:
        with result_directory_lock(config.results_dir):
            return _run_method_seed_benchmark_unlocked(
                evaluator_task,
                problem,
                methods,
                reference_utility=reference_utility,
                config=config,
                write_results=write_results,
            )
    return _run_method_seed_benchmark_unlocked(
        evaluator_task,
        problem,
        methods,
        reference_utility=reference_utility,
        config=config,
        write_results=write_results,
    )


def _run_method_seed_benchmark_unlocked(
    evaluator_task,
    problem: OfflineProblem,
    methods: Sequence[MethodSpec | str],
    *,
    reference_utility: np.ndarray,
    config: SeedBenchmarkConfig,
    write_results: bool,
) -> SeedBenchmarkResult:
    """Run registered methods across paired seeds, then evaluate candidates.

    ``problem`` is the only object passed to a method. ``evaluator_task`` and
    ``reference_utility`` remain evaluator-only, so final oracle information is
    unavailable during method training and candidate search.
    """

    specs = _normalize_method_specs(methods)
    _validate_specs(specs)
    reference = _validate_reference(reference_utility)
    reference_low = float(reference.min())
    reference_high = float(reference.max())
    existing_results = validate_existing_result_config(config)

    rows: list[dict[str, Any]] = []
    for spec in specs:
        method_config = replace(config, dtype=spec.dtype or config.dtype)
        metadata = get_method_metadata(spec.method_id)
        capabilities = get_method_capabilities(spec.method_id)
        _validate_method_compatibility(
            spec.method_id,
            capabilities,
            problem,
            method_config,
        )
        for seed in config.seeds:
            row = _base_row(
                spec=spec,
                metadata=metadata,
                problem=problem,
                config=method_config,
                seed=seed,
                reference_low=reference_low,
                reference_high=reference_high,
            )
            attempt = None
            method = None
            construction_error = None
            _seed_everything(seed)
            try:
                method = make_method(spec.method_id, **spec.kwargs)
                row["method_config_json"] = _json_dumps(
                    resolved_method_config(method, spec.kwargs)
                )
            except Exception as exc:  # noqa: BLE001 -- Persist arbitrary method construction failures.
                construction_error = exc
            if existing_results is not None:
                _validate_existing_method_rows(row, existing_results)
            if config.save_artifacts:
                environment = capture_environment(config.device)
                row["environment_json"] = _json_dumps(environment)
                attempt = RunAttempt.begin(
                    config.results_dir,
                    experiment_id=config.experiment_id,
                    task_id=row["task_id"],
                    run_id=str(spec.run_id),
                    seed=seed,
                    logical_config=_logical_config(row, problem, reference),
                    environment=environment,
                    resume=config.resume,
                    infrastructure_retry_reason=config.infrastructure_retry_reason,
                )
                if attempt.previous_result is not None:
                    restored = attempt.previous_result.copy()
                    restored["artifact_dir"] = str(attempt.path.resolve())
                    restored["artifact_relative_dir"] = attempt.path.relative_to(
                        config.results_dir
                    ).as_posix()
                    rows.append(restored)
                    attempt.close()
                    continue
                row["artifact_dir"] = str(attempt.path.resolve())
                row["artifact_relative_dir"] = attempt.path.relative_to(
                    config.results_dir
                ).as_posix()
                row["logical_fingerprint"] = attempt.fingerprint
                row["infrastructure_retry_reason"] = config.infrastructure_retry_reason
            started = time.perf_counter()
            stage = "construction"
            try:
                if construction_error is not None:
                    raise construction_error
                assert method is not None
                context = RunContext(
                    method_seed=seed,
                    candidate_budget=config.candidate_budget,
                    device=config.device,
                    dtype=method_config.dtype,
                    dataset_seed=config.dataset_seed,
                    split_seed=config.split_seed,
                )

                stage = "method"
                if config.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(config.device)
                method_started = time.perf_counter()
                method_result = method.run(problem, context)
                if config.device.type == "cuda":
                    torch.cuda.synchronize(config.device)
                row["method_seconds"] = time.perf_counter() - method_started

                stage = "candidate_persistence"
                candidates = method_result.candidates.detach().cpu().numpy().copy()
                row["training_summary_json"] = _json_dumps(
                    method_result.training_summary
                )
                row["diagnostics_json"] = _json_dumps(method_result.diagnostics)
                if attempt is not None:
                    atomic_npz(
                        attempt.path / "candidates.npz",
                        candidates=candidates,
                        target_context=problem.target_context.detach().cpu().numpy(),
                    )
                # No evaluator call, including target-fidelity conversion, occurs
                # until the exact ordered candidate batch is safely committed.
                stage = "evaluation"
                evaluation_started = time.perf_counter()
                candidate_batch = evaluator_task.at_target_fidelity(candidates)
                utilities = np.asarray(
                    evaluator_task.predict(candidate_batch),
                    dtype=float,
                )
                _validate_oracle_output(
                    utilities,
                    expected=config.candidate_budget,
                )
                row["evaluation_seconds"] = time.perf_counter() - evaluation_started
                negative_loss = (
                    problem.metadata.extra.get("utility_transform") == "negative_loss"
                )
                if attempt is not None:
                    evaluation_arrays = {
                        "utility": utilities,
                        "refnorm_score": np.asarray(
                            [
                                reference_normalize(
                                    value,
                                    reference_low=reference_low,
                                    reference_high=reference_high,
                                )
                                for value in utilities
                            ]
                        ),
                    }
                    if negative_loss:
                        evaluation_arrays["raw_loss"] = -utilities
                    atomic_npz(attempt.path / "evaluation.npz", **evaluation_arrays)

                stage = "diagnostics"
                row.update(
                    _candidate_scores(
                        utilities,
                        reference_low=reference_low,
                        reference_high=reference_high,
                    )
                )
                row.update(_candidate_diagnostics(candidates, problem))
                if negative_loss:
                    row.update(
                        {
                            "raw_min_loss": float((-utilities).min()),
                            "raw_median_loss": float(np.median(-utilities)),
                            "raw_mean_loss": float((-utilities).mean()),
                        }
                    )
                row["status"] = "success"
            except (Exception, KeyboardInterrupt) as exc:
                row["status"] = "failed"
                row["error_type"] = type(exc).__name__
                row["error_message"] = str(exc)
                row["failure_stage"] = stage
                row.update({column: float("nan") for column in _SCORE_COLUMNS})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if config.fail_fast or isinstance(exc, KeyboardInterrupt):
                    raise
            finally:
                row["total_seconds"] = time.perf_counter() - started
                if config.device.type == "cuda":
                    row["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(
                        config.device
                    )
                rows.append(row)
                if attempt is not None:
                    row["artifact_sha256"] = {
                        name: file_sha256(attempt.path / name)
                        for name in ("candidates.npz", "evaluation.npz")
                        if (attempt.path / name).is_file()
                    }
                    attempt.finish(row)
                if write_results:
                    _write_seed_progress(rows, config)

    per_seed = pd.DataFrame(rows)
    summary = summarize_seed_results(per_seed)
    if write_results:
        _write_seed_progress(rows, config)
    return SeedBenchmarkResult(per_seed=per_seed, summary=summary)


def summarize_seed_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    required = {
        "experiment_id",
        "run_id",
        "method_id",
        "status",
        "method_seed",
    }
    missing = sorted(required.difference(per_seed.columns))
    if missing:
        raise KeyError(f"missing seed-result columns: {missing}")

    rows: list[dict[str, Any]] = []
    group_columns = ["experiment_id"]
    group_columns.extend(
        column
        for column in (
            "suite",
            "task_id",
            "task_display_name",
            "category",
            "category_display_name",
        )
        if column in per_seed.columns
    )
    group_columns.append("run_id")
    groups = per_seed.groupby(group_columns, sort=False, dropna=False)
    for _, group in groups:
        first = group.iloc[0]
        successful = group[group["status"] == "success"]
        required_json = first.get("required_seeds_json")
        if isinstance(required_json, str):
            if group["required_seeds_json"].nunique(dropna=False) != 1:
                raise ValueError(
                    "required seeds must be consistent within a task/method"
                )
            required_seeds = set(json.loads(required_json))
        else:
            required_seeds = {int(seed) for seed in group["method_seed"]}
        observed_seeds = {int(seed) for seed in group["method_seed"]}
        if len(observed_seeds) != len(group):
            raise ValueError("duplicate method/task/seed rows are not allowed")
        if not observed_seeds.issubset(required_seeds):
            raise ValueError("observed seed is not in required_seeds")
        if "phase" in group and group["phase"].nunique(dropna=False) != 1:
            raise ValueError("phase must be consistent within a task/method")
        complete = len(successful) == len(required_seeds)
        rank_eligible = first.get("phase") != "pilot" and (
            complete if isinstance(required_json, str) else len(successful) > 0
        )
        environment_variants = (
            group["environment_json"].dropna().nunique()
            if "environment_json" in group
            else 0
        )
        row: dict[str, Any] = {column: first[column] for column in group_columns}
        row.update(
            {
                "method_id": first["method_id"],
                "method_display_name": first.get(
                    "method_display_name",
                    first.get("display_name", first["method_id"]),
                ),
                "display_name": first.get(
                    "method_display_name",
                    first.get("display_name", first["method_id"]),
                ),
                "family": first.get("family"),
                "implementation_kind": first.get("implementation_kind"),
                "adaptations_json": first.get("adaptations_json"),
                "source_url": first.get("source_url"),
                "source_commit": first.get("source_commit"),
                "requested_method_config_json": first.get(
                    "requested_method_config_json"
                ),
                "method_config_json": first.get("method_config_json"),
                "package_commit": first.get("package_commit"),
                "normalization_reference_id": first.get("normalization_reference_id"),
                "phase": first.get("phase", "legacy"),
                "required_seeds_json": _json_dumps(sorted(required_seeds)),
                "requested_runs": len(required_seeds),
                "attempted_runs": len(group),
                "successful_runs": len(successful),
                "failed_runs": int(len(group) - len(successful)),
                "missing_runs": len(required_seeds - observed_seeds),
                "complete_seed_set": complete,
                "rank_eligible": rank_eligible,
                "environment_variants": int(environment_variants),
                "mixed_environment": environment_variants > 1,
                "candidate_budget": int(first.get("candidate_budget", 0)),
            }
        )
        for metric in _SUMMARY_METRICS:
            values = pd.to_numeric(
                successful[metric] if metric in successful else pd.Series(dtype=float),
                errors="coerce",
            ).dropna()
            count = len(values)
            mean = float(values.mean()) if count else float("nan")
            sample_std = float(values.std(ddof=1)) if count > 1 else float("nan")
            standard_error = (
                sample_std / math.sqrt(count) if count > 1 else float("nan")
            )
            row[f"{metric}_n"] = count
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = sample_std
            row[f"{metric}_se"] = standard_error
        rows.append(row)
    return pd.DataFrame(rows)


def _normalize_method_specs(
    methods: Sequence[MethodSpec | str],
) -> tuple[MethodSpec, ...]:
    return tuple(
        method if isinstance(method, MethodSpec) else MethodSpec(str(method))
        for method in methods
    )


def _validate_specs(specs: tuple[MethodSpec, ...]) -> None:
    if not specs:
        raise ValueError("methods must not be empty")
    run_ids = [spec.run_id for spec in specs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("method run_id values must be unique")
    for spec in specs:
        get_method_metadata(spec.method_id)


def _validate_method_compatibility(
    method_id: str,
    capabilities,
    problem: OfflineProblem,
    config: SeedBenchmarkConfig,
) -> None:
    if (
        isinstance(problem.design_space, SimplexSpace)
        and not capabilities.supports_simplex
    ):
        raise ValueError(f"method {method_id!r} does not support simplex designs")
    if isinstance(problem.design_space, BoxSpace) and not capabilities.supports_box:
        raise ValueError(f"method {method_id!r} does not support box designs")
    if capabilities.requires_gpu and config.device.type != "cuda":
        raise ValueError(f"method {method_id!r} requires a CUDA device")
    if problem.context_dim and not capabilities.supports_context:
        context = problem.train_context.detach().cpu()
        context_varies = bool(
            torch.any(context != context[0]).item()
            or torch.any(problem.target_context.detach().cpu() != context[0]).item()
        )
        if context_varies:
            raise ValueError(
                f"method {method_id!r} does not support varying fidelity context"
            )


def _validate_reference(values: np.ndarray) -> np.ndarray:
    reference = np.asarray(values, dtype=float).reshape(-1)
    if reference.size == 0:
        raise ValueError("reference_utility must not be empty")
    if not np.isfinite(reference).all():
        raise ValueError("reference_utility must be finite")
    return reference


def _base_row(
    *,
    spec: MethodSpec,
    metadata,
    problem: OfflineProblem,
    config: SeedBenchmarkConfig,
    seed: int,
    reference_low: float,
    reference_high: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_source": "unified_runner",
        "run_id": spec.run_id,
        "experiment_id": config.experiment_id,
        "method_id": spec.method_id,
        "method_display_name": metadata.display_name,
        "display_name": metadata.display_name,
        "family": metadata.family.value,
        "implementation_kind": metadata.implementation_kind.value,
        "adaptations_json": _json_dumps(metadata.adaptations),
        "source_url": metadata.source_url,
        "source_commit": metadata.source_commit,
        "requested_method_config_json": _json_dumps(spec.kwargs),
        "method_config_json": _json_dumps(spec.kwargs),
        "package_commit": config.package_commit,
        "provenance_json": _json_dumps(config.provenance),
        "phase": config.phase,
        "required_seeds_json": _json_dumps(config.required_seeds),
        "task_id": config.task_id or problem.metadata.task_name,
        "task_name": problem.metadata.task_name,
        "objective_name": problem.metadata.objective_name,
        "problem_metadata_json": _json_dumps(problem.metadata.extra),
        "target_context_json": _json_dumps(
            problem.target_context.detach().cpu().tolist()
        ),
        "method_seed": seed,
        "dataset_seed": config.dataset_seed,
        "split_seed": config.split_seed,
        "candidate_budget": config.candidate_budget,
        "train_size": problem.sample_count,
        "reference_min_utility": reference_low,
        "reference_max_utility": reference_high,
        "normalization_reference_id": config.normalization_reference_id,
        "d_best_utility": float(problem.train_utility.max().detach().cpu()),
        "device": str(config.device),
        "dtype": str(config.dtype),
        "status": "failed",
        "error_type": None,
        "error_message": None,
        "method_seconds": float("nan"),
        "evaluation_seconds": float("nan"),
        "total_seconds": float("nan"),
        "training_summary_json": None,
        "diagnostics_json": None,
        "environment_json": None,
        "artifact_dir": None,
        "artifact_relative_dir": None,
        "peak_gpu_memory_bytes": None,
        "logical_fingerprint": None,
        "infrastructure_retry_reason": None,
        "failure_stage": None,
        "raw_min_loss": float("nan"),
        "raw_median_loss": float("nan"),
        "raw_mean_loss": float("nan"),
        "unique_candidate_count": float("nan"),
        "unique_candidate_fraction": float("nan"),
        "candidate_diversity": float("nan"),
        "candidate_novelty": float("nan"),
    }
    row.update({column: float("nan") for column in _SCORE_COLUMNS})
    row["refnorm_d_best_score"] = reference_normalize(
        row["d_best_utility"],
        reference_low=reference_low,
        reference_high=reference_high,
    )
    return row


def _logical_config(
    row: Mapping[str, Any],
    problem: OfflineProblem,
    reference: np.ndarray,
) -> dict[str, Any]:
    fields = (
        "schema_version",
        "experiment_id",
        "task_id",
        "run_id",
        "method_id",
        "method_config_json",
        "package_commit",
        "provenance_json",
        "phase",
        "required_seeds_json",
        "method_seed",
        "dataset_seed",
        "split_seed",
        "candidate_budget",
        "dtype",
        "device",
        "normalization_reference_id",
        "problem_metadata_json",
        "objective_name",
        "target_context_json",
        "reference_min_utility",
        "reference_max_utility",
    )
    return {
        **{field: row[field] for field in fields},
        "data_fingerprints": {
            field: array_fingerprint(getattr(problem, field))
            for field in (
                "train_designs",
                "train_context",
                "train_utility",
                "target_context",
            )
        },
        "reference_fingerprint": array_fingerprint(reference),
        "design_space": {
            "type": type(problem.design_space).__name__,
            "dimension": problem.design_space.dimension,
            "tolerance": getattr(problem.design_space, "tolerance", None),
            "bounds": (
                array_fingerprint(problem.design_space.bounds)
                if isinstance(problem.design_space, BoxSpace)
                else None
            ),
        },
    }


def merge_result_rows(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    resume: bool,
    infrastructure_retry_reason: str | None,
) -> pd.DataFrame:
    """Merge disjoint shards, replacing an attempted seed only under explicit policy."""
    keys = ["experiment_id", "task_id", "run_id", "method_seed"]
    for frame in (previous, current):
        if frame.duplicated(keys).any():
            raise ValueError("duplicate method/task/seed rows are not allowed")
    for column in ("experiment_id", "phase", "required_seeds_json", "provenance_json"):
        combined = pd.concat([previous, current], ignore_index=True)
        if column not in combined or combined[column].nunique(dropna=False) != 1:
            raise ValueError(f"cannot merge results with incompatible {column}")
    old = previous.set_index(keys, drop=False)
    new = current.set_index(keys, drop=False)
    overlap = old.index.intersection(new.index)
    for key in overlap:
        old_row, new_row = old.loc[key], new.loc[key]
        fingerprint = old_row.get("logical_fingerprint")
        if not isinstance(fingerprint, str) or fingerprint != new_row.get(
            "logical_fingerprint"
        ):
            raise ValueError(
                "refusing to replace a row with a different logical configuration"
            )
        if old_row["status"] == "success":
            if not resume or _artifact_identity(old_row) != _artifact_identity(new_row):
                raise FileExistsError(
                    "successful result requires explicit resume of its saved attempt"
                )
        elif not (
            infrastructure_retry_reason
            or _artifact_identity(old_row) == _artifact_identity(new_row)
        ):
            raise FileExistsError(
                "replacing a failure requires an explicit infrastructure retry reason"
            )
    return pd.concat([old.drop(index=overlap), new], ignore_index=True)


def validate_existing_result_config(config: SeedBenchmarkConfig) -> pd.DataFrame | None:
    """Reject incompatible experiment shards before allocating another trial."""
    path = config.results_dir / "method_seed_results.csv"
    if not config.save_artifacts or not path.exists():
        return None
    previous = pd.read_csv(path)
    expected = {
        "experiment_id": config.experiment_id,
        "phase": config.phase,
        "required_seeds_json": _json_dumps(config.required_seeds),
        "provenance_json": _json_dumps(config.provenance),
    }
    for key, value in expected.items():
        if (
            key not in previous
            or previous[key].nunique(dropna=False) != 1
            or previous.iloc[0][key] != value
        ):
            raise ValueError(f"existing result directory has incompatible {key}")
    return previous


def _validate_existing_method_rows(
    row: Mapping[str, Any], previous: pd.DataFrame
) -> None:
    matching = previous[previous["run_id"] == row["run_id"]]
    if matching.empty:
        return
    for column in (
        "method_id",
        "method_config_json",
        "requested_method_config_json",
        "candidate_budget",
        "dtype",
        "device",
        "package_commit",
    ):
        for value in matching[column]:
            expected = row[column]
            if pd.isna(value) and expected is None:
                continue
            if value != expected:
                raise ValueError(
                    f"logical configuration changed: existing run ID has incompatible {column}; "
                    "use a new experiment"
                )


def _artifact_identity(row: Mapping[str, Any]) -> Any:
    relative = row.get("artifact_relative_dir")
    return relative if isinstance(relative, str) else row.get("artifact_dir")


def _write_seed_progress(
    rows: list[dict[str, Any]], config: SeedBenchmarkConfig
) -> None:
    frame = pd.DataFrame(rows)
    path = config.results_dir / "method_seed_results.csv"
    if config.save_artifacts and path.exists():
        # Existing rows from this invocation are derived from the same immutable
        # attempts; merging them is equivalent to resuming those saved rows.
        frame = merge_result_rows(
            pd.read_csv(path),
            frame,
            resume=True,
            infrastructure_retry_reason=config.infrastructure_retry_reason,
        )
    atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"), replace=True)
    atomic_bytes(
        config.results_dir / "method_seed_summary.csv",
        summarize_seed_results(frame).to_csv(index=False).encode("utf-8"),
        replace=True,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_oracle_output(utilities: np.ndarray, expected: int) -> None:
    if utilities.shape != (expected,):
        raise ValueError(
            f"oracle returned utility shape {utilities.shape}; expected {(expected,)}"
        )
    if not np.isfinite(utilities).all():
        raise ValueError("oracle returned non-finite utilities")


def _candidate_scores(
    utilities: np.ndarray,
    *,
    reference_low: float,
    reference_high: float,
) -> dict[str, float]:
    maximum = float(np.max(utilities))
    median = float(np.median(utilities))
    mean = float(np.mean(utilities))
    return {
        "raw_max_utility": maximum,
        "raw_median_utility": median,
        "raw_mean_utility": mean,
        "refnorm_max_score": reference_normalize(
            maximum,
            reference_low=reference_low,
            reference_high=reference_high,
        ),
        "refnorm_median_score": reference_normalize(
            median,
            reference_low=reference_low,
            reference_high=reference_high,
        ),
        "refnorm_mean_score": reference_normalize(
            mean,
            reference_low=reference_low,
            reference_high=reference_high,
        ),
    }


def reference_normalize(
    value: float,
    *,
    reference_low: float,
    reference_high: float,
) -> float:
    width = reference_high - reference_low
    minimum_width = 1e-12 * max(
        1.0,
        abs(reference_low),
        abs(reference_high),
    )
    return 0.0 if width <= minimum_width else (value - reference_low) / width


def _candidate_diagnostics(
    candidates: np.ndarray,
    problem: OfflineProblem,
) -> dict[str, float | int]:
    normalized_candidates = _normalized_designs(candidates, problem)
    logged = problem.train_designs.detach().cpu().numpy()
    normalized_logged = _normalized_designs(logged, problem)
    rounded = np.round(candidates, decimals=12)
    unique_count = len(np.unique(rounded, axis=0))
    diagnostics: dict[str, float | int] = {
        "unique_candidate_count": unique_count,
        "unique_candidate_fraction": unique_count / len(candidates),
        "candidate_diversity": pairwise_diversity(normalized_candidates),
        "candidate_novelty": candidate_novelty(
            normalized_candidates,
            normalized_logged,
        ),
    }
    if isinstance(problem.design_space, SimplexSpace):
        safe = np.where(candidates > 0, candidates, 1.0)
        entropy = -np.sum(
            np.where(candidates > 0, candidates * np.log(safe), 0.0),
            axis=1,
        )
        diagnostics["mean_mixture_entropy"] = float(entropy.mean())
        diagnostics["mean_active_domain_count"] = float(
            np.sum(candidates > 0.01, axis=1).mean()
        )
    return diagnostics


def _normalized_designs(
    designs: np.ndarray,
    problem: OfflineProblem,
) -> np.ndarray:
    designs = np.asarray(designs, dtype=float)
    if isinstance(problem.design_space, BoxSpace):
        bounds = problem.design_space.bounds.detach().cpu().numpy()
        return (designs - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return designs


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def resolved_method_config(
    method,
    requested: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture constructor defaults without serializing learned model state."""

    resolved = dict(requested)
    try:
        signature = inspect.signature(type(method).__init__)
    except (TypeError, ValueError):
        return resolved
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if hasattr(method, name):
            resolved[name] = getattr(method, name)
        elif parameter.default is not inspect.Parameter.empty:
            resolved.setdefault(name, parameter.default)
    return resolved


# Keep the private spelling available to older local callers.
_resolved_method_config = resolved_method_config
