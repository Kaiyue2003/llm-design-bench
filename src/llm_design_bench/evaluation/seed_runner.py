from __future__ import annotations

import inspect
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

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


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.method_id:
            raise ValueError("method_id must not be empty")
        resolved_run_id = self.run_id or self.method_id
        if not resolved_run_id:
            raise ValueError("run_id must not be empty")
        object.__setattr__(self, "run_id", resolved_run_id)
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclass(frozen=True)
class SeedBenchmarkConfig:
    experiment_id: str = "default"
    normalization_reference_id: str = "provided_reference"
    seeds: tuple[int, ...] = DEFAULT_METHOD_SEEDS
    candidate_budget: int = 128
    device: torch.device | str = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    dataset_seed: int | None = None
    split_seed: int | None = None
    results_dir: Path = Path("results")
    fail_fast: bool = False
    package_commit: str | None = None

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
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError("dtype must be floating point")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "results_dir", Path(self.results_dir))


@dataclass(frozen=True)
class SeedBenchmarkResult:
    per_seed: pd.DataFrame
    summary: pd.DataFrame


def run_method_seed_benchmark(
    evaluator_task,
    problem: OfflineProblem,
    methods: Sequence[MethodSpec | str],
    *,
    reference_utility: np.ndarray,
    config: SeedBenchmarkConfig = SeedBenchmarkConfig(),
    write_results: bool = True,
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

    rows: list[dict[str, Any]] = []
    for spec in specs:
        metadata = get_method_metadata(spec.method_id)
        capabilities = get_method_capabilities(spec.method_id)
        _validate_method_compatibility(
            spec.method_id,
            capabilities,
            problem,
            config,
        )
        for seed in config.seeds:
            row = _base_row(
                spec=spec,
                metadata=metadata,
                problem=problem,
                config=config,
                seed=seed,
                reference_low=reference_low,
                reference_high=reference_high,
            )
            started = time.perf_counter()
            try:
                _seed_everything(seed)
                method = make_method(spec.method_id, **spec.kwargs)
                row["method_config_json"] = _json_dumps(
                    _resolved_method_config(method, spec.kwargs)
                )
                context = RunContext(
                    method_seed=seed,
                    candidate_budget=config.candidate_budget,
                    device=config.device,
                    dtype=config.dtype,
                    dataset_seed=config.dataset_seed,
                    split_seed=config.split_seed,
                )

                method_started = time.perf_counter()
                method_result = method.run(problem, context)
                row["method_seconds"] = time.perf_counter() - method_started

                evaluation_started = time.perf_counter()
                candidates = method_result.candidates.detach().cpu().numpy()
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

                row.update(
                    _candidate_scores(
                        utilities,
                        reference_low=reference_low,
                        reference_high=reference_high,
                    )
                )
                row.update(_candidate_diagnostics(candidates, problem))
                row["training_summary_json"] = _json_dumps(
                    method_result.training_summary
                )
                row["diagnostics_json"] = _json_dumps(method_result.diagnostics)
                row["status"] = "success"
            except Exception as exc:
                row["status"] = "failed"
                row["error_type"] = type(exc).__name__
                row["error_message"] = str(exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if config.fail_fast:
                    raise
            finally:
                row["total_seconds"] = time.perf_counter() - started
                rows.append(row)

    per_seed = pd.DataFrame(rows)
    summary = summarize_seed_results(per_seed)
    if write_results:
        config.results_dir.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(config.results_dir / "method_seed_results.csv", index=False)
        summary.to_csv(config.results_dir / "method_seed_summary.csv", index=False)
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
        row: dict[str, Any] = {
            column: first[column] for column in group_columns
        }
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
                "normalization_reference_id": first.get(
                    "normalization_reference_id"
                ),
                "requested_runs": int(len(group)),
                "successful_runs": int(len(successful)),
                "failed_runs": int(len(group) - len(successful)),
                "candidate_budget": int(first.get("candidate_budget", 0)),
            }
        )
        for metric in _SUMMARY_METRICS:
            values = pd.to_numeric(
                successful[metric] if metric in successful else pd.Series(dtype=float),
                errors="coerce",
            ).dropna()
            count = int(len(values))
            mean = float(values.mean()) if count else float("nan")
            sample_std = float(values.std(ddof=1)) if count > 1 else float("nan")
            standard_error = sample_std / math.sqrt(count) if count > 1 else float("nan")
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
    if isinstance(problem.design_space, SimplexSpace) and not capabilities.supports_simplex:
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
        "task_id": problem.metadata.task_name,
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
    unique_count = int(len(np.unique(rounded, axis=0)))
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


def _resolved_method_config(
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
