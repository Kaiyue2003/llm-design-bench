"""Typed seed-run configuration and result records; no execution or artifact I/O."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, TypeVar

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from llm_design_bench._integer_parameters import (
    optional_nonnegative_integer,
    require_integer,
)

DEFAULT_METHOD_SEEDS = tuple(range(38, 46))
RESULT_SCHEMA_VERSION = 1
FloatArray = NDArray[np.floating[Any]]


class SeedScores(TypedDict, total=False):
    """Scores may be null when JSON persistence encodes a non-finite metric."""

    raw_max_utility: float | None
    raw_median_utility: float | None
    raw_mean_utility: float | None
    refnorm_max_score: float | None
    refnorm_median_score: float | None
    refnorm_mean_score: float | None


class SeedDiagnostics(TypedDict, total=False):
    unique_candidate_count: int | float | None
    unique_candidate_fraction: float | None
    candidate_diversity: float | None
    candidate_novelty: float | None
    mean_mixture_entropy: float | None
    mean_active_domain_count: float | None


class _SeedIdentity(TypedDict):
    schema_version: int
    result_source: str
    run_id: str
    experiment_id: str
    method_id: str
    task_id: str
    method_seed: int
    status: Literal["success", "failed"]


class SeedResultRow(_SeedIdentity, SeedScores, SeedDiagnostics, total=False):
    """One run's record, including partially completed or failed attempts.

    Required keys identify a run. Optional keys describe fields populated at
    different execution stages, not arbitrary extensions. Nullable metrics
    reflect the JSON encoding of non-finite values, not a relaxed validator.
    """

    method_display_name: str
    display_name: str
    family: str
    implementation_kind: str
    adaptations_json: str
    source_url: str | None
    source_commit: str | None
    requested_method_config_json: str
    method_config_json: str
    package_commit: str | None
    provenance_json: str
    phase: str
    required_seeds_json: str
    task_name: str
    objective_name: str
    problem_metadata_json: str
    target_context_json: str
    dataset_seed: int | None
    split_seed: int | None
    candidate_budget: int
    train_size: int
    reference_min_utility: float
    reference_max_utility: float
    normalization_reference_id: str
    d_best_utility: float
    refnorm_d_best_score: float
    device: str
    dtype: str
    error_type: str | None
    error_message: str | None
    method_seconds: float | None
    evaluation_seconds: float | None
    total_seconds: float | None
    training_summary_json: str | None
    diagnostics_json: str | None
    environment_json: str | None
    artifact_dir: str | None
    artifact_relative_dir: str | None
    peak_gpu_memory_bytes: int | None
    logical_fingerprint: str | None
    infrastructure_retry_reason: str | None
    failure_stage: str | None
    raw_min_loss: float | None
    raw_median_loss: float | None
    raw_mean_loss: float | None
    artifact_sha256: dict[str, str]


_Batch = TypeVar("_Batch")


class SeedEvaluator(Protocol[_Batch]):
    """Only the evaluator-side operations used by the seed runner."""

    def at_target_fidelity(self, designs: NDArray[np.generic]) -> _Batch: ...

    def predict(self, batch: _Batch) -> NDArray[np.generic]: ...


def _update_scores(row: SeedScores, scores: SeedScores) -> None:
    """Apply only score fields to a record, preserving their insertion order."""
    row.update(scores)


def _update_diagnostics(row: SeedDiagnostics, diagnostics: SeedDiagnostics) -> None:
    """Apply only diagnostic fields; accept a result row by structural typing."""
    row.update(diagnostics)


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
        seeds = tuple(
            require_integer(seed, name=f"seeds[{index}]")
            for index, seed in enumerate(self.seeds)
        )
        if not seeds:
            raise ValueError("seeds must not be empty")
        if any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        candidate_budget = require_integer(
            self.candidate_budget, name="candidate_budget"
        )
        if candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        dataset_seed = optional_nonnegative_integer(
            self.dataset_seed, name="dataset_seed"
        )
        split_seed = optional_nonnegative_integer(self.split_seed, name="split_seed")
        _validate_dtype(self.dtype)
        if self.phase not in {"exploratory", "pilot", "formal"}:
            raise ValueError("phase must be exploratory, pilot, or formal")
        required = tuple(
            require_integer(seed, name=f"required_seeds[{index}]")
            for index, seed in enumerate(
                seeds if self.required_seeds is None else self.required_seeds
            )
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
        object.__setattr__(self, "candidate_budget", candidate_budget)
        object.__setattr__(self, "dataset_seed", dataset_seed)
        object.__setattr__(self, "split_seed", split_seed)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "results_dir", Path(self.results_dir))


@dataclass(frozen=True)
class SeedBenchmarkResult:
    per_seed: pd.DataFrame
    summary: pd.DataFrame


DEFAULT_SEED_BENCHMARK_CONFIG = SeedBenchmarkConfig()


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
