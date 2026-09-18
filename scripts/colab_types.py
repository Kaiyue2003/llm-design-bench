"""Dictionary-shaped contracts shared by the current Colab orchestration code.

These annotations do not coerce JSON, alter fingerprints, or replace the
runtime checks at filesystem and process boundaries. Historical notebooks keep
using their commit-pinned helpers and serialized records unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

Phase: TypeAlias = Literal["pilot", "formal"]
Setting: TypeAlias = Literal["multi_scale", "fixed_1b"]
Device: TypeAlias = Literal["cpu", "cuda"]


class JobIdentity(TypedDict):
    """The exact fields hashed to identify a frozen single-seed job."""

    plan_id: str
    run_id: str
    setting: Setting
    phase: Phase
    seed: int
    device: Device
    oracle_device: Literal["cpu"]


QueuedJob: TypeAlias = JobIdentity
# The generic runner accepts ordinary JSON dictionaries as well as the typed
# benchmark identity. TypedDict is not statically a dict, despite being one at
# runtime; name both shapes rather than claiming to accept arbitrary Mapping.
DispatchIdentity: TypeAlias = dict[str, Any] | JobIdentity


class ReadyQueuePreview(JobIdentity):
    status: Literal["pending", "complete"]


class BlockedQueuePreview(JobIdentity):
    status: Literal["blocked"]
    error: str


QueuePreview: TypeAlias = ReadyQueuePreview | BlockedQueuePreview


class QueueOutcome(JobIdentity):
    status: Literal["skipped", "completed"]


class DispatchIntent(TypedDict):
    """Append-only intent; support also accepts non-benchmark JSON identities."""

    schema_version: int
    dispatch_id: str
    identity: DispatchIdentity
    identity_sha256: str
    command: list[str]
    infrastructure_retry_reason: str | None


class DispatchCompletion(TypedDict):
    """A child process outcome, before/after its snapshot is published."""

    dispatch_id: str
    identity_sha256: str
    status: Literal["success", "failed"]
    returncode: int | None
    wall_seconds: float
    error_type: str | None
    error: str | None
    log: str
    resources: str | None
    snapshot: NotRequired[str]


class JournalEntry(TypedDict):
    intent: DispatchIntent
    path: Path
    completion: NotRequired[DispatchCompletion]


class EnvironmentContract(TypedDict):
    device: Device
    python: str
    cuda_runtime: str | None
    cudnn: int | None
    gpu: str | None
    packages: dict[str, str]


class PilotReport(TypedDict):
    """A cost/diagnostic report; intentionally excludes oracle scores."""

    setting: Setting
    run_id: str
    status: Literal["verified", "missing", "blocked"]
    error: NotRequired[str]
    train_size: NotRequired[int | None]
    candidate_budget: NotRequired[int | None]
    device: NotRequired[str | None]
    dtype: NotRequired[str | None]
    method_seconds: NotRequired[float | None]
    evaluation_seconds: NotRequired[float | None]
    total_seconds: NotRequired[float | None]
    peak_gpu_memory_bytes: NotRequired[int | float | None]
    unique_candidate_count: NotRequired[int | float | None]
    training_summary: NotRequired[object]
    diagnostics: NotRequired[object]
    peak_gpu_memory_mib: NotRequired[float | None]
    artifact_checks: NotRequired[Literal["passed"]]
    finite_numeric_diagnostics: NotRequired[Literal[True]]
