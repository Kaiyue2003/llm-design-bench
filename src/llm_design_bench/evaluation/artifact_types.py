"""Named fields for evaluator-owned manifests and runtime metadata.

Logical configurations stay extensible because RunAttempt also serves custom
evaluators. Reading an artifact must still perform checksum/statistic validation;
a type annotation alone does not establish that an external JSON file is valid.
"""

from collections.abc import Mapping
from typing import TypedDict


class GPUEnvironment(TypedDict):
    name: str
    total_memory_bytes: int


class RunEnvironment(TypedDict):
    python: str
    platform: str
    machine: str
    processor: str
    logical_cpu_count: int | None
    numpy: str
    torch: str
    cuda_runtime: str | None
    cudnn_version: int | None
    torch_threads: int
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    float32_matmul_precision: str
    gpu: GPUEnvironment | None
    device: str
    installed_packages: dict[str, str]


class AttemptManifest(TypedDict):
    artifact_schema_version: int
    created_at_utc: str
    logical_fingerprint: str
    logical_config: Mapping[str, object]
    environment: Mapping[str, object]
    infrastructure_retry_reason: str | None
    previous_attempt: str | None
