"""Evaluator-owned, immutable per-attempt artifacts and explicit resumption.

Result files are committed by atomic rename within a uniquely reserved attempt
directory. A process lock prevents concurrent workers from running the same
logical method/task/seed, including when an earlier process was interrupted.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


def json_value(value: Any) -> Any:
    """Produce portable JSON, representing missing numeric measurements as null."""
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def array_fingerprint(value: np.ndarray | torch.Tensor) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_saved_statistic(
    row: Mapping[str, Any], field: str, expected: float | int
) -> None:
    """Require a finite numeric summary; integer counts must match exactly."""
    actual = row.get(field)
    try:
        matches = (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and (
                actual == expected
                if isinstance(expected, int)
                else np.isclose(float(actual), expected, rtol=1e-12, atol=1e-12)
            )
        )
    except OverflowError:
        matches = False
    if not matches:
        raise ValueError(
            f"saved statistic {field} is missing, invalid, or disagrees with raw artifacts"
        )


def verify_successful_attempt(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a completed attempt before reuse; return ``(manifest, result_row)``."""
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    row = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if row.get("status") != "success":
        raise ValueError(f"attempt is not successful: {directory}")
    logical = manifest["logical_config"]
    fingerprint = stable_fingerprint(logical)
    if (
        fingerprint != manifest["logical_fingerprint"]
        or fingerprint != row["logical_fingerprint"]
    ):
        raise ValueError("attempt logical fingerprint does not match its configuration")
    for key, value in logical.items():
        if key in row and json_value(row[key]) != value:
            raise ValueError(
                f"saved result disagrees with its logical configuration: {key}"
            )
    hashes = row.get("artifact_sha256", {})
    for name in ("candidates.npz", "evaluation.npz"):
        artifact = directory / name
        if not artifact.is_file() or hashes.get(name) != file_sha256(artifact):
            raise ValueError(f"missing or corrupted successful artifact: {artifact}")
    with np.load(directory / "candidates.npz", allow_pickle=False) as archive:
        candidates = archive["candidates"]
        target_context = archive["target_context"]
    count = int(logical["candidate_budget"])
    dimension = int(logical["design_space"]["dimension"])
    dtype = logical["dtype"].removeprefix("torch.")
    if (
        count < 1
        or candidates.shape != (count, dimension)
        or str(candidates.dtype) != dtype
    ):
        raise ValueError(
            "saved candidate shape or dtype violates the logical configuration"
        )
    if not np.isfinite(candidates).all() or not np.isfinite(target_context).all():
        raise ValueError("saved candidates or target context contain non-finite values")
    if target_context.tolist() != json.loads(row["target_context_json"]):
        raise ValueError("saved target context differs from the result row")
    with np.load(directory / "evaluation.npz", allow_pickle=False) as archive:
        utility = archive["utility"]
        score = archive["refnorm_score"]
        for values in (utility, score):
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(
                    "saved evaluation has invalid shape or non-finite values"
                )
        negative_loss = (
            json.loads(logical["problem_metadata_json"]).get("utility_transform")
            == "negative_loss"
        )
        if negative_loss and (
            "raw_loss" not in archive
            or not np.array_equal(archive["raw_loss"], -utility)
        ):
            raise ValueError("saved raw loss disagrees with utility=-loss")
    low, high = row["reference_min_utility"], row["reference_max_utility"]
    width = high - low
    threshold = 1e-12 * max(1.0, abs(low), abs(high))
    expected_score = (
        np.zeros_like(utility) if width <= threshold else (utility - low) / width
    )
    if not np.allclose(score, expected_score, rtol=1e-12, atol=1e-12):
        raise ValueError("saved refnorm scores disagree with the fixed reference")
    for label, values in (("raw", utility), ("refnorm", score)):
        suffix = "utility" if label == "raw" else "score"
        for statistic, reduction in (
            ("max", np.max),
            ("median", np.median),
            ("mean", np.mean),
        ):
            if not np.isclose(
                row[f"{label}_{statistic}_{suffix}"],
                reduction(values),
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    "saved aggregate score disagrees with candidate evaluation"
                )
    if negative_loss:
        loss = -utility
        for statistic, reduction in (
            ("min", np.min),
            ("median", np.median),
            ("mean", np.mean),
        ):
            _verify_saved_statistic(row, f"raw_{statistic}_loss", float(reduction(loss)))

    # Match seed_runner._candidate_diagnostics exactly, including the saved
    # dtype: casting float32 candidates before rounding can change the count.
    unique_count = len(np.unique(np.round(candidates, decimals=12), axis=0))
    _verify_saved_statistic(row, "unique_candidate_count", unique_count)
    _verify_saved_statistic(row, "unique_candidate_fraction", unique_count / count)
    if not 0.0 <= row["unique_candidate_fraction"] <= 1.0:
        raise ValueError("saved statistic unique_candidate_fraction must be in [0, 1]")
    return manifest, row


def capture_environment(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        gpu = {"name": props.name, "total_memory_bytes": props.total_memory}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "gpu": gpu,
        "device": str(device),
        "installed_packages": _installed_packages(),
    }


@lru_cache(maxsize=1)
def _installed_packages() -> dict[str, str]:
    return dict(
        sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        )
    )


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    atomic_bytes(
        path,
        (json.dumps(json_value(value), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
        replace=replace,
    )


def atomic_bytes(path: Path, value: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _component(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")[:64] or "unnamed"
    return readable + "-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


class _ProcessLock:
    def __init__(self, path: Path) -> None:
        self.stream = path.open("a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            raise RuntimeError(
                f"another worker is running this logical trial: {path}"
            ) from exc

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()  # Closing releases the OS lock, also after crashes.


@contextmanager
def result_directory_lock(directory: Path):
    """Reject concurrent suite writers before any method training is started."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = _ProcessLock(directory / ".suite.lock")
    try:
        yield
    finally:
        lock.close()


@dataclass
class RunAttempt:
    path: Path
    fingerprint: str
    previous_result: dict[str, Any] | None
    _lock: _ProcessLock

    @classmethod
    def begin(
        cls,
        results_dir: Path,
        *,
        experiment_id: str,
        task_id: str,
        run_id: str,
        seed: int,
        logical_config: Mapping[str, Any],
        environment: Mapping[str, Any],
        resume: bool,
        infrastructure_retry_reason: str | None,
    ) -> RunAttempt:
        directory = (
            results_dir
            / "runs"
            / _component(experiment_id)
            / _component(task_id)
            / _component(run_id)
            / f"seed-{seed}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        lock = _ProcessLock(directory / ".process.lock")
        try:
            fingerprint = stable_fingerprint(logical_config)
            attempts = sorted(directory.glob("attempt-[0-9][0-9][0-9][0-9]"))
            if attempts:
                last = attempts[-1]
                manifest_path = last / "manifest.json"
                if not manifest_path.is_file():
                    raise RuntimeError(
                        f"incomplete attempt reservation requires inspection: {last}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["logical_fingerprint"] != fingerprint:
                    raise ValueError(
                        "logical configuration changed; use a new experiment or run ID"
                    )
                result_path = last / "result.json"
                previous = (
                    json.loads(result_path.read_text(encoding="utf-8"))
                    if result_path.is_file()
                    else None
                )
                if previous and previous["status"] == "success":
                    if not resume:
                        raise FileExistsError(
                            "successful trial exists; explicit resume is required"
                        )
                    _, verified = verify_successful_attempt(last)
                    return cls(last, fingerprint, verified, lock)
                if not infrastructure_retry_reason:
                    raise FileExistsError(
                        "failed or interrupted trial exists; an explicit infrastructure retry "
                        "reason is required (algorithm failures must not be silently retried)"
                    )
            attempt_path = directory / f"attempt-{len(attempts) + 1:04d}"
            attempt_path.mkdir(exist_ok=False)
            atomic_json(
                attempt_path / "manifest.json",
                {
                    "artifact_schema_version": 1,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "logical_fingerprint": fingerprint,
                    "logical_config": logical_config,
                    "environment": environment,
                    "infrastructure_retry_reason": infrastructure_retry_reason,
                    "previous_attempt": str(attempts[-1]) if attempts else None,
                },
            )
            return cls(attempt_path, fingerprint, None, lock)
        except BaseException:
            lock.close()
            raise

    def finish(self, row: Mapping[str, Any]) -> None:
        try:
            atomic_json(self.path / "result.json", row)
        finally:
            self.close()

    def close(self) -> None:
        self._lock.close()
