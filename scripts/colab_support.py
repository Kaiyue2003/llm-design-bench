"""Small, stdlib-only helpers for explicit, single-worker Colab jobs.

Run on the VM's local disk, not a mounted Drive filesystem. Drive is only an
append-only backup destination. The actual Colab/Drive backend is not tested by
the unit tests. Snapshots preserve artifacts, not in-memory optimizer progress.
There is no runtime keepalive or automatic retry here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import warnings
from datetime import UTC, datetime
from io import BufferedReader
from pathlib import Path, PurePosixPath
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from colab_types import DispatchCompletion, DispatchIntent

MAX_RESTORE_BYTES = 4 * 1024**3


def _id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _roots(local_root: Path | str, backup_root: Path | str) -> tuple[Path, Path]:
    local = Path(local_root).resolve()
    backup = Path(backup_root).resolve()
    if local.is_relative_to(backup) or backup.is_relative_to(local):
        raise ValueError(
            "local and backup roots must be separate, non-nested directories"
        )
    return local, backup


def _publish(source: Path, destination: Path) -> None:
    """Copy, verify, then publish a unique name; never remove the source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    temporary = destination.with_name(".pending-" + uuid.uuid4().hex)
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if _sha256(temporary) != _sha256(source):
            raise OSError("backup copy checksum mismatch")
        # Names contain UUIDs and this helper permits only one notebook worker.
        # Do not assume Drive implements cross-runtime filesystem locking.
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        if _sha256(destination) != _sha256(source):
            raise OSError("published backup checksum mismatch")
    finally:
        temporary.unlink(missing_ok=True)


def _publish_json(destination: Path, payload: Any) -> None:
    with tempfile.TemporaryDirectory(prefix="llmdm-journal-") as temporary:
        source = Path(temporary) / "record.json"
        source.write_bytes(_canonical(payload) + b"\n")
        _publish(source, destination)
        checksum = Path(temporary) / "checksum"
        checksum.write_text(_sha256(source) + "\n", encoding="ascii")
        _publish(checksum, destination.with_name(destination.name + ".sha256"))


def _verified_json(path: Path) -> dict[str, Any]:
    checksum = path.with_name(path.name + ".sha256")
    if path.is_symlink() or not checksum.is_file() or checksum.is_symlink():
        raise ValueError(f"journal record lacks a trusted checksum: {path}")
    if _sha256(path) != checksum.read_text(encoding="ascii").strip():
        raise ValueError(f"journal record is corrupt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"invalid journal record: {path}")
    return value


def _ignored(path: Path) -> bool:
    return any(
        part.startswith(".pending-") for part in path.parts
    ) or path.name.endswith(".lock")


def _check_archive_size(total: int) -> None:
    if total > MAX_RESTORE_BYTES:
        raise ValueError(
            f"archive contents ({total} bytes) exceed the restoration safety limit "
            f"({MAX_RESTORE_BYTES} uncompressed bytes)"
        )


def snapshot_tree(local_root: Path | str, backup_root: Path | str) -> Path:
    """Publish one size-bounded immutable archive and checksum; retain older copies."""
    local, backup = _roots(local_root, backup_root)
    if not local.is_dir():
        raise NotADirectoryError(local)
    files = []
    total = 0
    # Reject oversized state before spending time/disk space on compression.
    for path in sorted(local.rglob("*")):
        relative = path.relative_to(local)
        if _ignored(relative):
            continue
        if path.is_symlink():
            raise ValueError(f"snapshot refuses symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"snapshot accepts regular files only: {path}")
        total += path.stat().st_size
        _check_archive_size(total)
        files.append((path, relative))

    archived_bytes = 0

    def check_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
        nonlocal archived_bytes
        if not member.isfile():
            raise ValueError(f"snapshot accepts regular files only: {member.name}")
        # Logs can grow after preflight. Count the actual headers before tarfile
        # streams each member, so a growing source cannot publish an oversized tar.
        archived_bytes += member.size
        _check_archive_size(archived_bytes)
        return member

    destination = backup / "snapshots" / f"snapshot-{_id()}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="llmdm-snapshot-") as temporary:
        archive = Path(temporary) / "snapshot.tar.gz"
        with tarfile.open(archive, "w:gz", dereference=True) as handle:
            for path, relative in files:
                # Keep source-type checks next to add(), not just at preflight.
                if path.is_symlink():
                    raise ValueError(f"snapshot refuses symlinks: {path}")
                if not path.is_file():
                    raise ValueError(f"snapshot accepts regular files only: {path}")
                handle.add(
                    path,
                    arcname=relative.as_posix(),
                    recursive=False,
                    filter=check_member,
                )
        checksum = Path(temporary) / "checksum"
        checksum.write_text(_sha256(archive) + "\n", encoding="ascii")
        _publish(archive, destination)
        _publish(checksum, destination.with_name(destination.name + ".sha256"))
    return destination


def _members(archive: tarfile.TarFile) -> list[tuple[tarfile.TarInfo, Path]]:
    result = []
    names = set()
    total = 0
    for member in archive.getmembers():
        name = member.name
        pure = PurePosixPath(name)
        if (
            not member.isfile()
            or not name
            or pure.is_absolute()
            or "\\" in name
            or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or name in names
            or member.size < 0
        ):
            raise ValueError(f"unsafe or duplicate archive member: {name!r}")
        names.add(name)
        total += member.size
        _check_archive_size(total)
        result.append((member, Path(*pure.parts)))
    return result


def restore_latest(backup_root: Path | str, local_root: Path | str) -> Path:
    """Restore the newest valid snapshot into a fresh/empty directory.

    Invalid snapshots are retained and skipped with a warning. No merging,
    overwriting or tarfile.extractall is used. Return the selected archive path.
    """
    local, backup = _roots(local_root, backup_root)
    if Path(local_root).is_symlink():
        raise ValueError("restore target must not be a symlink")
    if local.exists() and (not local.is_dir() or any(local.iterdir())):
        raise FileExistsError("restore requires a fresh or empty local directory")
    local.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted((backup / "snapshots").glob("snapshot-*.tar.gz"), reverse=True)
    for source in candidates:
        try:
            checksum = source.with_name(source.name + ".sha256")
            if source.is_symlink() or checksum.is_symlink() or not checksum.is_file():
                raise ValueError("missing checksum or symlinked snapshot")
            expected = checksum.read_text(encoding="ascii").strip()
            if len(expected) != 64 or _sha256(source) != expected:
                raise ValueError("snapshot checksum mismatch")
            # Stage everything before altering the destination, including failure
            # halfway through a truncated archive whose checksum was published.
            with tempfile.TemporaryDirectory(
                prefix="llmdm-restore-", dir=local.parent
            ) as tmp:
                stage = Path(tmp) / "tree"
                stage.mkdir()
                with tarfile.open(source, "r:gz") as handle:
                    members = _members(handle)
                    for member, relative in members:
                        target = stage / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        reader = handle.extractfile(member)
                        if reader is None:
                            raise ValueError("archive file has no content")
                        with reader, target.open("xb") as writer:
                            shutil.copyfileobj(reader, writer)
                if local.exists():
                    local.rmdir()  # Checked empty above; concurrent changes fail.
                stage.rename(local)
            return source
        except (OSError, EOFError, ValueError, tarfile.TarError) as exc:
            warnings.warn(
                f"Skipping invalid snapshot {source.name}: {exc}", stacklevel=2
            )
    raise FileNotFoundError(
        "no complete, valid snapshot is available; nothing restored"
    )


def _check_previous(
    journal: Path, identity_hash: str, retry_reason: str | None, local_root: Path
) -> None:
    matching = []
    for path in sorted(journal.glob("*.intent.json")):
        record = _verified_json(path)
        if record.get("identity_sha256") == identity_hash:
            if (
                hashlib.sha256(_canonical(record["identity"])).hexdigest()
                != identity_hash
            ):
                raise ValueError("journal identity hash mismatch")
            matching.append(path)
    if not matching:
        return
    completion = matching[-1].with_name(
        matching[-1].name.removesuffix(".intent.json") + ".completion.json"
    )
    previous = _verified_json(completion) if completion.exists() else None
    if previous is not None and previous.get("identity_sha256") != identity_hash:
        raise ValueError("completion belongs to a different identity")
    if previous is not None and previous.get("status") == "success":
        local_completion = (
            local_root / "_colab_jobs" / previous["dispatch_id"] / "completion.json"
        )
        if not local_completion.is_file() or json.loads(
            local_completion.read_text(encoding="utf-8")
        ) != {key: value for key, value in previous.items() if key != "snapshot"}:
            raise RuntimeError(
                "a successful prior job is recorded but its local completion is missing "
                "or changed; restore its complete snapshot and inspect artifacts before "
                "continuing. Do not silently rerun after falling back to an older backup."
            )
    if (previous is None or previous.get("status") != "success") and not retry_reason:
        raise RuntimeError(
            "a prior job was interrupted or failed; inspect its records, restore the "
            "latest snapshot, and supply an explicit infrastructure retry reason. "
            "Algorithm failures must not be relabeled as infrastructure failures."
        )


def _kill_group(pid: int, sig: int) -> None:
    # This POSIX-only API has no Windows stub; it is called only in POSIX branches.
    cast(Callable[[int, int], None], getattr(os, "killpg"))(pid, sig)


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            _kill_group(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                _kill_group(process.pid, cast(int, getattr(signal, "SIGKILL")))
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_job(
    command: list[str],
    local_root: Path | str,
    backup_root: Path | str,
    identity: Mapping[str, object],
    *,
    snapshot_interval: float = 60,
    infrastructure_retry_reason: str | None = None,
) -> DispatchCompletion:
    """Run one explicit job, preserving dispatch, logs and periodic snapshots.

    Only one worker may use a backup root; cross-runtime locks are not promised.
    Nonzero exit codes raise CalledProcessError *after* saving failure artifacts.
    Even a complete VM loss leaves an intent requiring inspection before retry.
    """
    if not command or any(not isinstance(arg, str) or not arg for arg in command):
        raise ValueError("command must be a nonempty list of nonempty strings")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("identity must be a nonempty JSON object")
    if not math.isfinite(snapshot_interval) or not 0 < snapshot_interval <= 60:
        raise ValueError("snapshot_interval must be positive and at most 60 seconds")
    reason = infrastructure_retry_reason
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("infrastructure retry reason must be nonempty text")
    local, backup = _roots(local_root, backup_root)
    local.mkdir(parents=True, exist_ok=True)
    journal = backup / "journal"
    journal.mkdir(parents=True, exist_ok=True)
    identity_hash = hashlib.sha256(_canonical(identity)).hexdigest()
    _check_previous(journal, identity_hash, reason, local)
    dispatch_id = _id()
    intent: DispatchIntent = {
        "schema_version": 1,
        "dispatch_id": dispatch_id,
        "identity": identity,
        "identity_sha256": identity_hash,
        "command": command,
        "infrastructure_retry_reason": reason,
    }
    _publish_json(journal / f"{dispatch_id}.intent.json", intent)
    job_dir = local / "_colab_jobs" / dispatch_id
    job_dir.mkdir(parents=True)
    (job_dir / "intent.json").write_bytes(_canonical(intent) + b"\n")
    snapshot_tree(local, backup)
    log_path = job_dir / "stdout.log"
    resource_path = job_dir / "resources.txt"
    actual_command = list(command)
    if os.name == "posix" and Path("/usr/bin/time").is_file():
        actual_command = [
            "/usr/bin/time",
            "-v",
            "-o",
            str(resource_path),
            "--",
            *command,
        ]
    process = None
    reader_thread = None
    reader_errors: list[BaseException] = []
    error = None
    returncode = None
    started = time.monotonic()
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                actual_command,
                cwd=local,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )

            def read_output() -> None:
                try:
                    assert process is not None and process.stdout is not None
                    with process.stdout:
                        # Popen's binary PIPE uses a buffered reader at default bufsize.
                        while chunk := cast(BufferedReader, process.stdout).read1(8192):
                            log.write(chunk)
                            log.flush()
                            sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                            sys.stdout.flush()
                except BaseException as exc:  # noqa: BLE001 -- Surface thread failures to the owning process.
                    reader_errors.append(exc)

            reader_thread = threading.Thread(target=read_output, daemon=True)
            reader_thread.start()
            next_snapshot = time.monotonic() + snapshot_interval
            try:
                while process.poll() is None:
                    if reader_errors:
                        raise RuntimeError("job log capture failed") from reader_errors[
                            0
                        ]
                    if time.monotonic() >= next_snapshot:
                        snapshot_tree(local, backup)
                        next_snapshot = time.monotonic() + snapshot_interval
                    time.sleep(min(0.1, snapshot_interval))
                returncode = process.wait()
            finally:
                _stop(process)
                reader_thread.join(timeout=10)
            if reader_thread.is_alive():
                raise RuntimeError("job output reader did not close")
            if reader_errors:
                raise RuntimeError("job log capture failed") from reader_errors[0]
            log.flush()
            os.fsync(log.fileno())
    except BaseException as exc:  # noqa: BLE001 -- Preserve interruption evidence before re-raising.
        error = exc
        if process is not None:
            _stop(process)
            returncode = process.returncode
    result: DispatchCompletion = {
        "dispatch_id": dispatch_id,
        "identity_sha256": identity_hash,
        "status": "success" if error is None and returncode == 0 else "failed",
        "returncode": returncode,
        "wall_seconds": time.monotonic() - started,
        "error_type": type(error).__name__ if error else None,
        "error": str(error) if error else None,
        "log": str(log_path),
        "resources": str(resource_path) if resource_path.is_file() else None,
    }
    (job_dir / "completion.json").write_bytes(_canonical(result) + b"\n")
    # If this copy fails, no remote completion is published: the next invocation
    # must inspect the unresolved intent, even if the child actually succeeded.
    snapshot = snapshot_tree(local, backup)
    result["snapshot"] = str(snapshot)
    _publish_json(journal / f"{dispatch_id}.completion.json", result)
    if error is not None:
        raise error
    if returncode != 0:
        # Without an error, the completed wait() above supplied an integer status.
        raise subprocess.CalledProcessError(cast(int, returncode), command)
    return result
