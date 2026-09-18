"""Snapshots use one file version even while the local writer replaces a CSV."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


HELPER = Path(__file__).resolve().parents[1] / "scripts" / "colab_support.py"
SPEC = importlib.util.spec_from_file_location("colab_support_consistency", HELPER)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


@pytest.fixture
def trees(tmp_path):
    local, backup = tmp_path / "local", tmp_path / "backup"
    local.mkdir()
    return local, backup


@pytest.mark.parametrize("new", [b"short\n", b"method_id,runs\ntri_mentoring,8\n"])
def test_atomic_replacement_after_header_never_publishes_truncated_content(
    trees, tmp_path, monkeypatch, new
):
    local, backup = trees
    source = local / "method_seed_summary.csv"
    old = b"method_id,runs\nroma,3\n"
    source.write_bytes(old)
    original_info = support.tarfile.TarFile.gettarinfo

    def info_then_replace(handle, *args, **kwargs):
        member = original_info(handle, *args, **kwargs)
        assert kwargs["fileobj"].fileno() >= 0
        pending = local / ".pending-csv"
        pending.write_bytes(new)
        pending.replace(source)
        return member

    monkeypatch.setattr(support.tarfile.TarFile, "gettarinfo", info_then_replace)
    if os.name == "nt":
        # Direct concurrent use is unsupported on Windows. The job runner
        # prevents it; a caller violating that boundary still fails closed.
        with pytest.raises(PermissionError):
            support.snapshot_tree(local, backup)
        assert not list(backup.rglob("*.tar.gz"))
        assert source.read_bytes() == old
        (local / ".pending-csv").replace(source)  # The failed read released its FD.
        assert source.read_bytes() == new
        return
    snapshot = support.snapshot_tree(local, backup)
    restored = tmp_path / "restored"
    assert support.restore_latest(backup, restored) == snapshot
    assert (restored / source.name).read_bytes() == old
    assert source.read_bytes() == new


def test_atomic_replacement_before_open_keeps_complete_new_version(
    trees, tmp_path, monkeypatch
):
    local, backup = trees
    source = local / "method_seed_results.csv"
    source.write_bytes(b"old")
    new = b"complete new CSV version"
    original_open = support._open_snapshot_file

    def replace_then_open(path):
        pending = local / ".pending-csv"
        pending.write_bytes(new)
        pending.replace(path)
        return original_open(path)

    monkeypatch.setattr(support, "_open_snapshot_file", replace_then_open)
    support.snapshot_tree(local, backup)
    restored = tmp_path / "restored"
    support.restore_latest(backup, restored)
    assert (restored / source.name).read_bytes() == new


@pytest.mark.parametrize("filename", ["stdout.log", "resources.txt"])
def test_append_only_job_log_can_grow_after_header(
    trees, tmp_path, monkeypatch, filename
):
    local, backup = trees
    source = local / "_colab_jobs" / "dispatch" / filename
    source.parent.mkdir(parents=True)
    old, appended = b"before\n", b"after\n"
    source.write_bytes(old)
    original_info = support.tarfile.TarFile.gettarinfo

    def info_then_append(handle, *args, **kwargs):
        member = original_info(handle, *args, **kwargs)
        with source.open("ab") as writer:
            writer.write(appended)
        return member

    monkeypatch.setattr(support.tarfile.TarFile, "gettarinfo", info_then_append)
    support.snapshot_tree(local, backup)
    restored = tmp_path / "restored"
    support.restore_latest(backup, restored)
    assert (restored / source.relative_to(local)).read_bytes() == old
    assert source.read_bytes() == old + appended


@pytest.mark.parametrize(
    "relative",
    ["result.csv", "stdout.log", "unrelated/dispatch/stdout.log"],
)
def test_append_to_non_job_file_is_rejected_before_publishing(
    trees, tmp_path, monkeypatch, relative
):
    local, backup = trees
    source = local / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"original\n")
    previous = support.snapshot_tree(local, backup)
    before = {p.name: p.read_bytes() for p in (backup / "snapshots").iterdir()}
    original_info = support.tarfile.TarFile.gettarinfo

    def info_then_append(handle, *args, **kwargs):
        member = original_info(handle, *args, **kwargs)
        with source.open("ab") as writer:
            writer.write(b"unexpected\n")
        return member

    monkeypatch.setattr(support.tarfile.TarFile, "gettarinfo", info_then_append)
    with pytest.raises(OSError, match="source changed"):
        support.snapshot_tree(local, backup)
    assert {p.name: p.read_bytes() for p in (backup / "snapshots").iterdir()} == before
    restored = tmp_path / "restored"
    assert support.restore_latest(backup, restored) == previous
    assert (restored / relative).read_bytes() == b"original\n"


def test_same_size_in_place_rewrite_is_rejected(trees, monkeypatch):
    local, backup = trees
    source = local / "result.csv"
    source.write_bytes(b"old")
    original_info = support.tarfile.TarFile.gettarinfo

    def info_then_rewrite(handle, *args, **kwargs):
        member = original_info(handle, *args, **kwargs)
        before = source.stat()
        source.write_bytes(b"new")
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        return member

    monkeypatch.setattr(support.tarfile.TarFile, "gettarinfo", info_then_rewrite)
    with pytest.raises(OSError, match="source changed"):
        support.snapshot_tree(local, backup)
    assert not list(backup.rglob("*.tar.gz"))


@pytest.mark.parametrize("relative", ["result.csv", "_colab_jobs/dispatch/stdout.log"])
def test_truncated_open_file_is_rejected(trees, monkeypatch, relative):
    local, backup = trees
    source = local / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"original content")
    original_info = support.tarfile.TarFile.gettarinfo

    def info_then_truncate(handle, *args, **kwargs):
        member = original_info(handle, *args, **kwargs)
        source.write_bytes(b"x")
        return member

    monkeypatch.setattr(support.tarfile.TarFile, "gettarinfo", info_then_truncate)
    with pytest.raises(OSError, match="unexpected end"):
        support.snapshot_tree(local, backup)
    assert not list(backup.rglob("*.tar.gz"))


def test_hardlinked_sources_are_stored_as_regular_files(trees, tmp_path):
    local, backup = trees
    source = local / "source"
    source.write_bytes(b"shared bytes")
    try:
        (local / "alias").hardlink_to(source)
    except OSError as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    snapshot = support.snapshot_tree(local, backup)
    with support.tarfile.open(snapshot, "r:gz") as archive:
        assert all(member.isfile() for member in archive.getmembers())
    restored = tmp_path / "restored"
    support.restore_latest(backup, restored)
    assert (restored / "source").read_bytes() == (restored / "alias").read_bytes()


def test_reader_releases_descriptor_after_archive_failure(trees, monkeypatch):
    local, backup = trees
    source = local / "result.csv"
    source.write_bytes(b"original")
    readers = []
    original_open = support._open_snapshot_file

    def remember_reader(path):
        reader = original_open(path)
        readers.append(reader)
        return reader

    def fail_archive(*args, **kwargs):
        raise OSError("archive write failed")

    monkeypatch.setattr(support, "_open_snapshot_file", remember_reader)
    monkeypatch.setattr(support.tarfile.TarFile, "addfile", fail_archive)
    with pytest.raises(OSError, match="archive write failed"):
        support.snapshot_tree(local, backup)
    assert len(readers) == 1 and readers[0].closed
    assert not list(backup.rglob("*.tar.gz"))


def test_snapshot_open_missing_file_fails(tmp_path):
    with pytest.raises(OSError):
        support._open_snapshot_file(tmp_path / "missing")


def test_snapshot_open_closes_descriptor_when_fdopen_fails(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"test")
    descriptors = []

    def fail_fdopen(descriptor, *args, **kwargs):
        descriptors.append(descriptor)
        raise OSError("fdopen failed")

    monkeypatch.setattr(support.os, "fdopen", fail_fdopen)
    with pytest.raises(OSError, match="fdopen failed"):
        support._open_snapshot_file(source)
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_job_boundary_backups_do_not_read_while_child_replaces_csv(
    trees, tmp_path, monkeypatch
):
    local, backup = trees
    monkeypatch.setattr(support, "_LIVE_SNAPSHOTS_SUPPORTED", False)
    original_snapshot = support.snapshot_tree
    original_popen = support.subprocess.Popen
    children, snapshots = [], []

    def launch(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    def snapshot_with_idle_writer(*args, **kwargs):
        assert not children or children[0].poll() is not None
        snapshot = original_snapshot(*args, **kwargs)
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(support.subprocess, "Popen", launch)
    monkeypatch.setattr(support, "snapshot_tree", snapshot_with_idle_writer)
    code = (
        "from pathlib import Path\n"
        "import time\n"
        "for index in range(5):\n"
        "    pending = Path('.pending-csv')\n"
        "    pending.write_text(f'index\\n{index}\\n')\n"
        "    pending.replace('results.csv')\n"
        "    print(index, flush=True)\n"
        "    time.sleep(.05)\n"
    )
    with pytest.warns(UserWarning, match="Live snapshots are disabled"):
        result = support.run_job(
            [sys.executable, "-c", code],
            local,
            backup,
            {"job": "boundary-backup"},
            snapshot_interval=0.01,
        )
    assert result["status"] == "success"
    assert len(snapshots) == 2
    assert len(list((backup / "snapshots").glob("*.tar.gz"))) == 2
    restored = tmp_path / "restored"
    assert support.restore_latest(backup, restored) == snapshots[-1]
    assert (restored / "results.csv").read_text() == "index\n4\n"
    assert list(restored.glob("_colab_jobs/*/completion.json"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow and FIFO flags")
@pytest.mark.parametrize("replacement", ["symlink", "fifo"])
def test_posix_open_rejects_nonregular_swap_without_blocking(
    trees, monkeypatch, replacement
):
    local, backup = trees
    source = local / "result.csv"
    source.write_bytes(b"original")
    original_open = support._open_snapshot_file

    def replace_before_open(path):
        source.unlink()
        if replacement == "symlink":
            source.symlink_to(__file__)
        else:
            os.mkfifo(source)
        return original_open(path)

    monkeypatch.setattr(support, "_open_snapshot_file", replace_before_open)
    with pytest.raises((OSError, ValueError)):
        support.snapshot_tree(local, backup)
    assert not list(backup.rglob("*.tar.gz"))
