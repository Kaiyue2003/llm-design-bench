"""Snapshot limits are tested with tiny local trees, never Drive or large files."""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path

import pytest


HELPER = Path(__file__).resolve().parents[1] / "scripts" / "colab_support.py"
SPEC = importlib.util.spec_from_file_location("colab_support_limits", HELPER)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


@pytest.fixture
def trees(tmp_path, monkeypatch):
    local, backup = tmp_path / "local", tmp_path / "backup"
    local.mkdir()
    monkeypatch.setattr(support, "MAX_RESTORE_BYTES", 8)
    return local, backup


def _write_files(root, sizes):
    contents = {f"file-{index}.bin": b"x" * size for index, size in enumerate(sizes)}
    for name, content in contents.items():
        (root / name).write_bytes(content)
    return contents


def _tree_contents(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("sizes", [[], [0], [7], [8], [3, 5], [0, 3, 5]])
def test_snapshot_at_or_below_limit_round_trips(trees, tmp_path, sizes):
    local, backup = trees
    contents = _write_files(local, sizes)

    snapshot = support.snapshot_tree(local, backup)

    with support.tarfile.open(snapshot, "r:gz") as archive:
        assert sum(member.size for member in archive.getmembers()) == sum(sizes)
    checksum = snapshot.with_name(snapshot.name + ".sha256")
    assert checksum.read_text(encoding="ascii").strip() == support._sha256(snapshot)
    restored = tmp_path / "restored"
    assert support.restore_latest(backup, restored) == snapshot
    assert _tree_contents(restored) == contents
    assert _tree_contents(local) == contents


@pytest.mark.parametrize("sizes", [[9], [4, 5], [3, 3, 3]])
def test_oversized_snapshot_does_not_publish_or_alter_input(trees, monkeypatch, sizes):
    local, backup = trees
    contents = _write_files(local, sizes)
    publications = []
    monkeypatch.setattr(support, "_publish", lambda *args: publications.append(args))

    with pytest.raises(ValueError, match="safety limit"):
        support.snapshot_tree(local, backup)

    assert publications == []
    assert _tree_contents(backup) == {}
    assert _tree_contents(local) == contents


def test_oversized_tree_is_rejected_before_tar_is_opened(trees, monkeypatch):
    local, backup = trees
    _write_files(local, [4, 5])

    def unexpected_tar_open(*args, **kwargs):
        pytest.fail("the oversized source tree should fail its preflight check")

    monkeypatch.setattr(support.tarfile, "open", unexpected_tar_open)
    with pytest.raises(ValueError, match="safety limit"):
        support.snapshot_tree(local, backup)


def test_limit_counts_uncompressed_file_bytes(trees, monkeypatch):
    local, backup = trees
    limit = 4096
    content = b"0" * (limit + 1)
    assert len(gzip.compress(content)) < limit
    monkeypatch.setattr(support, "MAX_RESTORE_BYTES", limit)
    (local / "highly-compressible.bin").write_bytes(content)

    with pytest.raises(ValueError, match="safety limit"):
        support.snapshot_tree(local, backup)

    assert _tree_contents(backup) == {}
    assert (local / "highly-compressible.bin").read_bytes() == content


def test_ignored_files_do_not_consume_snapshot_budget(trees, tmp_path):
    local, backup = trees
    (local / "result.bin").write_bytes(b"12345678")
    (local / "worker.lock").write_bytes(b"x" * 100)
    (local / ".pending-file").write_bytes(b"x" * 100)
    (local / ".pending-directory").mkdir()
    (local / ".pending-directory" / "result.bin").write_bytes(b"x" * 100)
    before = _tree_contents(local)

    snapshot = support.snapshot_tree(local, backup)

    restored = tmp_path / "restored"
    assert support.restore_latest(backup, restored) == snapshot
    assert _tree_contents(restored) == {"result.bin": b"12345678"}
    assert _tree_contents(local) == before


def test_rejected_snapshot_keeps_previous_snapshot_restorable(trees, tmp_path):
    local, backup = trees
    original = _write_files(local, [8])
    previous = support.snapshot_tree(local, backup)
    backup_before = _tree_contents(backup)
    (local / "extra.bin").write_bytes(b"x")
    source_before = _tree_contents(local)

    with pytest.raises(ValueError, match="safety limit"):
        support.snapshot_tree(local, backup)

    assert _tree_contents(backup) == backup_before
    assert _tree_contents(local) == source_before
    restored = tmp_path / "restored"
    assert support.restore_latest(backup, restored) == previous
    assert _tree_contents(restored) == original


def test_growth_after_preflight_is_rejected_before_publication(trees, monkeypatch):
    local, backup = trees
    _write_files(local, [4, 4])
    original_open = support._open_snapshot_file
    added_names = []

    def open_after_growth(source):
        added_names.append(source.name)
        if source.name == "file-1.bin":
            source.write_bytes(source.read_bytes() + b"g")
        return original_open(source)

    monkeypatch.setattr(support, "_open_snapshot_file", open_after_growth)
    publications = []
    monkeypatch.setattr(support, "_publish", lambda *args: publications.append(args))

    with pytest.raises(ValueError, match="safety limit"):
        support.snapshot_tree(local, backup)

    assert added_names == ["file-0.bin", "file-1.bin"]
    assert publications == []
    assert _tree_contents(backup) == {}
    assert _tree_contents(local) == {
        "file-0.bin": b"xxxx",
        "file-1.bin": b"xxxxg",
    }


@pytest.mark.parametrize(
    ("kind", "message"),
    [("symlink", "refuses symlinks"), ("nonregular", "regular files only")],
)
def test_source_type_is_rechecked_before_archiving(trees, monkeypatch, kind, message):
    local, backup = trees
    original = _write_files(local, [8])
    source = local / "file-0.bin"
    packing = False
    original_open = support.tarfile.open
    original_is_symlink = Path.is_symlink
    original_is_file = Path.is_file

    def start_packing(*args, **kwargs):
        nonlocal packing
        packing = True
        return original_open(*args, **kwargs)

    def changed_is_symlink(path):
        if packing and path == source and kind == "symlink":
            return True
        return original_is_symlink(path)

    def changed_is_file(path):
        if packing and path == source and kind == "nonregular":
            return False
        return original_is_file(path)

    monkeypatch.setattr(support.tarfile, "open", start_packing)
    monkeypatch.setattr(Path, "is_symlink", changed_is_symlink)
    monkeypatch.setattr(Path, "is_file", changed_is_file)
    publications = []
    monkeypatch.setattr(support, "_publish", lambda *args: publications.append(args))

    with pytest.raises(ValueError, match=message):
        support.snapshot_tree(local, backup)

    assert packing
    assert publications == []
    assert _tree_contents(backup) == {}
    packing = False
    assert _tree_contents(local) == original
