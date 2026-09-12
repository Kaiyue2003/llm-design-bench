"""Colab orchestration tests use local temporary directories, not actual Drive."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "scripts" / "colab_support.py"
SPEC = importlib.util.spec_from_file_location("colab_support", HELPER)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


@pytest.fixture
def trees(tmp_path):
    local, backup = tmp_path / "local", tmp_path / "backup"
    local.mkdir()
    return local, backup


def test_round_trip_and_immutable_snapshots(trees, tmp_path):
    local, backup = trees
    (local / "nested").mkdir()
    (local / "nested" / "values.txt").write_text("first", encoding="utf-8")
    first = support.snapshot_tree(local, backup)
    first_bytes = first.read_bytes()
    (local / "nested" / "values.txt").write_text("second", encoding="utf-8")
    latest = support.snapshot_tree(local, backup)
    restored = tmp_path / "restored"
    assert support.restore_latest(backup, restored) == latest
    assert (restored / "nested" / "values.txt").read_text() == "second"
    assert first.read_bytes() == first_bytes
    assert len(list((backup / "snapshots").glob("*.tar.gz"))) == 2


def test_skips_ephemeral_locks_and_pending_files(trees, tmp_path):
    local, backup = trees
    for name in [".suite.lock", ".process.lock", ".pending-123", "result.json"]:
        (local / name).write_text("{}")
    support.snapshot_tree(local, backup)
    restored = tmp_path / "restored"
    support.restore_latest(backup, restored)
    assert [path.name for path in restored.iterdir()] == ["result.json"]


@pytest.mark.parametrize("kind", ["truncated", "missing_checksum", "bad_hash"])
def test_invalid_latest_falls_back_with_warning(trees, tmp_path, kind):
    local, backup = trees
    (local / "data").write_text("good")
    first = support.snapshot_tree(local, backup)
    latest = support.snapshot_tree(local, backup)
    checksum = latest.with_name(latest.name + ".sha256")
    if kind == "missing_checksum":
        checksum.unlink()
    else:
        latest.write_bytes(b"not a gzip archive")
        if kind == "truncated":
            checksum.write_text(hashlib.sha256(latest.read_bytes()).hexdigest())
    restored = tmp_path / "restored"
    with pytest.warns(UserWarning, match="Skipping invalid snapshot"):
        assert support.restore_latest(backup, restored) == first
    assert (restored / "data").read_text() == "good"


@pytest.mark.parametrize(
    "member_name",
    ["../outside", "/outside", "a/../../outside", "a\\outside", "C:evil", "a/./x"],
)
def test_rejects_unsafe_archive_paths(trees, tmp_path, member_name):
    _, backup = trees
    destination = backup / "snapshots" / "snapshot-999.tar.gz"
    destination.parent.mkdir(parents=True)
    with tarfile.open(destination, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    destination.with_name(destination.name + ".sha256").write_text(
        hashlib.sha256(destination.read_bytes()).hexdigest()
    )
    with (
        pytest.warns(UserWarning, match="unsafe"),
        pytest.raises(FileNotFoundError, match="no complete"),
    ):
        support.restore_latest(backup, tmp_path / "restore")
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize(
    "kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE]
)
def test_rejects_nonregular_archive_members(trees, tmp_path, kind):
    _, backup = trees
    path = backup / "snapshots" / "snapshot-999.tar.gz"
    path.parent.mkdir(parents=True)
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("unexpected")
        info.type = kind
        info.linkname = "../escape"
        archive.addfile(info)
    path.with_name(path.name + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.warns(UserWarning, match="unsafe"), pytest.raises(FileNotFoundError):
        support.restore_latest(backup, tmp_path / "restore")


def test_refuses_nonempty_restore_target(trees):
    local, backup = trees
    (local / "keep.txt").write_text("user data")
    support.snapshot_tree(local, backup)
    with pytest.raises(FileExistsError, match="fresh or empty"):
        support.restore_latest(backup, local)
    assert (local / "keep.txt").read_text() == "user data"


def test_refuses_symlinked_source_without_traversing(trees, monkeypatch):
    local, backup = trees
    source = local / "link"
    source.write_text("not followed")
    original = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink", lambda path: path == source or original(path)
    )
    with pytest.raises(ValueError, match="refuses symlinks"):
        support.snapshot_tree(local, backup)
    assert not list((backup / "snapshots").glob("*.tar.gz"))


@pytest.mark.parametrize("names", [["same", "same"], ["parent", "parent/child"]])
def test_conflicting_archive_members_never_partially_restore(trees, tmp_path, names):
    _, backup = trees
    source = backup / "snapshots" / "snapshot-999.tar.gz"
    source.parent.mkdir(parents=True)
    with tarfile.open(source, "w:gz") as archive:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 4
            archive.addfile(info, io.BytesIO(b"test"))
    source.with_name(source.name + ".sha256").write_text(
        hashlib.sha256(source.read_bytes()).hexdigest()
    )
    restored = tmp_path / "restore"
    with pytest.warns(UserWarning), pytest.raises(FileNotFoundError):
        support.restore_latest(backup, restored)
    assert not restored.exists()


def test_restore_rejects_oversized_archive_before_writing(trees, tmp_path, monkeypatch):
    local, backup = trees
    (local / "file").write_bytes(b"12345")
    support.snapshot_tree(local, backup)
    monkeypatch.setattr(support, "MAX_RESTORE_BYTES", 4)
    restored = tmp_path / "restore"
    with (
        pytest.warns(UserWarning, match="safety limit"),
        pytest.raises(FileNotFoundError),
    ):
        support.restore_latest(backup, restored)
    assert not restored.exists()


def test_refuses_nested_roots(trees):
    local, _ = trees
    with pytest.raises(ValueError, match="non-nested"):
        support.snapshot_tree(local, local / "backup")


def test_success_records_output_intent_and_completion(trees, capsys):
    local, backup = trees
    result = support.run_job(
        [sys.executable, "-c", "print('fake child output')"],
        local,
        backup,
        {"plan": "fake", "seed": 0},
    )
    assert result["returncode"] == 0
    assert result["status"] == "success"
    assert "fake child output" in Path(result["log"]).read_text()
    assert "fake child output" in capsys.readouterr().out
    assert len(list((backup / "journal").glob("*.intent.json"))) == 1
    assert len(list((backup / "journal").glob("*.completion.json"))) == 1
    assert Path(result["snapshot"]).is_file()


def test_child_failure_preserved_and_not_automatically_retried(trees):
    local, backup = trees
    identity = {"plan": "fake", "seed": 38}
    with pytest.raises(subprocess.CalledProcessError) as caught:
        support.run_job(
            [sys.executable, "-c", "raise SystemExit(7)"], local, backup, identity
        )
    assert caught.value.returncode == 7
    completed = list((backup / "journal").glob("*.completion.json"))
    assert support._verified_json(completed[0])["status"] == "failed"
    with pytest.raises(RuntimeError, match="prior job"):
        support.run_job([sys.executable, "-c", "pass"], local, backup, identity)
    assert len(list((backup / "journal").glob("*.intent.json"))) == 1


def test_fallback_cannot_silently_rerun_known_success(trees, tmp_path):
    local, backup = trees
    identity = {"plan": "fake", "seed": 39}
    result = support.run_job([sys.executable, "-c", "pass"], local, backup, identity)
    Path(result["snapshot"]).write_bytes(b"corrupt final snapshot")
    restored = tmp_path / "restored"
    with pytest.warns(UserWarning, match="Skipping invalid snapshot"):
        support.restore_latest(backup, restored)
    with pytest.raises(RuntimeError, match="successful prior job"):
        support.run_job([sys.executable, "-c", "pass"], restored, backup, identity)


def test_successful_restored_job_can_use_explicit_runner_resume(trees, tmp_path):
    local, backup = trees
    identity = {"plan": "fake", "seed": 39}
    support.run_job([sys.executable, "-c", "pass"], local, backup, identity)
    restored = tmp_path / "restored"
    support.restore_latest(backup, restored)
    # The actual CLI adds --resume and verifies full candidate/evaluation files.
    # This generic helper only checks dispatch provenance.
    result = support.run_job([sys.executable, "-c", "pass"], restored, backup, identity)
    assert result["status"] == "success"


def test_corrupt_journal_stops_before_launch(trees):
    local, backup = trees
    journal = backup / "journal" / "000.intent.json"
    journal.parent.mkdir(parents=True)
    journal.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        support.run_job([sys.executable, "-c", "pass"], local, backup, {"job": "bad"})


def test_failure_to_publish_final_snapshot_leaves_unresolved_intent(trees, monkeypatch):
    local, backup = trees
    original = support.snapshot_tree
    calls = 0

    def snapshot(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated final backup failure")
        return original(*args)

    monkeypatch.setattr(support, "snapshot_tree", snapshot)
    with pytest.raises(OSError, match="final backup failure"):
        support.run_job([sys.executable, "-c", "pass"], local, backup, {"job": "final"})
    assert len(list((backup / "journal").glob("*.intent.json"))) == 1
    assert not list((backup / "journal").glob("*.completion.json"))


def test_unfinished_intent_survives_missing_local_artifacts(trees, tmp_path):
    local, backup = trees
    identity = {"plan": "fake", "seed": 40}
    digest = hashlib.sha256(support._canonical(identity)).hexdigest()
    support._publish_json(
        backup / "journal" / "000.intent.json",
        {
            "identity": identity,
            "identity_sha256": digest,
        },
    )
    with pytest.raises(RuntimeError, match="prior job"):
        support.run_job([sys.executable, "-c", "pass"], local, backup, identity)
    result = support.run_job(
        [sys.executable, "-c", "pass"],
        local,
        backup,
        identity,
        infrastructure_retry_reason="Verified simulated VM termination; original seed retained",
    )
    assert result["status"] == "success"
    assert (backup / "journal" / "000.intent.json").exists()
    assert len(list((backup / "journal").glob("*.intent.json"))) == 2


def test_periodic_snapshot_and_restore_artifacts(trees, tmp_path):
    local, backup = trees
    result = support.run_job(
        [
            sys.executable,
            "-c",
            "import pathlib,time; pathlib.Path('manifest.json').write_text('{}'); time.sleep(.4)",
        ],
        local,
        backup,
        {"job": "periodic"},
        snapshot_interval=0.05,
    )
    assert len(list((backup / "snapshots").glob("*.tar.gz"))) >= 3
    restored = tmp_path / "restored"
    assert support.restore_latest(backup, restored) == Path(result["snapshot"])
    assert json.loads((restored / "manifest.json").read_text()) == {}


def test_failed_initial_backup_never_starts_child(trees, monkeypatch):
    local, backup = trees

    def fail_backup(*args):
        raise OSError("simulated backup outage")

    monkeypatch.setattr(support, "snapshot_tree", fail_backup)
    with pytest.raises(OSError, match="outage"):
        support.run_job(
            [sys.executable, "-c", "from pathlib import Path; Path('started').touch()"],
            local,
            backup,
            {"job": "backup-failure"},
        )
    assert not (local / "started").exists()
    assert len(list((backup / "journal").glob("*.intent.json"))) == 1
    assert not list((backup / "journal").glob("*.completion.json"))


@pytest.mark.parametrize("interval", [0, -1, 61, float("inf"), float("nan")])
def test_validates_snapshot_interval(trees, interval):
    local, backup = trees
    with pytest.raises(ValueError, match="snapshot_interval"):
        support.run_job(
            [sys.executable, "-c", "pass"],
            local,
            backup,
            {"job": "bad"},
            snapshot_interval=interval,
        )
