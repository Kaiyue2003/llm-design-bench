"""Deterministic single-job lifecycle tests: no real process, thread, or Drive."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_colab_support import support


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    state = SimpleNamespace(
        local=tmp_path / "local",
        backup=tmp_path / "backup",
        events=[],
        clock=0.0,
        exit_code=0,
        running=False,
        launch_error=None,
        reader_error=None,
        interrupt=None,
        periodic_error=None,
        initial_error=None,
        final_error=None,
        publish_error=None,
        local_completion_error=None,
        fsync_error=None,
        stop_error=None,
        reader_alive=False,
        child=None,
        log=None,
    )
    job_dir = state.local / "_colab_jobs" / "test-dispatch"

    class Output(io.BytesIO):
        def read1(self, size):
            if state.reader_error is not None:
                raise state.reader_error
            return super().read1(size)

    class Child:
        def __init__(self):
            self.stdout = Output(b"captured output\n")
            self.returncode = None

        def poll(self):
            if not state.running:
                self.returncode = state.exit_code
            return self.returncode

        def wait(self):
            state.events.append("wait")
            self.returncode = state.exit_code
            return self.returncode

    class Reader:
        def __init__(self, *, target, args=(), daemon):
            assert daemon is True
            self.target, self.args = target, args

        def start(self):
            state.events.append("reader-start")
            self.target(*self.args)

        def join(self, *, timeout):
            assert timeout == 10
            state.events.append("reader-join")

        def is_alive(self):
            return state.reader_alive

    def launch(command, **kwargs):
        state.events.append("launch")
        assert kwargs["cwd"] == state.local
        assert (job_dir / "intent.json").is_file()
        assert "initial-snapshot" in state.events
        if state.launch_error is not None:
            raise state.launch_error
        state.child = Child()
        return state.child

    def stop(child):
        state.events.append("stop")
        if state.stop_error is not None:
            raise state.stop_error
        if child.returncode is None:
            child.returncode = -15

    def publish(path, payload):
        if path.name.endswith(".intent.json"):
            state.events.append("remote-intent")
            assert not (job_dir / "intent.json").exists()
        else:
            state.events.append("remote-completion")
            assert state.events[-2] == "final-snapshot"
            if state.publish_error is not None:
                raise state.publish_error
        path.write_bytes(support._canonical(payload))

    original_open = Path.open
    original_write = Path.write_bytes

    def opened(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == job_dir / "stdout.log":
            state.log = handle
        return handle

    def write(path, content):
        if path == job_dir / "intent.json":
            state.events.append("local-intent")
        elif path == job_dir / "completion.json":
            state.events.append("local-completion")
            assert state.log is not None and state.log.closed
            if state.child is not None:
                assert state.child.returncode is not None
            if state.local_completion_error is not None:
                raise state.local_completion_error
        return original_write(path, content)

    def snapshot(local, backup):
        assert (job_dir / "intent.json").is_file()
        if (job_dir / "completion.json").exists():
            phase, error = "final", state.final_error
            state.clock += 1000
        elif "launch" not in state.events:
            phase, error = "initial", state.initial_error
            state.clock += 1000
        else:
            phase, error = "periodic", state.periodic_error
        state.events.append(f"{phase}-snapshot")
        if error is not None:
            raise error
        return backup / f"{phase}-snapshot.tar.gz"

    def monotonic():
        current = state.clock
        state.clock += 1
        return current

    def sleep(seconds):
        state.events.append("sleep")
        if state.interrupt is not None:
            raise state.interrupt
        pytest.fail("a scripted running child must fail or interrupt, never spin")

    def fsync(fd):
        state.events.append("fsync")
        if state.fsync_error is not None:
            raise state.fsync_error

    monkeypatch.setattr(support, "_id", lambda: "test-dispatch")
    # Fake processes and fake snapshots exercise POSIX lifecycle ordering on
    # every host; real Windows job-boundary-only behavior has separate tests.
    monkeypatch.setattr(support, "_LIVE_SNAPSHOTS_SUPPORTED", True)
    monkeypatch.setattr(support, "_publish_json", publish)
    monkeypatch.setattr(support, "snapshot_tree", snapshot)
    monkeypatch.setattr(support, "_stop", stop)
    monkeypatch.setattr(support.subprocess, "Popen", launch)
    monkeypatch.setattr(support.threading, "Thread", Reader)
    monkeypatch.setattr(support.time, "monotonic", monotonic)
    monkeypatch.setattr(support.time, "sleep", sleep)
    monkeypatch.setattr(support.os, "fsync", fsync)
    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(Path, "write_bytes", write)
    state.job_dir = job_dir
    state.run = lambda **kwargs: support.run_job(
        [sys.executable, "-c", "pass"],
        state.local,
        state.backup,
        {"job": "lifecycle"},
        **kwargs,
    )
    state.completed = lambda: json.loads((job_dir / "completion.json").read_text())
    return state


def test_success_orders_intent_backup_child_cleanup_and_completion(lifecycle):
    result = lifecycle.run()
    assert lifecycle.events == [
        "remote-intent",
        "local-intent",
        "initial-snapshot",
        "launch",
        "reader-start",
        "wait",
        "stop",
        "reader-join",
        "fsync",
        "local-completion",
        "final-snapshot",
        "remote-completion",
    ]
    assert result["status"] == "success"
    # Each fake backup advances the clock by 1000; neither belongs to run time.
    assert result["wall_seconds"] == 2.0
    assert "snapshot" not in lifecycle.completed()


def test_nonzero_exit_publishes_failure_before_raising(lifecycle):
    lifecycle.exit_code = 7
    with pytest.raises(subprocess.CalledProcessError) as caught:
        lifecycle.run()
    assert caught.value.returncode == 7
    assert lifecycle.events[-3:] == [
        "local-completion",
        "final-snapshot",
        "remote-completion",
    ]
    assert lifecycle.completed()["status"] == "failed"
    assert lifecycle.completed()["error_type"] is None


def test_launch_failure_publishes_evidence_without_process_cleanup(lifecycle):
    failure = OSError("cannot start child")
    lifecycle.launch_error = failure
    with pytest.raises(OSError) as caught:
        lifecycle.run()
    assert caught.value is failure
    assert "stop" not in lifecycle.events
    assert "reader-start" not in lifecycle.events
    assert lifecycle.events[-1] == "remote-completion"
    assert lifecycle.completed()["returncode"] is None
    assert lifecycle.completed()["error_type"] == "OSError"


def test_keyboard_interrupt_stops_joins_and_publishes_before_reraise(lifecycle):
    failure = KeyboardInterrupt("simulated notebook interruption")
    lifecycle.running, lifecycle.interrupt = True, failure
    with pytest.raises(KeyboardInterrupt) as caught:
        lifecycle.run()
    assert caught.value is failure
    assert lifecycle.events.index("stop") < lifecycle.events.index("reader-join")
    assert lifecycle.events.index("reader-join") < lifecycle.events.index(
        "local-completion"
    )
    assert lifecycle.events[-1] == "remote-completion"
    assert lifecycle.completed()["error_type"] == "KeyboardInterrupt"
    assert lifecycle.completed()["returncode"] == -15


@pytest.mark.parametrize("failure", [OSError("pipe failed"), KeyboardInterrupt()])
def test_reader_base_exception_reaches_owner_and_failure_record(lifecycle, failure):
    lifecycle.running, lifecycle.reader_error = True, failure
    with pytest.raises(RuntimeError, match="job log capture failed") as caught:
        lifecycle.run()
    assert caught.value.__cause__ is failure
    assert lifecycle.events.index("stop") < lifecycle.events.index("reader-join")
    assert lifecycle.events[-1] == "remote-completion"
    assert lifecycle.completed()["error_type"] == "RuntimeError"


def test_periodic_backup_failure_still_stops_joins_and_publishes(lifecycle):
    failure = OSError("periodic backup unavailable")
    lifecycle.running, lifecycle.periodic_error = True, failure
    with pytest.raises(OSError) as caught:
        lifecycle.run(snapshot_interval=0.01)
    assert caught.value is failure
    assert lifecycle.events.index("periodic-snapshot") < lifecycle.events.index("stop")
    assert lifecycle.events.index("reader-join") < lifecycle.events.index(
        "local-completion"
    )
    assert lifecycle.events[-1] == "remote-completion"


def test_reader_that_did_not_close_is_not_recorded_as_success(lifecycle):
    lifecycle.reader_alive = True
    with pytest.raises(RuntimeError, match="job output reader did not close"):
        lifecycle.run()
    assert "reader-join" in lifecycle.events
    assert "fsync" not in lifecycle.events
    assert lifecycle.completed()["status"] == "failed"


def test_initial_backup_failure_does_not_start_or_complete_job(lifecycle):
    lifecycle.initial_error = OSError("initial backup unavailable")
    with pytest.raises(OSError, match="initial backup unavailable"):
        lifecycle.run()
    assert lifecycle.events == ["remote-intent", "local-intent", "initial-snapshot"]
    assert not (lifecycle.job_dir / "completion.json").exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_final_backup_failure_blocks_remote_completion_even_after_exit(
    lifecycle, exit_code
):
    failure = OSError("final backup unavailable")
    lifecycle.exit_code, lifecycle.final_error = exit_code, failure
    with pytest.raises(OSError) as caught:
        lifecycle.run()
    assert caught.value is failure
    assert lifecycle.completed()["returncode"] == exit_code
    assert lifecycle.events[-1] == "final-snapshot"
    assert "remote-completion" not in lifecycle.events


def test_stop_failure_keeps_original_no_completion_boundary(lifecycle):
    failure = OSError("cannot stop child")
    lifecycle.stop_error = failure
    with pytest.raises(OSError) as caught:
        lifecycle.run()
    assert caught.value is failure
    # Original cleanup retries stop in its outer exception handler; neither
    # attempt is swallowed or mislabeled as a completed dispatch.
    assert lifecycle.events[-2:] == ["stop", "stop"]
    assert "reader-join" not in lifecycle.events
    assert "local-completion" not in lifecycle.events


def test_log_fsync_failure_publishes_a_failed_record(lifecycle):
    failure = OSError("log cannot be flushed to disk")
    lifecycle.fsync_error = failure
    with pytest.raises(OSError) as caught:
        lifecycle.run()
    assert caught.value is failure
    assert lifecycle.events.index("reader-join") < lifecycle.events.index("fsync")
    assert lifecycle.completed()["status"] == "failed"
    assert lifecycle.events[-1] == "remote-completion"


def test_local_completion_failure_prevents_final_backup_and_publication(lifecycle):
    failure = OSError("local completion cannot be written")
    lifecycle.local_completion_error = failure
    with pytest.raises(OSError) as caught:
        lifecycle.run()
    assert caught.value is failure
    assert lifecycle.events[-1] == "local-completion"
    assert "final-snapshot" not in lifecycle.events


def test_remote_publication_failure_preserves_local_completion(lifecycle):
    failure = OSError("remote completion unavailable")
    lifecycle.publish_error = failure
    with pytest.raises(OSError) as caught:
        lifecycle.run()
    assert caught.value is failure
    assert lifecycle.events[-1] == "remote-completion"
    assert lifecycle.completed()["status"] == "success"
    assert not list((lifecycle.backup / "journal").glob("*.completion.json"))
