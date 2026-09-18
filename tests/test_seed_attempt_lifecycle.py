"""Exercise the single-seed lifecycle, including failures between its stages."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_design_bench.evaluation import run_artifacts, seed_runner as runner
from llm_design_bench.optimizers.base import OfflineBBOMethod
from llm_design_bench.problem import MethodResult, OfflineProblem, ProblemMetadata
from llm_design_bench.spaces import SimplexSpace


def _problem():
    return OfflineProblem(
        train_designs=torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float64),
        train_context=torch.tensor([[20, 1000], [1000, 19500]], dtype=torch.float64),
        train_utility=torch.tensor([-3.0, -2.0], dtype=torch.float64),
        target_context=torch.tensor([1000, 19500], dtype=torch.float64),
        design_space=SimplexSpace(2),
        metadata=ProblemMetadata(
            task_name="lifecycle-toy",
            objective_name="cross_entropy",
            extra={"utility_transform": "negative_loss"},
        ),
    )


class _Harness:
    """Real artifact writes and locks; deterministic probes replace computation."""

    def __init__(self, monkeypatch, directory, *, failure=None, error=None):
        self.directory = directory
        self.failure = failure
        self.error = error or RuntimeError("injected lifecycle failure")
        self.events = []
        self.constructions = []
        self.dtypes = []
        self.attempts = []
        self.clock = 0.0
        self.clock_reads = []

        def counter():
            self.clock_reads.append(self.clock)
            return self.clock

        monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=counter))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        harness = self

        class ProbeMethod(OfflineBBOMethod):
            def run(self, problem, context):
                harness.hit("method", 7)
                harness.dtypes.append(context.dtype)
                # No real GPU allocation is needed to verify CUDA timing hooks.
                return MethodResult(
                    candidates=torch.tensor([[0.25, 0.75]], dtype=context.dtype).repeat(
                        context.candidate_budget, 1
                    ),
                    training_summary={"probe": True},
                )

            def optimize(self, problem, *, context, generator):
                raise AssertionError("the probe provides run() directly")

        def construct(method_id, **kwargs):
            self.constructions.append((method_id, torch.initial_seed()))
            self.hit("construction", 101)
            return ProbeMethod()

        monkeypatch.setattr(runner, "make_method", construct)

        def environment(device):
            self.hit("environment", 103)
            return {"device": str(device), "fixture": True}

        monkeypatch.setattr(runner, "capture_environment", environment)
        original_begin = runner.RunAttempt.begin

        def begin(cls, *args, **kwargs):
            self.hit("begin", 107)
            attempt = original_begin(*args, **kwargs)
            self.attempts.append(attempt)
            return attempt

        monkeypatch.setattr(runner.RunAttempt, "begin", classmethod(begin))
        original_npz = runner.atomic_npz

        def persist(path, **arrays):
            if path.name == "candidates.npz":
                self.hit("candidate_persistence", 13)
            else:
                self.hit("evaluation_persistence", 23)
            original_npz(path, **arrays)

        monkeypatch.setattr(runner, "atomic_npz", persist)
        original_diagnostics = runner._candidate_diagnostics

        def diagnostics(*args):
            self.hit("diagnostics", 29)
            return original_diagnostics(*args)

        monkeypatch.setattr(runner, "_candidate_diagnostics", diagnostics)
        original_finish = runner.RunAttempt.finish

        def finish(attempt, row, **kwargs):
            self.hit("finish", 47)
            original_finish(attempt, row, **kwargs)

        monkeypatch.setattr(runner.RunAttempt, "finish", finish)
        original_progress = runner._write_seed_progress

        def progress(rows, config):
            self.hit("progress", 53)
            original_progress(rows, config)

        monkeypatch.setattr(runner, "_write_seed_progress", progress)
        monkeypatch.setattr(
            torch.cuda, "reset_peak_memory_stats", lambda device: self.hit("reset", 5)
        )
        monkeypatch.setattr(
            torch.cuda, "synchronize", lambda device: self.hit("synchronize", 11)
        )

        def peak_memory(device):
            self.hit("peak_memory", 31)
            return 12345

        monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak_memory)

    def hit(self, name, duration):
        self.events.append(name)
        self.clock += duration
        if name == self.failure:
            self.failure = None
            raise self.error

    def at_target_fidelity(self, candidates):
        self.hit("target_fidelity", 17)
        path = self.attempts[-1].path / "candidates.npz"
        with np.load(path) as archive:
            np.testing.assert_array_equal(archive["candidates"], candidates)
        return candidates

    def predict(self, candidates):
        self.hit("oracle", 19)
        return -2.0 - candidates[:, 0]

    def run(self, *, methods=None, config=None, write_results=True):
        return runner.run_method_seed_benchmark(
            self,
            _problem(),
            methods or ["random_search"],
            reference_utility=np.array([-4.0, -1.0]),
            config=config or self.config(),
            write_results=write_results,
        )

    def config(self, **kwargs):
        return runner.SeedBenchmarkConfig(
            results_dir=self.directory,
            seeds=(38,),
            candidate_budget=4,
            save_artifacts=True,
            **kwargs,
        )

    def assert_locks_released(self):
        assert all(attempt._lock.stream.closed for attempt in self.attempts)
        with runner.result_directory_lock(self.directory):
            pass


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attempt_order_and_timer_boundaries(monkeypatch, tmp_path, device):
    harness = _Harness(monkeypatch, tmp_path)
    result = harness.run(config=harness.config(device=device))
    row = result.per_seed.iloc[0]
    expected = ["construction", "environment", "begin"]
    if device == "cuda":
        expected += ["reset"]
    expected += ["method"]
    if device == "cuda":
        expected += ["synchronize"]
    expected += [
        "candidate_persistence",
        "target_fidelity",
        "oracle",
        "evaluation_persistence",
        "diagnostics",
    ]
    if device == "cuda":
        expected += ["peak_memory"]
    expected += ["finish", "progress", "progress"]
    assert harness.events == expected
    assert harness.constructions == [("random_search", 38)]
    assert row["method_seconds"] == (18 if device == "cuda" else 7)
    assert row["evaluation_seconds"] == 36
    assert row["total_seconds"] == (124 if device == "cuda" else 108)
    assert len(harness.clock_reads) == 6
    assert harness.clock_reads[0] == 311  # Construction and preparation excluded.
    if device == "cuda":
        assert row["peak_gpu_memory_bytes"] == 12345
    run_artifacts.verify_successful_attempt(harness.attempts[-1].path)
    harness.assert_locks_released()


@pytest.mark.parametrize(
    ("failure", "stage", "has_candidates", "has_evaluation"),
    [
        ("construction", "construction", False, False),
        ("method", "method", False, False),
        ("candidate_persistence", "candidate_persistence", False, False),
        ("target_fidelity", "evaluation", True, False),
        ("oracle", "evaluation", True, False),
        ("evaluation_persistence", "evaluation", True, False),
        ("diagnostics", "diagnostics", True, True),
    ],
)
@pytest.mark.parametrize("fail_fast", [False, True])
def test_failures_commit_before_continue_or_reraise(
    monkeypatch, tmp_path, failure, stage, has_candidates, has_evaluation, fail_fast
):
    harness = _Harness(monkeypatch, tmp_path, failure=failure)
    config = replace(
        harness.config(fail_fast=fail_fast), seeds=(38, 39), required_seeds=(38, 39)
    )
    if fail_fast:
        with pytest.raises(RuntimeError, match="injected lifecycle failure"):
            harness.run(config=config)
        assert len(harness.constructions) == 1
        assert harness.events[-2:] == ["finish", "progress"]
    else:
        result = harness.run(config=config)
        assert result.per_seed["status"].tolist() == ["failed", "success"]
        assert harness.constructions == [("random_search", 38), ("random_search", 39)]
    path = harness.attempts[0].path
    saved = json.loads((path / "result.json").read_text())
    assert saved["failure_stage"] == stage
    assert saved["error_type"] == "RuntimeError"
    assert saved["raw_max_utility"] is None
    assert (path / "candidates.npz").exists() == has_candidates
    assert (path / "evaluation.npz").exists() == has_evaluation
    assert (tmp_path / "method_seed_results.csv").is_file()
    harness.assert_locks_released()


@pytest.mark.parametrize("failure", ["method", "candidate_persistence", "oracle"])
def test_interrupt_saves_and_stops_even_without_fail_fast(
    monkeypatch, tmp_path, failure
):
    harness = _Harness(
        monkeypatch, tmp_path, failure=failure, error=KeyboardInterrupt()
    )
    config = replace(harness.config(), seeds=(38, 39), required_seeds=(38, 39))
    with pytest.raises(KeyboardInterrupt):
        harness.run(config=config)
    assert len(harness.constructions) == 1
    assert harness.events[-2:] == ["finish", "progress"]
    saved = json.loads((harness.attempts[0].path / "result.json").read_text())
    assert saved["error_type"] == "KeyboardInterrupt"
    assert saved["status"] == "failed"
    harness.assert_locks_released()


def test_constructor_interrupt_is_not_reclassified_as_construction_failure(
    monkeypatch, tmp_path
):
    harness = _Harness(
        monkeypatch, tmp_path, failure="construction", error=KeyboardInterrupt()
    )
    with pytest.raises(KeyboardInterrupt):
        harness.run()
    assert harness.events == ["construction"]
    assert not harness.attempts
    assert not (tmp_path / "method_seed_results.csv").exists()
    harness.assert_locks_released()


def test_unhandled_base_exception_still_finalizes_without_relabeling(
    monkeypatch, tmp_path
):
    harness = _Harness(monkeypatch, tmp_path, failure="method", error=SystemExit(2))
    with pytest.raises(SystemExit):
        harness.run()
    saved = json.loads((harness.attempts[0].path / "result.json").read_text())
    assert saved["status"] == "failed"
    assert saved["error_type"] is None
    assert saved["failure_stage"] is None
    assert harness.events[-2:] == ["finish", "progress"]
    harness.assert_locks_released()


def test_resume_constructs_but_does_not_execute_time_or_rewrite_attempt(
    monkeypatch, tmp_path
):
    harness = _Harness(monkeypatch, tmp_path)
    first = harness.run()
    path = harness.attempts[0].path / "result.json"
    original = path.read_bytes()
    harness.events.clear()
    harness.clock_reads.clear()
    resumed = harness.run(config=harness.config(resume=True))
    assert harness.events == ["construction", "environment", "begin", "progress"]
    assert not harness.clock_reads
    assert harness.constructions == [("random_search", 38)] * 2
    assert path.read_bytes() == original
    assert len(list(tmp_path.rglob("result.json"))) == 1
    assert (
        resumed.per_seed.iloc[0]["method_seconds"]
        == first.per_seed.iloc[0]["method_seconds"]
    )
    harness.assert_locks_released()


def test_per_method_dtype_override_does_not_leak_to_later_methods(
    monkeypatch, tmp_path
):
    harness = _Harness(monkeypatch, tmp_path)
    result = harness.run(
        methods=[
            runner.MethodSpec("random_search", dtype=torch.float64),
            "best_logged",
        ],
        config=harness.config(dtype=torch.float32),
    )
    assert harness.dtypes == [torch.float64, torch.float32]
    assert result.per_seed["dtype"].tolist() == ["torch.float64", "torch.float32"]
    for attempt, dtype in zip(harness.attempts, [np.float64, np.float32], strict=True):
        with np.load(attempt.path / "candidates.npz") as archive:
            assert archive["candidates"].dtype == dtype
    harness.assert_locks_released()


def test_result_write_failure_releases_locks_without_publishing_progress(
    monkeypatch, tmp_path
):
    harness = _Harness(monkeypatch, tmp_path)
    original = run_artifacts.atomic_json

    def fail_result(path, value):
        if path.name == "result.json":
            raise OSError("result write interrupted")
        original(path, value)

    monkeypatch.setattr(run_artifacts, "atomic_json", fail_result)
    with pytest.raises(OSError, match="result write interrupted"):
        harness.run()
    assert harness.events[-1] == "finish"
    assert not (tmp_path / "method_seed_results.csv").exists()
    assert not list(tmp_path.rglob("result.json"))
    harness.assert_locks_released()


def test_hash_read_failure_releases_lock_while_caller_retains_traceback(
    monkeypatch, tmp_path
):
    # Do not capture RunAttempt: the retained exception alone used to retain its
    # open lock through the runner's traceback in a long-lived Python process.
    task = SimpleNamespace(
        at_target_fidelity=lambda candidates: candidates,
        predict=lambda candidates: -2.0 - candidates[:, 0],
    )
    config = runner.SeedBenchmarkConfig(
        results_dir=tmp_path, seeds=(38,), candidate_budget=2, save_artifacts=True
    )

    def unreadable(path):
        raise OSError("artifact checksum read failed")

    monkeypatch.setattr(runner, "file_sha256", unreadable)
    monkeypatch.setattr(runner, "capture_environment", lambda device: {})
    with pytest.raises(OSError, match="artifact checksum read failed") as caught:
        runner.run_method_seed_benchmark(
            task,
            _problem(),
            ["random_search"],
            reference_utility=np.array([-4.0, -1.0]),
            config=config,
        )
    assert caught.value.__traceback__ is not None
    process_lock = next(tmp_path.rglob(".process.lock"))
    acquired = run_artifacts._ProcessLock(process_lock)
    acquired.close()
    with runner.result_directory_lock(tmp_path):
        pass
    assert not list(tmp_path.rglob("result.json"))
    assert not (tmp_path / "method_seed_results.csv").exists()


@pytest.mark.parametrize("resume", [False, True])
def test_path_setup_failure_closes_new_or_resumed_attempt(
    monkeypatch, tmp_path, resume
):
    harness = _Harness(monkeypatch, tmp_path)
    previous_result = None
    if resume:
        harness.run()
        previous_result = (harness.attempts[0].path / "result.json").read_bytes()
        harness.events.clear()
        harness.clock_reads.clear()
    original_begin = runner._begin_seed_attempt

    class UnresolvablePath(type(tmp_path)):
        def resolve(self, *args, **kwargs):
            raise OSError("attempt path resolution failed")

    def begin(*args, **kwargs):
        attempt = original_begin(*args, **kwargs)
        attempt.path = UnresolvablePath(attempt.path)
        return attempt

    monkeypatch.setattr(runner, "_begin_seed_attempt", begin)
    with pytest.raises(OSError, match="attempt path resolution failed"):
        harness.run(config=harness.config(resume=resume))
    assert harness.events == ["construction", "environment", "begin"]
    assert not harness.clock_reads
    if resume:
        path = Path(harness.attempts[-1].path) / "result.json"
        assert path.read_bytes() == previous_result
    else:
        assert not list(tmp_path.rglob("result.json"))
    harness.assert_locks_released()


def test_start_clock_failure_also_releases_reserved_attempt(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path)

    def broken_counter():
        raise RuntimeError("timer unavailable")

    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=broken_counter))
    with pytest.raises(RuntimeError, match="timer unavailable"):
        harness.run()
    assert harness.events == ["construction", "environment", "begin"]
    assert not list(tmp_path.rglob("result.json"))
    harness.assert_locks_released()


def test_peak_memory_failure_releases_lock_before_result_commit(monkeypatch, tmp_path):
    harness = _Harness(monkeypatch, tmp_path)

    def broken_peak(device):
        raise RuntimeError("CUDA memory statistics unavailable")

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", broken_peak)
    with pytest.raises(RuntimeError, match="CUDA memory statistics unavailable"):
        harness.run(config=harness.config(device="cuda"))
    assert list(tmp_path.rglob("evaluation.npz"))
    assert not list(tmp_path.rglob("result.json"))
    assert "finish" not in harness.events
    assert "progress" not in harness.events
    harness.assert_locks_released()


@pytest.mark.parametrize("primary", [OSError("checksum failure"), KeyboardInterrupt()])
def test_cleanup_failure_does_not_replace_escaping_primary_error(
    monkeypatch, tmp_path, primary
):
    harness = _Harness(monkeypatch, tmp_path)
    original_close = runner.RunAttempt.close

    def bad_checksum(path):
        raise primary

    def bad_close(attempt):
        original_close(attempt)
        raise OSError("secondary close failure")

    monkeypatch.setattr(runner, "file_sha256", bad_checksum)
    monkeypatch.setattr(runner.RunAttempt, "close", bad_close)
    with pytest.raises(type(primary)) as caught:
        harness.run()
    assert caught.value is primary
    assert any("OSError: secondary close failure" in note for note in primary.__notes__)
    assert not list(tmp_path.rglob("result.json"))
    harness.assert_locks_released()


def test_cleanup_failure_without_primary_error_is_reported(tmp_path):
    failure = OSError("close failed without an earlier error")

    def bad_close():
        raise failure

    attempt = run_artifacts.RunAttempt(
        path=tmp_path,
        fingerprint="fixture",
        previous_result=None,
        _lock=SimpleNamespace(close=bad_close),
    )
    with pytest.raises(OSError) as caught:
        runner._close_seed_attempt(attempt, None)
    assert caught.value is failure


@pytest.mark.parametrize("write_fails", [False, True])
def test_real_result_finalization_preserves_write_error_when_close_also_fails(
    monkeypatch, tmp_path, write_fails
):
    # Exercise the real runner -> finalization -> atomic_json/close chain. Only
    # the two I/O boundaries fail, rather than replacing RunAttempt.finish.
    task = SimpleNamespace(
        at_target_fidelity=lambda candidates: candidates,
        predict=lambda candidates: -2.0 - candidates[:, 0],
    )
    config = runner.SeedBenchmarkConfig(
        results_dir=tmp_path, seeds=(38,), candidate_budget=2, save_artifacts=True
    )
    write_error = OSError("original result.json write failure")
    close_error = OSError("secondary attempt close failure")
    original_json = run_artifacts.atomic_json
    original_close = run_artifacts.RunAttempt.close

    def write_result(path, value, **kwargs):
        if write_fails and path.name == "result.json":
            raise write_error
        original_json(path, value, **kwargs)

    def close_then_fail(attempt):
        original_close(attempt)
        raise close_error

    monkeypatch.setattr(run_artifacts, "atomic_json", write_result)
    monkeypatch.setattr(run_artifacts.RunAttempt, "close", close_then_fail)
    monkeypatch.setattr(runner, "capture_environment", lambda device: {})
    with pytest.raises(OSError) as caught:
        runner.run_method_seed_benchmark(
            task,
            _problem(),
            ["random_search"],
            reference_utility=np.array([-4.0, -1.0]),
            config=config,
        )
    assert caught.value is (write_error if write_fails else close_error)
    if write_fails:
        assert any(
            "secondary attempt close failure" in note for note in write_error.__notes__
        )
    assert bool(list(tmp_path.rglob("result.json"))) is not write_fails
    assert not (tmp_path / "method_seed_results.csv").exists()
    # Holding the propagated traceback must not hold either actual OS lock.
    assert caught.value.__traceback__ is not None
    acquired = run_artifacts._ProcessLock(next(tmp_path.rglob(".process.lock")))
    acquired.close()
    with runner.result_directory_lock(tmp_path):
        pass


@pytest.mark.parametrize("write_fails", [False, True])
@pytest.mark.parametrize(
    ("error_type", "fail_fast", "method_error_escapes"),
    [
        (RuntimeError, True, True),
        (KeyboardInterrupt, False, True),
        (RuntimeError, False, False),
        (SystemExit, False, True),
    ],
)
def test_real_finalization_close_does_not_replace_escaping_method_error(
    monkeypatch, tmp_path, write_fails, error_type, fail_fast, method_error_escapes
):
    method_error = error_type("method failed before finalization")
    write_error = OSError("result write failed after method error")
    close_error = OSError("close failed after method error")
    original_json = run_artifacts.atomic_json
    original_close = run_artifacts.RunAttempt.close

    class BrokenMethod(OfflineBBOMethod):
        def run(self, problem, context):
            raise method_error

        def optimize(self, problem, *, context, generator):
            raise AssertionError("the probe provides run() directly")

    def write_result(path, value, **kwargs):
        if write_fails and path.name == "result.json":
            raise write_error
        original_json(path, value, **kwargs)

    def close_then_fail(attempt):
        original_close(attempt)
        raise close_error

    monkeypatch.setattr(runner, "make_method", lambda *args, **kwargs: BrokenMethod())
    monkeypatch.setattr(runner, "capture_environment", lambda device: {})
    monkeypatch.setattr(run_artifacts, "atomic_json", write_result)
    monkeypatch.setattr(run_artifacts.RunAttempt, "close", close_then_fail)
    expected = (
        write_error
        if write_fails
        else (method_error if method_error_escapes else close_error)
    )
    with pytest.raises(type(expected)) as caught:
        runner.run_method_seed_benchmark(
            SimpleNamespace(),
            _problem(),
            ["random_search"],
            reference_utility=np.array([-4.0, -1.0]),
            config=runner.SeedBenchmarkConfig(
                results_dir=tmp_path,
                seeds=(38, 39),
                candidate_budget=2,
                save_artifacts=True,
                fail_fast=fail_fast,
            ),
        )
    assert caught.value is expected
    assert any(
        "close failed after method error" in note for note in caught.value.__notes__
    )
    attempt_paths = list(tmp_path.rglob("manifest.json"))
    assert len(attempt_paths) == 1  # Every propagated error stops the next seed.
    result_path = attempt_paths[0].with_name("result.json")
    assert result_path.exists() is not write_fails
    if not write_fails:
        saved = json.loads(result_path.read_text())
        assert saved["status"] == "failed"
        # Unhandled exits remain unclassified, as before the lock fix.
        assert saved["error_type"] == (
            None if error_type is SystemExit else error_type.__name__
        )
        assert saved["failure_stage"] == (
            None if error_type is SystemExit else "method"
        )
    assert (tmp_path / "method_seed_results.csv").exists() is (
        method_error_escapes and not write_fails
    )
    acquired = run_artifacts._ProcessLock(next(tmp_path.rglob(".process.lock")))
    acquired.close()
    with runner.result_directory_lock(tmp_path):
        pass


@pytest.mark.parametrize("failure", [None, "oracle"])
def test_non_artifact_runs_never_acquire_or_close_attempts(
    monkeypatch, tmp_path, failure
):
    harness = _Harness(monkeypatch, tmp_path, failure=failure)

    def forbidden(*args, **kwargs):
        raise AssertionError("an artifact-free run must not own an attempt lock")

    monkeypatch.setattr(runner.RunAttempt, "begin", forbidden)
    monkeypatch.setattr(runner.RunAttempt, "close", forbidden)
    monkeypatch.setattr(harness, "at_target_fidelity", lambda candidates: candidates)
    result = harness.run(config=replace(harness.config(), save_artifacts=False))
    assert result.per_seed.iloc[0]["status"] == ("failed" if failure else "success")
    assert not harness.attempts
    assert not list(tmp_path.rglob(".process.lock"))
