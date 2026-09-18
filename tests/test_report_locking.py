"""All shared report writers must cooperate across real OS processes."""

from __future__ import annotations

import multiprocessing
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from llm_design_bench import report_cli
from llm_design_bench.evaluation import run_artifacts, seed_runner, unified_report
from llm_design_bench.evaluation.seed_types import SeedBenchmarkConfig
from llm_design_bench.evaluation.task_specs import make_synthetic_task_spec


LEGACY = Path(__file__).resolve().parents[1] / "reference_results/publication"
PROCESS_TIMEOUT = 45


def _rows():
    return unified_report.load_legacy_publication_results(LEGACY).iloc[[0]].copy()


def _config(directory, *, save_artifacts=False, resume=False):
    return SeedBenchmarkConfig(
        results_dir=directory,
        experiment_id="report-lock-test",
        seeds=(0,),
        candidate_budget=2,
        save_artifacts=save_artifacts,
        resume=resume,
    )


def _run_suite(directory, *, save_artifacts=False, resume=False):
    return unified_report.run_benchmark_suite(
        [make_synthetic_task_spec("booth", logged_samples=8)],
        ["best_logged"],
        config=_config(directory, save_artifacts=save_artifacts, resume=resume),
    )


def _run_seed(directory, *, save_artifacts=False, write_results=True):
    trial = make_synthetic_task_spec("booth", logged_samples=8).trial_factory(0)
    return seed_runner.run_method_seed_benchmark(
        trial.evaluator_task,
        trial.problem,
        ["best_logged"],
        reference_utility=trial.reference_utility,
        config=_config(directory, save_artifacts=save_artifacts),
        write_results=write_results,
    )


def _report_bytes(directory):
    # A lock holder necessarily opens .suite.lock. Its contents are not a
    # report, and Windows does not allow reading its locked byte here.
    return {
        name: (directory / name).read_bytes()
        for name in unified_report.UNIFIED_REPORT_FILENAMES
        if (directory / name).is_file()
    }


def _paused_writer_process(directory, owner, connection, release):
    """Pause the real public entry point after it acquires the real OS lock."""
    original_lock = run_artifacts.result_directory_lock

    @contextmanager
    def announced_lock(path):
        with original_lock(path):
            connection.send(("locked", None))
            if not release.wait(PROCESS_TIMEOUT):
                raise TimeoutError("parent did not release the report writer")
            yield

    unified_report.result_directory_lock = announced_lock
    seed_runner.result_directory_lock = announced_lock
    try:
        if owner == "suite":
            _run_suite(directory, save_artifacts=True, resume=True)
        elif owner == "seed":
            _run_seed(directory)
        elif owner == "report":
            unified_report.write_unified_report(
                _rows(), directory, metadata={"lock_test_writer": "child"}
            )
        else:
            raise AssertionError(f"unknown owner: {owner}")
        connection.send(("done", None))
    except BaseException as error:
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


@contextmanager
def _paused_writer(directory, owner):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    release = context.Event()
    process = context.Process(
        target=_paused_writer_process,
        args=(directory, owner, child, release),
    )
    process.start()
    child.close()
    try:
        assert parent.poll(PROCESS_TIMEOUT), "writer did not acquire its lock"
        first = parent.recv()
        assert first == ("locked", None), first
        yield
        release.set()
        assert parent.poll(PROCESS_TIMEOUT), "writer did not finish after release"
        last = parent.recv()
        assert last == ("done", None), last
        process.join(PROCESS_TIMEOUT)
        assert process.exitcode == 0
    finally:
        release.set()
        process.join(PROCESS_TIMEOUT)
        if process.is_alive():
            process.terminate()
            process.join(PROCESS_TIMEOUT)
        parent.close()
        if not process.is_alive():
            process.close()


@pytest.mark.parametrize("entry", ["api", "cli-unified", "cli-legacy"])
def test_suite_excludes_report_api_and_cli_without_overwriting(tmp_path, entry):
    output = tmp_path / "report"
    result = _run_suite(output, save_artifacts=True)
    source = tmp_path / "input.csv"
    result.per_seed.to_csv(source, index=False)
    source_before = source.read_bytes()

    with _paused_writer(output, "suite"):
        before = _report_bytes(output)
        if entry == "api":
            with pytest.raises(RuntimeError, match="another worker"):
                unified_report.write_unified_report(_rows(), output)
        else:
            arguments = (
                ["from-unified", "--input-csv", str(source)]
                if entry == "cli-unified"
                else ["from-legacy", "--publication-dir", str(LEGACY)]
            )
            invocation = CliRunner().invoke(
                report_cli.app, [*arguments, "--results-dir", str(output)]
            )
            assert invocation.exit_code != 0, invocation.output
            assert isinstance(invocation.exception, RuntimeError)
            assert "another worker" in str(invocation.exception)
        assert _report_bytes(output) == before
        assert source.read_bytes() == source_before


def test_public_report_writers_are_mutually_exclusive_between_processes(tmp_path):
    output = tmp_path / "report"
    unified_report.write_unified_report(_rows(), output)

    with _paused_writer(output, "report"):
        before = _report_bytes(output)
        with pytest.raises(RuntimeError, match="another worker"):
            unified_report.write_unified_report(
                _rows(), output, metadata={"lock_test_writer": "contender"}
            )
        assert _report_bytes(output) == before

    # The child finished normally and released its lock; a subsequent writer
    # is allowed, so contention does not poison later report generation.
    unified_report.write_unified_report(_rows(), output)


def test_report_lock_blocks_plain_seed_csv_before_method_execution(
    tmp_path, monkeypatch
):
    output = tmp_path / "report"
    unified_report.write_unified_report(_rows(), output)

    def unexpected_method(*args, **kwargs):
        pytest.fail("a competing CSV writer must stop before method construction")

    monkeypatch.setattr(seed_runner, "make_method", unexpected_method)
    with _paused_writer(output, "report"):
        before = _report_bytes(output)
        with pytest.raises(RuntimeError, match="another worker"):
            _run_seed(output, save_artifacts=False)
        assert _report_bytes(output) == before


def test_plain_seed_csv_lock_blocks_public_report_writer(tmp_path):
    output = tmp_path / "report"
    unified_report.write_unified_report(_rows(), output)

    with _paused_writer(output, "seed"):
        before = _report_bytes(output)
        with pytest.raises(RuntimeError, match="another worker"):
            unified_report.write_unified_report(_rows(), output)
        assert _report_bytes(output) == before


@pytest.mark.parametrize("entry", ["suite", "seed"])
@pytest.mark.parametrize("save_artifacts", [False, True])
def test_internal_progress_and_final_writes_acquire_one_lock(
    tmp_path, monkeypatch, entry, save_artifacts
):
    acquired = []
    original_lock = run_artifacts.result_directory_lock

    @contextmanager
    def counted_lock(directory):
        acquired.append(Path(directory))
        with original_lock(directory):
            yield

    monkeypatch.setattr(unified_report, "result_directory_lock", counted_lock)
    monkeypatch.setattr(seed_runner, "result_directory_lock", counted_lock)
    output = tmp_path / "report"
    result = (
        _run_suite(output, save_artifacts=save_artifacts)
        if entry == "suite"
        else _run_seed(output, save_artifacts=save_artifacts)
    )

    assert acquired == [output]
    assert result.per_seed["status"].tolist() == ["success"]
    saved = pd.read_csv(output / "method_seed_results.csv")
    assert saved["status"].tolist() == ["success"]
    assert (output / "method_seed_summary.csv").is_file()


def test_memory_only_seed_execution_never_acquires_report_lock(tmp_path, monkeypatch):
    def unexpected_lock(*args, **kwargs):
        pytest.fail("write_results=False must not reacquire the suite report lock")

    monkeypatch.setattr(seed_runner, "result_directory_lock", unexpected_lock)
    output = tmp_path / "unused"
    result = _run_seed(output, write_results=False)
    assert result.per_seed["status"].tolist() == ["success"]
    assert not output.exists()


@pytest.mark.parametrize("entry", ["report", "suite", "seed"])
def test_writer_exception_releases_report_lock(tmp_path, monkeypatch, entry):
    output = tmp_path / "report"

    def fail(*args, **kwargs):
        raise RuntimeError("injected writer failure")

    with monkeypatch.context() as changes:
        with pytest.raises(RuntimeError, match="injected writer failure"):
            if entry == "report":
                changes.setattr(unified_report, "atomic_bytes", fail)
                unified_report.write_unified_report(_rows(), output)
            elif entry == "suite":
                task = replace(
                    make_synthetic_task_spec("booth", logged_samples=8),
                    trial_factory=fail,
                )
                unified_report.run_benchmark_suite(
                    [task], ["best_logged"], config=_config(output)
                )
            else:
                changes.setattr(seed_runner, "_write_seed_progress", fail)
                _run_seed(output)

    with run_artifacts.result_directory_lock(output):
        pass
    unified_report.write_unified_report(_rows(), output)


def test_public_shard_report_reads_and_merges_existing_rows_under_lock(
    tmp_path, monkeypatch
):
    output = tmp_path / "report"
    config = replace(_config(output, save_artifacts=True), required_seeds=(0, 1))
    task = make_synthetic_task_spec("booth", logged_samples=8)
    unified_report.run_benchmark_suite([task], ["best_logged"], config=config)
    incoming = unified_report.run_benchmark_suite(
        [task],
        ["best_logged"],
        config=replace(config, seeds=(1,), results_dir=tmp_path / "new-shard"),
    )
    original_lock = run_artifacts.result_directory_lock
    original_read = pd.read_csv
    original_merge = unified_report.merge_result_rows
    active = False
    events = []

    @contextmanager
    def observed_lock(directory):
        nonlocal active
        with original_lock(directory):
            active = True
            events.append("lock")
            try:
                yield
            finally:
                active = False
                events.append("unlock")

    def checked_read(path, *args, **kwargs):
        assert active, "existing report rows were read before acquiring the lock"
        assert Path(path) == output / unified_report.UNIFIED_RESULTS_FILENAME
        events.append("read")
        return original_read(path, *args, **kwargs)

    def checked_merge(*args, **kwargs):
        assert active, "existing and incoming rows were merged outside the lock"
        events.append("merge")
        return original_merge(*args, **kwargs)

    monkeypatch.setattr(unified_report, "result_directory_lock", observed_lock)
    monkeypatch.setattr(unified_report.pd, "read_csv", checked_read)
    monkeypatch.setattr(unified_report, "merge_result_rows", checked_merge)

    result = unified_report.write_unified_report(
        incoming.per_seed, output, config=config
    )

    assert events == ["lock", "read", "merge", "unlock"]
    assert not active
    assert sorted(result.per_seed["method_seed"].tolist()) == [0, 1]
    assert result.per_seed["status"].tolist() == ["success", "success"]


def test_invalid_standalone_metadata_is_rejected_before_creating_directory(
    tmp_path, monkeypatch
):
    def unexpected_lock(*args, **kwargs):
        pytest.fail("invalid standalone report metadata must not create a lock file")

    monkeypatch.setattr(unified_report, "result_directory_lock", unexpected_lock)
    output = tmp_path / "unused"

    with pytest.raises(ValueError, match="reserved field"):
        unified_report.write_unified_report(
            _rows(), output, metadata={"schema_version": 999}
        )

    assert not output.exists()
