"""Exercise the typed queue/read-only verification split without real training."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_colab_batch import (
    _job,
    _save_job,
    batch,
    plan as plan,
    runner as runner,
    support,
)

ROOT = Path(__file__).resolve().parents[1]


def test_queue_types_preserve_identity_json_and_declared_fields(runner):
    spec = importlib.util.spec_from_file_location(
        "_colab_types_contract", ROOT / "scripts/colab_types.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = _job(runner)
    expected = {
        "plan_id": runner.plan["plan_id"],
        "run_id": "offline_mlp",
        "setting": "multi_scale",
        "phase": "pilot",
        "seed": 0,
        "device": "cuda",
        "oracle_device": "cpu",
    }
    assert module.JobIdentity.__required_keys__ == expected.keys()
    assert not module.JobIdentity.__optional_keys__
    assert type(job) is dict
    assert support._canonical(job) == support._canonical(expected)
    assert json.loads(json.dumps(job)) == expected


def test_pending_verification_does_not_build_task_or_modify_state(runner, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a pending row does not need a task, oracle, or process")

    monkeypatch.setattr(runner, "_expected_logical", forbidden)
    monkeypatch.setattr(batch, "run_job", forbidden)
    before = set(runner.state.parent.rglob("*"))
    assert runner._verified_row(_job(runner)) is None
    assert set(runner.state.parent.rglob("*")) == before


def test_journal_block_prevents_artifact_verifier_from_running(runner, monkeypatch):
    def interrupted(job):
        raise RuntimeError("prior job was interrupted")

    def forbidden(**kwargs):
        pytest.fail("the completion journal must be checked first")

    monkeypatch.setattr(runner, "_journal", interrupted)
    monkeypatch.setattr(batch, "verified_result_row", forbidden)
    with pytest.raises(RuntimeError, match="prior job was interrupted"):
        runner._verified_row(_job(runner))


def test_saved_verification_builds_expected_configuration_once(runner, monkeypatch):
    job = _job(runner)
    _, saved = _save_job(runner, job)
    original = runner._expected_logical
    calls = []

    def expected(identity):
        calls.append(identity)
        return original(identity)

    monkeypatch.setattr(runner, "_expected_logical", expected)
    verified = runner._verified_row(job)
    assert verified == saved
    assert calls == [job]


def test_pilot_report_retains_only_cost_diagnostics_and_no_scores(runner):
    job = _job(runner)
    _save_job(runner, job)
    [report] = runner.pilot_report(methods=[job["run_id"]])
    assert set(report) == {
        "setting",
        "run_id",
        "status",
        "train_size",
        "candidate_budget",
        "device",
        "dtype",
        "method_seconds",
        "evaluation_seconds",
        "total_seconds",
        "peak_gpu_memory_bytes",
        "unique_candidate_count",
        "training_summary",
        "diagnostics",
        "peak_gpu_memory_mib",
        "artifact_checks",
        "finite_numeric_diagnostics",
    }
    assert report["status"] == "verified"
    assert report["training_summary"] == {"final_standardized_mse": 0.01}
    assert report["peak_gpu_memory_mib"] == 1024 / 2**20


def test_verification_module_has_no_queue_or_process_dependency():
    path = ROOT / "scripts/colab_verification.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        item.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for item in node.names
    }
    assert not imports.intersection({"colab_batch", "colab_support", "subprocess"})
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"run_job", "snapshot_tree", "atomic_json"}
        for node in ast.walk(tree)
    )


def test_current_colab_helpers_import_from_a_complete_checkout(tmp_path):
    code = """
import pathlib
import sys
scripts, source = map(pathlib.Path, sys.argv[1:])
sys.path[:0] = [str(scripts), str(source)]
import colab_batch
import colab_verification
assert pathlib.Path(colab_batch.__file__).resolve() == scripts / 'colab_batch.py'
assert pathlib.Path(colab_verification.__file__).resolve() == scripts / 'colab_verification.py'
assert colab_batch.verified_result_row is colab_verification.verified_result_row
"""
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl-cache")}
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "scripts"), str(ROOT / "src")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", ["colab_types", "colab_verification"])
def test_preloaded_companion_from_another_checkout_is_rejected(
    tmp_path, monkeypatch, name
):
    spec = importlib.util.spec_from_file_location(
        "_colab_v2_module_boundaries", ROOT / "scripts/colab_v2.py"
    )
    assert spec is not None and spec.loader is not None
    v2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)
    other = tmp_path / "other-checkout" / "scripts" / f"{name}.py"
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(other)))
    with pytest.raises(RuntimeError, match=f"{name} is loaded from another checkout"):
        v2._check_loaded_sources(ROOT)
