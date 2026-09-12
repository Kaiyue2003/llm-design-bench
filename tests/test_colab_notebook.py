"""Validate launcher cells without Colab, Drive, network or model training."""

import ast
import copy
import json
import platform
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/LLMDM_Colab.ipynb"
NOTEBOOK = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
CELLS = {cell["id"]: cell for cell in NOTEBOOK["cells"]}
PLAN = json.loads(
    (ROOT / "experiments/llmdm_forward_v1/plan.json").read_text(encoding="utf-8")
)


def _source(cell_id):
    return "".join(CELLS[cell_id]["source"])


def _execute(cell_id, namespace):
    # These repository-owned cells run only with the fake dependencies below.
    exec(compile(_source(cell_id), f"{NOTEBOOK_PATH}:{cell_id}", "exec"), namespace)  # noqa: S102


def test_notebook_is_clean_and_all_code_cells_compile():
    assert NOTEBOOK["nbformat"] == 4
    assert len(CELLS) == len(NOTEBOOK["cells"])
    for cell in CELLS.values():
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cell["id"])
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), str(NOTEBOOK_PATH), "exec")


def test_checkout_is_pinned_and_preparation_never_freezes_or_trains():
    tree = ast.parse(_source("llmdm-01"))
    constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert re.fullmatch("[0-9a-f]{40}", constants["RELEASE_REVISION"])
    assert constants["UPSTREAM_REVISION"] == "37269969a0957448d51622e0c083977bc5d260e8"
    code = "\n".join(_source(f"llmdm-{i:02d}") for i in (1, 3, 5, 7, 9))
    calls = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)]
    forbidden = {"run_job", "freeze_method_plan", "fit", "propose", "predict"}
    for node in calls:
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        assert name not in forbidden
    assert 'BACKUPS / "snapshots"' in _source("llmdm-07")
    assert 'release["artifact_sha256"]' in _source("llmdm-05")


@pytest.fixture
def launcher(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    calls = []

    def fake_job(command, local, backup, identity, **kwargs):
        calls.append((command, local, backup, identity, kwargs))
        return {"status": "success"}

    namespace = {
        "Path": Path,
        "json": json,
        "sys": sys,
        "platform": platform,
        "torch": SimpleNamespace(
            version=SimpleNamespace(cuda="test-cuda"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 123)),
            cuda=SimpleNamespace(
                is_available=lambda: True, get_device_name=lambda index: "test GPU"
            )
        ),
        "importlib": SimpleNamespace(
            metadata=SimpleNamespace(version=lambda name: "test")
        ),
        "plan": copy.deepcopy(PLAN),
        "STATE": state,
        "BACKUPS": tmp_path / "fake-drive",
        "ASSETS": ROOT / "experiments/llmdm_forward_v1",
        "UPSTREAM": tmp_path / "upstream",
        "run_job": fake_job,
    }
    _execute("llmdm-09", namespace)
    return namespace, calls


def test_default_execution_gate_has_no_filesystem_or_job_side_effects(launcher):
    namespace, calls = launcher
    assert namespace["METHOD_ID"] == "offline_mlp"
    assert namespace["PHASE"] == "pilot"
    assert namespace["SEED"] == 0
    assert namespace["RUN_JOB"] is False
    _execute("llmdm-11", namespace)
    assert not calls
    assert not list(namespace["STATE"].iterdir())


@pytest.mark.parametrize("phase,seed", [("pilot", 0), ("formal", 38), ("formal", 45)])
def test_explicit_launch_passes_single_frozen_job_and_absolute_paths(
    launcher, phase, seed
):
    namespace, calls = launcher
    namespace.update(RUN_JOB=True, PHASE=phase, SEED=seed, CONFIRM_PILOT_REVIEWED=True)
    _execute("llmdm-11", namespace)
    assert len(calls) == 1
    command, state, backup, identity, options = calls[0]
    assert command[:5] == [
        sys.executable,
        "-u",
        "-m",
        "llm_design_bench.llmdm_cli",
        "run",
    ]
    assert command.count("--run-id") == command.count("--seed") == 1
    assert command[command.index("--seed") + 1] == str(seed)
    assert command[command.index("--phase") + 1] == phase
    assert command[command.index("--oracle-device") + 1] == "cpu"
    assert command[command.index("--torch-threads") + 1] == "1"
    assert "--resume" in command
    for flag in ("--data-recipes-root", "--data-bundle", "--plan", "--results-dir"):
        assert Path(command[command.index(flag) + 1]).is_absolute()
    assert ("--pilot-results" in command) is (phase == "formal")
    assert identity == {
        "plan_id": PLAN["plan_id"],
        "run_id": "offline_mlp",
        "setting": "multi_scale",
        "phase": phase,
        "seed": seed,
        "device": "cuda",
        "oracle_device": "cpu",
    }
    assert state == namespace["STATE"] and backup == namespace["BACKUPS"]
    assert options == {"snapshot_interval": 60, "infrastructure_retry_reason": None}


def test_formal_requires_human_pilot_review_before_dispatch(launcher):
    namespace, calls = launcher
    namespace.update(RUN_JOB=True, PHASE="formal", SEED=38)
    with pytest.raises(RuntimeError, match="CONFIRM_PILOT_REVIEWED"):
        _execute("llmdm-11", namespace)
    assert not calls


def test_environment_changes_need_review_and_device_cannot_change(launcher):
    namespace, calls = launcher
    namespace["RUN_JOB"] = True
    _execute("llmdm-11", namespace)
    assert len(calls) == 1
    namespace["torch"].cuda.get_device_name = lambda index: "other GPU"
    with pytest.raises(RuntimeError, match="GPU"):
        _execute("llmdm-11", namespace)
    assert len(calls) == 1
    namespace["ALLOW_ENVIRONMENT_CHANGE"] = True
    _execute("llmdm-11", namespace)
    assert len(calls) == 2
    namespace["DEVICE"] = "cpu"
    with pytest.raises(RuntimeError):
        _execute("llmdm-11", namespace)
    assert len(calls) == 2


def test_infrastructure_reason_reaches_both_journal_and_evaluator(launcher):
    namespace, calls = launcher
    namespace.update(RUN_JOB=True, INFRASTRUCTURE_RETRY_REASON="  VM disconnected  ")
    _execute("llmdm-11", namespace)
    command, _, _, _, options = calls[0]
    assert (
        command[command.index("--infrastructure-retry-reason") + 1] == "VM disconnected"
    )
    assert options["infrastructure_retry_reason"] == "VM disconnected"
