"""Only current runners are installed; historical artifacts remain readable."""

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import llm_design_bench
from llm_design_bench import llmdm_cli
from llm_design_bench.evaluation.logged_data import LoggedDatasetView
from llm_design_bench.types import CandidateBatch


ROOT = Path(__file__).resolve().parents[1]
CURRENT_COMMANDS = {
    "llm-design-bench": "llm_design_bench:main",
    "llm-design-bench-llmdm": "llm_design_bench.llmdm_cli:main",
    "llm-design-bench-suite": "llm_design_bench.suite_cli:main",
    "llm-design-bench-report": "llm_design_bench.report_cli:main",
}


def test_installed_command_contract_and_ci_use_current_runners():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert project["project"]["scripts"] == CURRENT_COMMANDS
    workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
    for command in CURRENT_COMMANDS:
        assert f"{command} --help" in workflow
    for suffix in ("offline", "publication", "synthetic"):
        assert f"llm-design-bench-{suffix} --help" not in workflow


@pytest.mark.parametrize("command,target", CURRENT_COMMANDS.items())
def test_current_entrypoint_help_exits_without_running(
    command, target, monkeypatch, capsys
):
    module_name, attribute = target.split(":")
    entrypoint = getattr(importlib.import_module(module_name), attribute)
    monkeypatch.setattr(sys, "argv", [command, "--help"])
    with pytest.raises(SystemExit) as error:
        entrypoint()
    assert error.value.code == 0
    assert "Usage:" in capsys.readouterr().out


def test_primary_entrypoint_delegates_to_frozen_protocol(monkeypatch):
    calls = []
    monkeypatch.setattr(llmdm_cli, "app", lambda: calls.append("formal"))
    llm_design_bench.main()
    assert calls == ["formal"]


@pytest.mark.parametrize("legacy_option", ["--queries", "--logged-model-scale"])
def test_formal_entrypoint_does_not_reinterpret_legacy_options(legacy_option):
    result = CliRunner().invoke(llmdm_cli.app, [legacy_option, "1"])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "module_name",
    [
        "cli",
        "offline_cli",
        "publication_cli",
        "synthetic_cli",
        "evaluation.runner",
        "evaluation.offline_runner",
        "evaluation.publication_runner",
        "evaluation.synthetic_runner",
        "evaluation.report",
        "optimizers.offline_utils",
    ],
)
def test_retired_execution_modules_are_not_shipped(module_name):
    assert importlib.util.find_spec(f"llm_design_bench.{module_name}") is None


def test_logged_view_keeps_selected_data_and_delegated_metadata():
    all_designs = CandidateBatch.at_fidelity(
        np.array([[0.2, 0.8], [0.7, 0.3]]), 1000, 19500
    )
    all_utility = np.array([-2.0, -1.0])
    task = SimpleNamespace(
        logged_x=all_designs, logged_y=all_utility, target_model_scale=1000
    )
    selected_designs = CandidateBatch.at_fidelity(all_designs.mixtures[:1], 1000, 19500)
    selected_utility = all_utility[:1]
    view = LoggedDatasetView(task, selected_designs, selected_utility)
    assert view.logged_x is selected_designs
    assert view.logged_y is selected_utility
    assert view.target_model_scale == 1000
    assert task.logged_x is all_designs
    assert task.logged_y is all_utility


def test_nonplotting_import_preserves_backend_without_loading_pyplot(tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    env["PYTHONPATH"] = str(ROOT / "src")
    code = """
import sys
import matplotlib
matplotlib.use('svg')
import llm_design_bench.evaluation.run_artifacts
import llm_design_bench.evaluation.task_specs
assert matplotlib.get_backend().lower() == 'svg'
assert 'matplotlib.pyplot' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
