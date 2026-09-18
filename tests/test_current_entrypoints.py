"""The default CLI and installed scripts expose only the current workflow."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
from typer.testing import CliRunner

import llm_design_bench
from llm_design_bench import llmdm_cli, report_cli
from llm_design_bench.evaluation.formal_methods import FORMAL_METHOD_IDS


ROOT = Path(__file__).resolve().parents[1]


def test_package_scripts_have_one_training_workflow():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {
        "llm-design-bench": "llm_design_bench.llmdm_cli:main",
        "llm-design-bench-llmdm": "llm_design_bench.llmdm_cli:main",
        "llm-design-bench-report": "llm_design_bench.report_cli:main",
    }


def test_package_main_delegates_to_formal_cli(monkeypatch):
    calls = []
    monkeypatch.setattr(llmdm_cli, "main", lambda: calls.append("formal"))
    llm_design_bench.main()
    assert calls == ["formal"]


def test_module_invocation_lists_current_methods_without_writing(tmp_path):
    environment = dict(os.environ)
    # Explicit checkout path, not whichever editable distribution is installed.
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "llm_design_bench", "methods"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (
        tuple(row["method_id"] for row in json.loads(result.stdout))
        == FORMAL_METHOD_IDS
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "module",
    [
        "cli",
        "offline_cli",
        "publication_cli",
        "synthetic_cli",
        "evaluation.runner",
        "evaluation.offline_runner",
        "evaluation.publication_runner",
        "evaluation.synthetic_runner",
        "optimizers.legacy",
        "optimizers.unified_baselines",
        "optimizers.offline_utils",
    ],
)
def test_retired_modules_are_not_importable(module):
    assert importlib.util.find_spec(f"llm_design_bench.{module}") is None


def test_formal_help_does_not_expose_old_run_options():
    result = CliRunner().invoke(llmdm_cli.app, ["--help"])
    assert result.exit_code == 0
    for command in ("prepare", "freeze", "run", "methods"):
        assert command in result.output
    for command in ("benchmark", "publication", "--reference-queries"):
        assert command not in result.output


def test_report_keeps_current_subcommand_and_rejects_retired_converter(tmp_path):
    runner = CliRunner()
    help_result = runner.invoke(report_cli.app, ["--help"])
    assert help_result.exit_code == 0
    assert "from-unified" in help_result.output
    assert "from-legacy" not in help_result.output
    result = runner.invoke(
        report_cli.app,
        ["from-legacy", "--publication-dir", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert not list(tmp_path.iterdir())
