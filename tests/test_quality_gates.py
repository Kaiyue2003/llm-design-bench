"""The documented quality gates must reject errors, not merely run commands."""

from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "pyproject.toml"


def _check_tool(module, *arguments, source=None):
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        input=source,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _check_types(tmp_path, source, *companion_sources):
    probe = tmp_path / "quality_probe.py"
    probe.write_text(source, encoding="utf-8")
    return _check_tool(
        "mypy",
        "--config-file",
        str(CONFIG),
        # Share within this pytest run, but isolate probes from other runs and
        # CLI checks targeting a different platform (Windows versus Linux).
        "--cache-dir",
        str(tmp_path.parent / "quality-mypy-cache"),
        str(probe),
        *(str(ROOT / path) for path in companion_sources),
    )


def test_quality_tools_are_pinned_in_both_install_paths_and_lock():
    project = tomllib.loads(CONFIG.read_text("utf-8"))
    extras = project["project"]["optional-dependencies"]["dev"]
    group = project["dependency-groups"]["dev"]
    locked = {
        package["name"]: package["version"]
        for package in tomllib.loads((ROOT / "uv.lock").read_text("utf-8"))["package"]
    }
    for name in ("ruff", "mypy", "pandas-stubs"):
        pin = f"{name}=={locked[name]}"
        assert pin in extras
        assert pin in group
    assert project["tool"]["ruff"]["required-version"] == f"=={locked['ruff']}"
    assert project["tool"]["mypy"]["strict"] is True
    for path in project["tool"]["mypy"]["files"]:
        assert (ROOT / path).is_file(), path


@pytest.mark.parametrize(
    "command",
    [
        "python -m ruff check src scripts examples tests setup.py",
        "python -m ruff format --check src scripts examples tests setup.py",
        "python -m mypy",
    ],
)
def test_ci_and_contributor_commands_agree(command):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text("utf-8")
    quality_job = workflow.split("  quality:\n", 1)[1].split("\n  test:", 1)[0]
    assert f"run: {command}\n" in quality_job
    assert command in (ROOT / "CONTRIBUTING.md").read_text("utf-8")


def test_lint_gate_rejects_undefined_names():
    result = _check_tool(
        "ruff",
        "check",
        "--config",
        str(CONFIG),
        "--stdin-filename",
        "src/quality_probe.py",
        "-",
        source="value = undefined_name\n",
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "F821" in result.stdout


@pytest.mark.parametrize("source,expected_code", [("value=1\n", 1), ("value = 1\n", 0)])
def test_format_gate_checks_without_rewriting_files(source, expected_code):
    result = _check_tool(
        "ruff",
        "format",
        "--check",
        "--config",
        str(CONFIG),
        "--stdin-filename",
        "src/quality_probe.py",
        "-",
        source=source,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr


@pytest.mark.parametrize(
    "directory", ["experiments", "notebooks", "reference_results", "configs"]
)
def test_format_gate_excludes_explicit_frozen_paths(directory):
    result = _check_tool(
        "ruff",
        "format",
        "--check",
        "--config",
        str(CONFIG),
        "--stdin-filename",
        str(ROOT / directory / "quality_probe.py"),
        "-",
        source="value=1\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("seed,expected_code", [("38", 0), ('"invalid"', 1)])
def test_type_gate_checks_the_real_run_context_contract(tmp_path, seed, expected_code):
    code = (
        "from llm_design_bench.problem import RunContext\n"
        f"context = RunContext(method_seed={seed}, candidate_budget=128)\n"
    )
    result = _check_types(tmp_path, code)
    assert result.returncode == expected_code, result.stdout + result.stderr
    if expected_code:
        assert "[arg-type]" in result.stdout


def test_type_gate_rejects_untyped_functions(tmp_path):
    result = _check_types(tmp_path, "def untyped(value):\n    return value\n")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[no-untyped-def]" in result.stdout


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_array_annotations_preserve_supported_input_dtypes(tmp_path, dtype):
    code = f"""
import numpy as np
from llm_design_bench.tasks.base import Task
from llm_design_bench.types import CandidateBatch

designs = np.ones((2, 3), dtype=np.{dtype})
fidelity = np.ones(2, dtype=np.{dtype})
batch = CandidateBatch(designs, fidelity, fidelity)

def at_target(task: Task) -> CandidateBatch:
    return task.at_target_fidelity(designs)
"""
    result = _check_types(tmp_path, code)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module,record,field,value",
    [
        (
            "llm_design_bench.evaluation.plan_types",
            "MethodPlanEntry",
            "dtype",
            '"float16"',
        ),
        (
            "llm_design_bench.evaluation.plan_types",
            "PackageSourceIdentity",
            "git_dirty",
            '"false"',
        ),
        (
            "llm_design_bench.evaluation.seed_types",
            "SeedResultRow",
            "method_seed",
            '"38"',
        ),
        (
            "llm_design_bench.evaluation.artifact_types",
            "RunEnvironment",
            "torch_threads",
            '"one"',
        ),
        ("scripts.colab_types", "JobIdentity", "phase", '"validation"'),
        ("scripts.colab_types", "DispatchCompletion", "returncode", '"zero"'),
    ],
)
def test_record_contracts_reject_wrong_fixed_fields(
    tmp_path, module, record, field, value
):
    code = (
        f"from {module} import {record}\n"
        f"def modify(record: {record}) -> None:\n"
        f'    record["{field}"] = {value}\n'
    )
    result = _check_types(tmp_path, code)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[typeddict-item]" in result.stdout


@pytest.mark.parametrize(
    "identity_type,expected_code",
    [
        ("dict[str, str]", 0),
        ("dict[str, object]", 0),
        ("JobIdentity", 0),
        ("Mapping[str, object]", 1),
        ("MappingProxyType[str, object]", 1),
    ],
)
def test_dispatch_identity_type_matches_dict_only_runtime_contract(
    tmp_path, identity_type, expected_code
):
    source = f"""
from collections.abc import Mapping
from types import MappingProxyType
from colab_support import run_job
from colab_types import JobIdentity

def launch(identity: {identity_type}) -> None:
    run_job(["python", "-c", "pass"], "local", "backups", identity)
"""
    # These are standalone scripts, not installed package modules. Give mypy
    # their real top-level import roots, matching the current Colab checkout.
    result = _check_types(
        tmp_path, source, "scripts/colab_support.py", "scripts/colab_types.py"
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    if expected_code:
        assert "[arg-type]" in result.stdout
        assert 'Argument 4 to "run_job"' in result.stdout
