"""Offline integrity of the actual fresh release and its Colab entry point."""

import ast
import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest

from llm_design_bench.evaluation.llmdm_protocol import package_source_identity

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "experiments/llmdm_forward_gpytorch_v2"
PREVIOUS = ROOT / "experiments/llmdm_forward_v1"
RELEASE = json.loads((ASSETS / "release.json").read_text("utf-8"))


@pytest.mark.parametrize("relative", RELEASE["artifact_sha256"])
def test_new_release_artifact_hashes(relative):
    content = (ASSETS / relative).read_bytes()
    assert hashlib.sha256(content).hexdigest() == RELEASE["artifact_sha256"][relative]


def test_new_plan_is_frozen_from_current_gpytorch_source_with_unchanged_budgets():
    plan = json.loads((ASSETS / "plan.json").read_text("utf-8"))
    old_plan = json.loads((PREVIOUS / "plan.json").read_text("utf-8"))
    assert plan.pop("plan_id") == RELEASE["plan_id"] != old_plan["plan_id"]
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == RELEASE["plan_id"]
    assert (
        plan["experiment_id"] == RELEASE["experiment_id"] == "llmdm_forward_gpytorch_v2"
    )
    assert plan["package_source"]["git_dirty"] is False
    assert plan["package_source"]["git_commit"] == RELEASE["code_commit"]
    assert plan["package_source"]["sha256"] == RELEASE["package_source_sha256"]
    assert package_source_identity()["sha256"] == RELEASE["package_source_sha256"]
    assert (
        plan["data_manifest_id"]
        == old_plan["data_manifest_id"]
        == RELEASE["data_manifest_id"]
    )
    assert plan["methods"] == old_plan["methods"]
    assert plan["shared_settings"] == old_plan["shared_settings"]
    assert json.loads((ASSETS / "methods.json").read_text("utf-8")) == json.loads(
        (ROOT / "configs/llmdm_methods.forward_v1.json").read_text("utf-8")
    )


def test_new_release_reuses_only_identical_data_not_old_results():
    for name in ("manifest.json", "visible.npz", "reference.npz"):
        assert (ASSETS / "data" / name).read_bytes() == (
            PREVIOUS / "data" / name
        ).read_bytes()
    assert RELEASE["counts"] == {
        "source_rows": 472,
        "empty_histories": 18,
        "usable_logged": 454,
        "main_visible": 184,
        "fixed_1b_visible": 26,
        "methods": 19,
    }
    assert RELEASE["result_policy"] == "fresh_experiment_no_v1_result_reuse"
    assert not any((ASSETS / name).exists() for name in ("pilot", "formal", "results"))


def test_frozen_gp_library_pins_agree_with_package_and_lock():
    constraints = (ASSETS / "runtime-constraints.txt").read_text("utf-8")
    pins = {
        line.strip()
        for line in constraints.splitlines()
        if line and not line.startswith("#")
    }
    expected = {"gpytorch==1.15.2", "linear_operator==0.6.1"}
    assert pins == expected
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert expected.issubset(project["project"]["dependencies"])
    lock = tomllib.loads((ROOT / "uv.lock").read_text("utf-8"))
    locked = {entry["name"]: entry["version"] for entry in lock["package"]}
    assert locked["gpytorch"] == "1.15.2"
    assert locked["linear-operator"] == "0.6.1"
    assert RELEASE["gp_backend"]["packages"] == {
        "gpytorch": "1.15.2",
        "linear_operator": "0.6.1",
    }
    assert RELEASE["gp_backend"]["methods"] == ["ga_on_gp", "bo_qei"]


def test_new_notebook_uses_complete_pins_and_no_old_results():
    notebook = json.loads(
        (ROOT / "notebooks/LLMDM_GPyTorch_Colab.ipynb").read_text("utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    constants = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    for name in ("CODE_COMMIT", "ASSETS_COMMIT", "UPSTREAM_COMMIT"):
        assert re.fullmatch(r"[a-f0-9]{40}", constants[name])
    assert constants["CODE_COMMIT"] == RELEASE["code_commit"]
    assert constants["UPSTREAM_COMMIT"] == RELEASE["upstream_commit"]
    assert constants["CODE_COMMIT"] != constants["ASSETS_COMMIT"]
    assert constants["INCLUDE_FIXED_1B"] is False
    assert "REPLACE_" not in code
    assert "llmdm_forward_v1" not in code
