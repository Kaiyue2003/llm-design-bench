"""Release publishing checks use fake data and never train or load an oracle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llm_design_bench.evaluation import llmdm_protocol
from llm_design_bench.optimizers.base import OfflineBBOMethod
from llm_design_bench.tasks.data_recipes import DataRecipesTask

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_gpytorch_release_under_test", ROOT / "scripts/freeze_gpytorch_release.py"
)
assert SPEC is not None and SPEC.loader is not None
release_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_script)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _forbidden(*args, **kwargs):
    pytest.fail("publishing must not train a method or instantiate/query an oracle")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "project"
    previous = root / "experiments/llmdm_forward_v1"
    previous.mkdir(parents=True)
    contents = {
        "plan.json": b'{"old_plan": true}\r\n',
        "data/manifest.json": b'{"fake_manifest": true}\r\n',
        "data/visible.npz": b"fake visible arrays\x00\xff\r\n",
        "data/reference.npz": b"fake reference arrays\x00\xff\n",
    }
    for relative, content in contents.items():
        path = previous / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    original = {
        "data_manifest_id": "frozen-visible-identity",
        "upstream_commit": "b" * 40,
        "counts": {"source_rows": 7, "empty_histories": 2},
        "artifact_sha256": {path: _digest(value) for path, value in contents.items()},
    }
    _json(previous / "release.json", original)
    # Old runs are deliberately outside data; fresh releases must not inherit them.
    _json(previous / "formal/old-result.json", {"status": "success"})
    methods = [
        {"method_id": "ga_on_gp", "kwargs": {"gp_training_steps": 2}},
        {"method_id": "bo_qei", "kwargs": {"gp_training_steps": 2}},
    ]
    _json(root / "configs/llmdm_methods.forward_v1.json", methods)
    constraints = root / "configs/llmdm_gpytorch_runtime.txt"
    constraints.write_bytes(b"gpytorch==1.15.2\r\nlinear_operator==0.6.1\r\n")
    source = {"git_dirty": False, "git_commit": "a" * 40, "sha256": "c" * 64}
    bundle = SimpleNamespace(
        manifest_id=original["data_manifest_id"],
        reference_utility=np.arange(5, dtype=float),
        utility=np.arange(3, dtype=float),
        context=np.array([[20.0, 100.0], [60.0, 500.0], [1000.0, 19500.0]]),
    )
    calls = []

    def load(path, *, data_recipes_root):
        calls.append((path, data_recipes_root))
        return bundle

    monkeypatch.setattr(release_script, "load_data_manifest", load)
    monkeypatch.setattr(release_script, "package_source_identity", lambda: dict(source))
    monkeypatch.setattr(llmdm_protocol, "package_source_identity", lambda: dict(source))
    monkeypatch.setattr(
        release_script.importlib.metadata,
        "version",
        lambda name: release_script.GP_REQUIREMENTS[name],
    )
    monkeypatch.setattr(OfflineBBOMethod, "run", _forbidden)
    monkeypatch.setattr(DataRecipesTask, "__init__", _forbidden)
    monkeypatch.setattr(DataRecipesTask, "predict", _forbidden)
    return SimpleNamespace(
        root=root,
        upstream=tmp_path / "upstream",
        output=root / "experiments/llmdm_forward_gpytorch_v2",
        previous=previous,
        original=original,
        source=source,
        bundle=bundle,
        calls=calls,
        contents=contents,
        methods=methods,
    )


def _publish(environment):
    return release_script.publish(
        environment.root, environment.upstream, environment.output
    )


@pytest.mark.parametrize(
    "pins",
    [
        "gpytorch==1.15.1\nlinear_operator==0.6.1\n",
        "gpytorch==1.15.2\n",
        "gpytorch==1.15.2\ngpytorch==1.15.2\nlinear_operator==0.6.1\n",
    ],
)
def test_publish_rejects_constraints_that_disagree_with_declared_backend(
    environment, pins
):
    path = environment.root / "configs/llmdm_gpytorch_runtime.txt"
    path.write_text(pins, encoding="utf-8")
    with pytest.raises(ValueError, match="constraints differ"):
        _publish(environment)
    assert not environment.output.exists()


def test_publish_io_failure_does_not_leave_a_partial_release(environment, monkeypatch):
    def fail_copy(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(release_script.shutil, "copytree", fail_copy)
    with pytest.raises(OSError, match="simulated write failure"):
        _publish(environment)
    assert not environment.output.exists()
    assert not list(environment.output.parent.glob(f".{environment.output.name}-*"))


def test_publish_preserves_data_bytes_and_produces_consistent_fresh_plan(environment):
    env = environment
    result = _publish(env)
    saved = json.loads((env.output / "release.json").read_text("utf-8"))
    plan = json.loads((env.output / "plan.json").read_text("utf-8"))

    assert saved == result
    assert result["experiment_id"] == release_script.EXPERIMENT_ID
    assert result["plan_id"] == plan["plan_id"]
    assert result["data_manifest_id"] == env.original["data_manifest_id"]
    assert result["code_commit"] == env.source["git_commit"]
    assert result["package_source_sha256"] == env.source["sha256"]
    assert plan["package_source"] == env.source
    llmdm_protocol.validate_method_plan(plan, env.bundle)
    assert plan["experiment_id"] == release_script.EXPERIMENT_ID
    assert result["gp_backend"]["packages"] == release_script.GP_REQUIREMENTS
    assert result["gp_backend"]["methods"] == ["ga_on_gp", "bo_qei"]
    assert {entry["dtype"] for entry in plan["methods"]} == {"float64"}
    assert all(entry["kwargs"]["gp_training_steps"] == 2 for entry in plan["methods"])
    assert result["result_policy"] == "fresh_experiment_no_v1_result_reuse"
    assert result["counts"] == {
        "source_rows": 7,
        "empty_histories": 2,
        "usable_logged": 5,
        "main_visible": 3,
        "fixed_1b_visible": 1,
        "methods": 2,
    }
    assert env.calls == [(env.previous / "data", env.upstream.resolve())]
    for relative, content in env.contents.items():
        assert (env.previous / relative).read_bytes() == content
        if relative.startswith("data/"):
            assert (env.output / relative).read_bytes() == content
    for source, target in [
        ("llmdm_methods.forward_v1.json", "methods.json"),
        ("llmdm_gpytorch_runtime.txt", "runtime-constraints.txt"),
    ]:
        assert (env.root / "configs" / source).read_bytes() == (
            env.output / target
        ).read_bytes()
    assert (env.output / ".gitattributes").is_file()
    assert not (env.output / "formal").exists()
    for relative, expected in result["artifact_sha256"].items():
        assert _digest((env.output / relative).read_bytes()) == expected


@pytest.mark.parametrize("existing_directory", [False, True])
def test_publish_refuses_existing_output_without_overwriting(
    environment, existing_directory
):
    env = environment
    if existing_directory:
        env.output.mkdir()
        sentinel = env.output / "sentinel.bin"
    else:
        sentinel = env.output
    sentinel.write_bytes(b"existing release")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _publish(env)

    assert sentinel.read_bytes() == b"existing release"
    assert not env.calls


@pytest.mark.parametrize(
    "change",
    [
        {"git_dirty": True},
        {"git_dirty": None},
        {"git_commit": None},
        {"git_commit": ""},
    ],
)
def test_publish_requires_clean_committed_package(environment, change):
    environment.source.update(change)

    with pytest.raises(RuntimeError, match="clean, committed"):
        _publish(environment)

    assert not environment.output.exists()
    assert not environment.calls


@pytest.mark.parametrize("package", ["gpytorch", "linear_operator"])
def test_publish_rejects_mismatched_backend_dependencies(
    environment, monkeypatch, package
):
    versions = dict(release_script.GP_REQUIREMENTS)
    versions[package] = "0.0.0"
    monkeypatch.setattr(
        release_script.importlib.metadata, "version", versions.__getitem__
    )

    with pytest.raises(RuntimeError, match="backend versions differ"):
        _publish(environment)

    assert not environment.output.exists()
    assert not environment.calls


@pytest.mark.parametrize(
    "relative",
    ["plan.json", "data/manifest.json", "data/visible.npz", "data/reference.npz"],
)
def test_publish_rejects_corrupted_previous_artifacts(environment, relative):
    (environment.previous / relative).write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="original frozen artifact changed"):
        _publish(environment)

    assert not environment.output.exists()
    assert not environment.calls


@pytest.mark.parametrize("absolute_path", [False, True])
def test_publish_rejects_release_artifact_path_escape(environment, absolute_path):
    env = environment
    outside = env.root / "outside.bin"
    outside.write_bytes(b"outside original release")
    relative = str(outside.resolve()) if absolute_path else "../../outside.bin"
    env.original["artifact_sha256"][relative] = _digest(outside.read_bytes())
    _json(env.previous / "release.json", env.original)

    with pytest.raises(ValueError, match="original frozen artifact changed"):
        _publish(env)

    assert not env.output.exists()
    assert not env.calls


def test_publish_rejects_different_loaded_data_identity(environment):
    environment.bundle.manifest_id = "different-visible-identity"

    with pytest.raises(ValueError, match="original data identity differs"):
        _publish(environment)

    assert not environment.output.exists()


def test_publish_rejects_source_change_between_initial_check_and_plan(
    environment, monkeypatch
):
    original_freeze = release_script.freeze_method_plan

    def changed_freeze(*args, **kwargs):
        plan = original_freeze(*args, **kwargs)
        plan["package_source"]["sha256"] = "d" * 64
        return plan

    monkeypatch.setattr(release_script, "freeze_method_plan", changed_freeze)

    with pytest.raises(RuntimeError, match="checkout changed"):
        _publish(environment)

    assert not environment.output.exists()
