"""V2 setup/workflow tests: no training, network calls or real Drive mounting."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_colab_v2_test", ROOT / "scripts/colab_v2.py"
)
assert SPEC is not None and SPEC.loader is not None
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)
CODE, UPSTREAM, PLAN = "a" * 40, "b" * 40, "c" * 64


@pytest.fixture
def release_files(tmp_path):
    assets = tmp_path / "assets"
    artifacts = {}
    for relative in v2.REQUIRED_ARTIFACTS:
        path = assets / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "gpytorch==1.15.2\nlinear_operator==0.6.1\n"
            if relative.endswith(".txt")
            else "{}",
            encoding="utf-8",
        )
        artifacts[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    release = {
        "schema_version": 1,
        "experiment_id": v2.EXPERIMENT,
        "code_commit": CODE,
        "upstream_commit": UPSTREAM,
        "plan_id": PLAN,
        "data_manifest_id": "d" * 64,
        "package_source_sha256": "e" * 64,
        "counts": dict(v2.EXPECTED_COUNTS),
        "artifact_sha256": artifacts,
    }
    (assets / "release.json").write_text(json.dumps(release), encoding="utf-8")
    return assets, release


def validate(assets):
    return v2.validate_release(assets, code_commit=CODE, upstream_commit=UPSTREAM)


def test_release_verifies_all_required_artifacts(release_files):
    assets, expected = release_files
    assert validate(assets) == expected
    (assets / "data/visible.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact mismatch"):
        validate(assets)


@pytest.mark.parametrize(
    "relative",
    [
        "../outside",
        "/outside",
        "C:/outside",
        "data\\outside",
        "data/../x",
        "./plan.json",
        "data//x",
        "",
    ],
)
def test_release_rejects_unsafe_paths(release_files, relative):
    assets, release = release_files
    release["artifact_sha256"][relative] = "0" * 64
    (assets / "release.json").write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ValueError):
        validate(assets)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(experiment_id="llmdm_forward_v1"),
        lambda r: r.update(code_commit="f" * 40),
        lambda r: r.update(plan_id="../escape"),
        lambda r: r["counts"].update(methods=18),
        lambda r: r["artifact_sha256"].pop("runtime-constraints.txt"),
    ],
)
def test_release_rejects_wrong_identity_or_missing_files(release_files, mutation):
    assets, release = release_files
    mutation(release)
    (assets / "release.json").write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ValueError):
        validate(assets)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "gpytorch>=1.15",
        "-r other.txt",
        "gpytorch==1.15\ngpytorch==1.14",
        "gpytorch @ https://example.com/a.whl",
    ],
)
def test_constraints_require_exact_unique_pins(tmp_path, content):
    path = tmp_path / "constraints.txt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        v2.read_runtime_constraints(path)


def test_constraints_accept_comments_and_canonical_names(tmp_path):
    path = tmp_path / "constraints.txt"
    path.write_text(
        "# frozen\n\nGPyTorch==1.15.2\nlinear-operator==0.6.1\n", encoding="utf-8"
    )
    assert v2.read_runtime_constraints(path) == {
        "gpytorch": "1.15.2",
        "linear_operator": "0.6.1",
    }


def test_checkout_does_not_reset_wrong_existing_directory(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(v2.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(
        v2.subprocess,
        "check_output",
        lambda argv, **kw: "wrong" if argv[-1] == "HEAD" else "",
    )
    with pytest.raises(RuntimeError, match="checkout/version/origin"):
        v2.checkout_exact(
            "https://github.com/Kaiyue2003/llm-design-bench.git", tmp_path, CODE
        )
    assert calls == []


@pytest.mark.parametrize("revision", ["main", "a" * 7, "REPLACE_CODE_COMMIT"])
def test_checkout_rejects_unpinned_revisions(tmp_path, revision):
    with pytest.raises(ValueError, match="pinned"):
        v2.checkout_exact(
            "https://github.com/Kaiyue2003/llm-design-bench.git", tmp_path, revision
        )


@pytest.fixture
def installer(release_files, monkeypatch):
    assets, _ = release_files
    versions = {
        "torch": "2.11.0+cu128",
        "gpytorch": "1.15.2",
        "linear_operator": "0.6.1",
    }
    monkeypatch.setattr(v2, "MODULE_DISTRIBUTIONS", {})
    monkeypatch.setattr(v2, "_check_loaded_sources", lambda _: None)
    monkeypatch.setattr(v2.importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(v2.importlib, "import_module", lambda name: SimpleNamespace())
    monkeypatch.setattr(v2.sys, "path", list(sys.path))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, "No broken requirements found.", ""
        )

    monkeypatch.setattr(v2.subprocess, "run", run)
    return assets, versions, calls


def test_install_preserves_torch_and_adds_src_after_success(installer):
    assets, _, calls = installer
    repo = assets.parent / "repo"
    v2.install_environment(repo, assets)
    assert calls[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert calls[0].count("-c") == 2
    assert str(assets / "runtime-constraints.txt") in calls[0]
    assert calls[0][-2:] == ["-e", str(repo.resolve())]
    assert str(repo.resolve() / "src") in sys.path


def test_install_does_not_ignore_unknown_pip_conflicts(installer, monkeypatch):
    assets, _, calls = installer

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, int(command[-1] == "check"), "otherpackage requires missing", ""
        )

    monkeypatch.setattr(v2.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="pip check failed"):
        v2.install_environment(assets.parent / "repo", assets)
    assert len(calls) == 2


def test_install_repairs_only_known_jedi_issue_then_rechecks(installer, monkeypatch):
    assets, _, calls = installer
    checks = []

    def run(command, **kwargs):
        calls.append(command)
        if command[-1] == "check":
            checks.append(True)
            return subprocess.CompletedProcess(
                command,
                int(len(checks) == 1),
                "ipython 7.34.0 requires jedi, which is not installed."
                if len(checks) == 1
                else "No broken requirements found.",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(v2.subprocess, "run", run)
    v2.install_environment(assets.parent / "repo", assets)
    assert len(checks) == 2
    assert "jedi==0.19.2" in calls[2]


def test_stale_loaded_gp_remains_blocked_on_repeated_attempt(monkeypatch):
    monkeypatch.setattr(v2, "MODULE_DISTRIBUTIONS", {"gpytorch": "gpytorch"})
    monkeypatch.setitem(sys.modules, "gpytorch", SimpleNamespace(__version__="1.14"))
    monkeypatch.setattr(v2.importlib.metadata, "version", lambda _: "1.15.2")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="restart Python"):
            v2._check_loaded_versions()


class FakeBatch:
    def __init__(self, state):
        self.state = state
        self.plan = {"plan_id": PLAN, "methods": [{"run_id": "one"}, {"run_id": "two"}]}
        self.events = []
        self.report_status = "verified"
        self.blocked = False

    def preview(self, methods=None, settings=("multi_scale",), phase="pilot"):
        self.events.append(("preview", phase))
        methods = methods or ["one", "two"]
        return [
            {
                "setting": setting,
                "run_id": method,
                "status": "blocked" if self.blocked else "pending",
            }
            for setting in settings
            for method in methods
            for seed in ([0] if phase == "pilot" else range(8))
        ]

    def run(self, **kwargs):
        self.events.append(("run", kwargs))
        return [kwargs]

    def pilot_report(self, methods=None, settings=("multi_scale",)):
        return [
            {
                "setting": setting,
                "run_id": method,
                "status": self.report_status,
                "training_summary": {},
                "diagnostics": {},
            }
            for setting in settings
            for method in methods
        ]


def token(methods=("one", "two"), settings=("multi_scale",)):
    scope = {
        "plan_id": PLAN,
        "reviewed_pilots": sorted((s, m) for s in settings for m in methods),
        "formal_seeds": list(range(38, 46)),
    }
    digest = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()[:12]
    return f"RUN FORMAL {digest}"


def test_workflow_import_and_construction_do_not_run(tmp_path):
    batch = FakeBatch(tmp_path)
    v2.Workflow(batch, lambda: None)
    assert batch.events == []


def test_one_review_starts_formal_for_exact_scope(tmp_path):
    batch = FakeBatch(tmp_path)
    prompts, reports = [], []
    workflow = v2.Workflow(batch, lambda: None)
    result = workflow.run_all(
        display_report=reports.append, input_fn=lambda p: prompts.append(p) or token()
    )
    assert result["formal_started"]
    assert len(prompts) == len(reports) == 1
    dispatches = [value for event, value in batch.events if event == "run"]
    assert [d["phase"] for d in dispatches] == ["pilot", "formal"]
    assert dispatches[1]["reviewed_pilots"] == [
        ("multi_scale", "one"),
        ("multi_scale", "two"),
    ]
    reviews = list((tmp_path / "reviews").glob("*.json"))
    assert len(reviews) == 1
    assert json.loads(reviews[0].read_text())["formal_seeds"] == list(range(38, 46))


@pytest.mark.parametrize(
    "answer", ["", "yes", "RUN FORMAL", token(settings=("fixed_1b",))]
)
def test_wrong_or_other_scope_confirmation_does_not_start_formal(tmp_path, answer):
    batch = FakeBatch(tmp_path)
    result = v2.Workflow(batch, lambda: None).run_all(
        display_report=lambda _: None, input_fn=lambda _: answer
    )
    assert not result["formal_started"]
    assert [value["phase"] for event, value in batch.events if event == "run"] == [
        "pilot"
    ]
    assert not (tmp_path / "reviews").exists()


def test_missing_pilot_blocks_before_prompt(tmp_path):
    batch = FakeBatch(tmp_path)
    batch.report_status = "missing"
    with pytest.raises(RuntimeError, match="coverage/verification"):
        v2.Workflow(batch, lambda: None).run_all(
            input_fn=lambda _: pytest.fail("must not prompt")
        )
    assert [value["phase"] for event, value in batch.events if event == "run"] == [
        "pilot"
    ]


def test_blocked_queue_starts_no_pilot(tmp_path):
    batch = FakeBatch(tmp_path)
    batch.blocked = True
    with pytest.raises(RuntimeError, match="queue blocked"):
        v2.Workflow(batch, lambda: None).run_all()
    assert not any(event == "run" for event, _ in batch.events)


def test_environment_rechecked_before_formal_and_coverage(tmp_path):
    batch = FakeBatch(tmp_path)
    calls = []

    def check():
        calls.append(True)
        if len(calls) > 1:
            raise RuntimeError("changed")

    workflow = v2.Workflow(batch, check)
    with pytest.raises(RuntimeError, match="changed"):
        workflow.run_all(display_report=lambda _: None, input_fn=lambda _: token())
    assert [value["phase"] for event, value in batch.events if event == "run"] == [
        "pilot"
    ]
    with pytest.raises(RuntimeError, match="changed"):
        workflow.coverage()


def test_coverage_is_read_only_and_counts_all_eight(tmp_path):
    batch = FakeBatch(tmp_path)
    result = v2.Workflow(batch, lambda: None).coverage()
    assert all(row["pending"] == 8 and row["verified_successes"] == 0 for row in result)
    assert not any(event == "run" for event, _ in batch.events)


def test_drive_mount_required_before_training(release_files, monkeypatch, tmp_path):
    assets, release = release_files
    monkeypatch.setattr(v2, "_check_loaded_sources", lambda _: None)
    monkeypatch.setattr(v2.os.path, "ismount", lambda _: False)
    drive = tmp_path / "drive"
    (drive / "MyDrive").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="not mounted"):
        v2.prepare_workflow(
            ROOT, assets, tmp_path, release, content_root=tmp_path, drive_root=drive
        )


def test_prepare_isolates_new_plan_snapshots_contract_and_blocks_environment_change(
    release_files, monkeypatch, tmp_path
):
    assets, release = release_files
    monkeypatch.setattr(v2, "_check_loaded_sources", lambda _: None)
    monkeypatch.setattr(v2, "_check_versions", lambda _: None)
    monkeypatch.setattr(v2.os.path, "ismount", lambda _: True)
    environment = {"gpu": "A100", "packages": {"gpytorch": "1.15.2"}}
    monkeypatch.setattr(v2, "runtime_identity", lambda: environment.copy())
    snapshots = []

    class PreparedBatch:
        def __init__(self, assets, upstream, state, backups, **kwargs):
            self.state, self.backups = state, backups
            context = np.zeros((184, 2))
            context[:26, 0] = 1000
            self.bundle = SimpleNamespace(
                reference_utility=list(range(454)),
                utility=list(range(184)),
                context=context,
            )
            self.plan = {
                "experiment_id": v2.EXPERIMENT,
                "plan_id": PLAN,
                "package_source": {"git_dirty": False, "git_commit": CODE},
                "methods": [{"run_id": str(i)} for i in range(19)],
                "shared_settings": {
                    "pilot_seeds": [0],
                    "formal_seeds": list(range(38, 46)),
                },
            }

        def _check_environment(self, *args, **kwargs):
            return None

    monkeypatch.setitem(
        sys.modules, "colab_batch", SimpleNamespace(BatchRunner=PreparedBatch)
    )
    monkeypatch.setitem(
        sys.modules,
        "colab_support",
        SimpleNamespace(
            restore_latest=lambda *args: pytest.fail("fresh v2 must not restore v1"),
            snapshot_tree=lambda *args: snapshots.append(args),
        ),
    )
    drive = tmp_path / "drive"
    (drive / "MyDrive/llm_design_bench/llmdm_forward_v1/old").mkdir(parents=True)
    workflow = v2.prepare_workflow(
        ROOT, assets, tmp_path, release, content_root=tmp_path, drive_root=drive
    )
    assert workflow.batch.state == tmp_path / "llmdm_gpytorch_v2_state" / PLAN
    assert (
        workflow.batch.backups
        == drive / "MyDrive/llm_design_bench" / v2.EXPERIMENT / PLAN
    )
    assert len(snapshots) == 1
    assert (workflow.batch.state / "runtime-v2.json").is_file()
    workflow.check_runtime()
    assert len(snapshots) == 1
    environment["gpu"] = "T4"
    with pytest.raises(RuntimeError, match="runtime changed"):
        workflow.batch._check_environment("one", "cuda")


def test_prepare_rejects_modified_in_memory_release(
    release_files, monkeypatch, tmp_path
):
    assets, release = release_files
    monkeypatch.setattr(v2, "_check_loaded_sources", lambda _: None)
    release["plan_id"] = "f" * 64
    with pytest.raises(ValueError, match="on-disk metadata"):
        v2.prepare_workflow(ROOT, assets, tmp_path, release)


def test_notebook_cells_compile_and_training_has_single_entrypoint():
    notebook = json.loads(
        (ROOT / "notebooks/LLMDM_GPyTorch_Colab.ipynb").read_text("utf-8")
    )
    code = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            source = "".join(cell["source"])
            compile(source, cell["id"], "exec")
            code.append(source)
    source = "\n".join(code)
    calls = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
    run_all = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "run_all"
    ]
    assert len(run_all) == 1
    assert 'drive.mount("/content/drive")' in source
    assert "INCLUDE_FIXED_1B = False" in source
    assert "llmdm_forward_v1" not in source
    assert "force_remount" not in source
    assert "allow_environment_change" not in source
