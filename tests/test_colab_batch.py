"""Batch orchestration tests use static artifacts and fake jobs, never training."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_design_bench.evaluation.run_artifacts import (
    _component,
    file_sha256,
    stable_fingerprint,
    verify_successful_attempt,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads(
    (ROOT / "experiments/llmdm_forward_v1/plan.json").read_text(encoding="utf-8")
)


def _load_script(name):
    specification = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


support = _load_script("colab_support")
batch = _load_script("colab_batch")
REAL_EXPECTED_LOGICAL = batch.BatchRunner._expected_logical
REAL_VALIDATE_RELEASE = batch.BatchRunner._validate_release


@pytest.fixture
def plan():
    return copy.deepcopy(PLAN)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _logical(
    *,
    plan=PLAN,
    run_id="offline_mlp",
    setting="multi_scale",
    phase="pilot",
    seed=0,
    device="cuda",
):
    entry = next(item for item in plan["methods"] if item["run_id"] == run_id)
    task_id = "data_recipes_stack_exchange" + ("_1b" if setting == "fixed_1b" else "")
    logical = {
        "experiment_id": f"{plan['experiment_id']}_{phase}",
        "task_id": task_id,
        "run_id": run_id,
        "method_id": entry["method_id"],
        "method_seed": seed,
        "phase": phase,
        "candidate_budget": 128,
        "train_size": 26 if setting == "fixed_1b" else 184,
        "device": device,
        "dtype": f"torch.{entry['dtype']}",
        "design_space": {"dimension": 5},
        "method_config_json": json.dumps(entry["kwargs"], sort_keys=True),
        "problem_metadata_json": json.dumps({"utility_transform": "negative_loss"}),
        "provenance_json": json.dumps(
            {
                "plan_id": plan["plan_id"],
                "protocol_id": plan["shared_settings"]["protocol_id"],
                "data_manifest_id": plan["data_manifest_id"],
                "package_source_sha256": plan["package_source"]["sha256"],
                "oracle_device": "cpu",
            },
            sort_keys=True,
        ),
    }
    return logical


def _write_success(state, *, logical_overrides=None, **job):
    """Create genuinely verifiable artifacts from constants, not a real run."""
    logical = _logical(**job)
    logical.update(logical_overrides or {})
    phase, seed = logical["phase"], logical["method_seed"]
    task_id, run_id = logical["task_id"], logical["run_id"]
    dtype = logical["dtype"].removeprefix("torch.")
    device = logical["device"]
    relative = (
        Path("runs")
        / _component(logical["experiment_id"])
        / _component(task_id)
        / _component(run_id)
        / f"seed-{seed}"
        / "attempt-0001"
    )
    directory = state / phase / relative
    directory.mkdir(parents=True)
    candidates = np.full((128, 5), 0.2, dtype=dtype)
    target = np.array([1000.0, 19500.0], dtype=dtype)
    utility = np.full(128, -2.0, dtype=np.float64)
    score = np.full(128, 0.5, dtype=np.float64)
    np.savez(directory / "candidates.npz", candidates=candidates, target_context=target)
    np.savez(
        directory / "evaluation.npz",
        utility=utility,
        raw_loss=-utility,
        refnorm_score=score,
    )
    fingerprint = stable_fingerprint(logical)
    manifest = {"logical_config": logical, "logical_fingerprint": fingerprint}
    row = {
        **logical,
        "status": "success",
        "logical_fingerprint": fingerprint,
        "artifact_relative_dir": relative.as_posix(),
        "target_context_json": json.dumps(target.tolist()),
        "reference_min_utility": -3.0,
        "reference_max_utility": -1.0,
        "artifact_sha256": {
            name: file_sha256(directory / name)
            for name in ("candidates.npz", "evaluation.npz")
        },
        "training_summary_json": json.dumps({"final_standardized_mse": 0.01}),
        "diagnostics_json": "{}",
        "method_seconds": 2.0,
        "evaluation_seconds": 1.0,
        "total_seconds": 3.0,
        "peak_gpu_memory_bytes": 1024 if device == "cuda" else None,
        "raw_min_loss": 2.0,
        "raw_median_loss": 2.0,
        "raw_mean_loss": 2.0,
        "unique_candidate_count": 1,
        "unique_candidate_fraction": 1 / 128,
    }
    for name, value in (("raw", -2.0), ("refnorm", 0.5)):
        for statistic in ("max", "median", "mean"):
            suffix = "utility" if name == "raw" else "score"
            row[f"{name}_{statistic}_{suffix}"] = value
    _write_json(directory / "manifest.json", manifest)
    _write_json(directory / "result.json", row)
    csv_path = state / phase / "method_seed_results.csv"
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    verify_successful_attempt(directory)
    return directory, row


def _write_journal_success(state, backup, identity):
    digest = hashlib.sha256(support._canonical(identity)).hexdigest()
    dispatch_id = "20260912-fake-" + digest[:20]
    support._publish_json(
        backup / "journal" / f"{dispatch_id}.intent.json",
        {"dispatch_id": dispatch_id, "identity": identity, "identity_sha256": digest},
    )
    completion = {
        "dispatch_id": dispatch_id,
        "identity_sha256": digest,
        "status": "success",
        "returncode": 0,
    }
    local = state / "_colab_jobs" / dispatch_id / "completion.json"
    _write_json(local, completion)
    support._publish_json(
        backup / "journal" / f"{dispatch_id}.completion.json",
        {**completion, "snapshot": "retained-previous-snapshot"},
    )
    return local


@pytest.fixture
def runner(tmp_path, monkeypatch, plan):
    def validate(self):
        self.plan = copy.deepcopy(plan)
        self.bundle = SimpleNamespace(manifest_id=plan["data_manifest_id"])
        self._tasks = {}
        self._journal_index = None

    def expected(self, job):
        return _logical(
            plan=self.plan,
            **{
                name: job[name]
                for name in ("run_id", "setting", "phase", "seed", "device")
            },
        )

    monkeypatch.setattr(batch.BatchRunner, "_validate_release", validate)
    monkeypatch.setattr(batch.BatchRunner, "_expected_logical", expected)
    monkeypatch.setattr(
        batch,
        "_environment",
        lambda device: {
            "device": device,
            "python": "fake-python",
            "cuda_runtime": "fake-cuda",
            "cudnn": 1,
            "gpu": "fake-GPU" if device == "cuda" else None,
            "packages": {"torch": "fake-torch"},
        },
    )
    instance = batch.BatchRunner(
        assets=tmp_path / "assets",
        data_recipes_root=tmp_path / "upstream",
        state=tmp_path / "state",
        backups=tmp_path / "drive",
    )
    return instance


def _job(runner, run_id="offline_mlp", setting="multi_scale", phase="pilot"):
    return next(iter(runner._jobs([run_id], [setting], phase)))


def _save_job(runner, job, *, journal=True, **options):
    result = _write_success(
        runner.state,
        plan=runner.plan,
        **{
            name: job[name] for name in ("run_id", "setting", "phase", "seed", "device")
        },
        **options,
    )
    contract = runner.state / f"environment-{job['run_id']}.json"
    if not contract.exists():
        _write_json(contract, batch._environment(job["device"]))
    if journal:
        _write_journal_success(runner.state, runner.backups, job)
    return result


def _journal_result(runner, identity):
    digest = hashlib.sha256(support._canonical(identity)).hexdigest()
    return support._verified_json(
        runner.backups / "journal" / f"20260912-fake-{digest[:20]}.completion.json"
    )


def test_default_queue_covers_all_frozen_methods_and_seeds(runner):
    pilot = list(runner._jobs(None, ["multi_scale"], "pilot"))
    formal = list(runner._jobs(None, ["multi_scale"], "formal"))
    assert len(pilot) == 19 and len(formal) == 19 * 8
    assert [job["run_id"] for job in pilot] == [
        item["run_id"] for item in PLAN["methods"]
    ]
    assert all(job["seed"] == 0 for job in pilot)
    for entry in PLAN["methods"]:
        assert [
            job["seed"] for job in formal if job["run_id"] == entry["run_id"]
        ] == list(range(38, 46))
    assert len(list(runner._jobs(None, ["multi_scale", "fixed_1b"], "formal"))) == 304


def test_subset_order_and_method_specific_device_mapping(runner):
    names = ["spade", "bdi", "offline_mlp", "bo_qei", "ga_on_gp", "sobol"]
    jobs = list(runner._jobs(names, ["fixed_1b", "multi_scale"], "pilot"))
    assert [job["run_id"] for job in jobs[:6]] == names
    assert [job["device"] for job in jobs[:6]] == [
        "cuda",
        "cpu",
        "cuda",
        "cpu",
        "cpu",
        "cpu",
    ]
    assert all(job["setting"] == "fixed_1b" for job in jobs[:6])
    assert all(job["setting"] == "multi_scale" for job in jobs[6:])


@pytest.mark.parametrize("methods", [[], ["unknown"], ["coms", "coms"], "coms"])
def test_invalid_method_selections_are_rejected(runner, methods):
    with pytest.raises((ValueError, TypeError)):
        list(runner._jobs(methods, ["multi_scale"], "pilot"))


@pytest.mark.parametrize(
    "settings", [[], ["both"], ["multi_scale", "multi_scale"], "multi_scale"]
)
def test_invalid_setting_selections_are_rejected(runner, settings):
    with pytest.raises((ValueError, TypeError)):
        list(runner._jobs(["coms"], settings, "pilot"))


@pytest.mark.parametrize("phase", ["train", "both", "", None])
def test_invalid_phase_is_rejected(runner, phase):
    with pytest.raises((ValueError, TypeError)):
        list(runner._jobs(["coms"], ["multi_scale"], phase))


def test_preview_has_no_dispatch_or_state_writes(runner, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preview must not dispatch a job")

    monkeypatch.setattr(batch, "run_job", forbidden)
    before = set(runner.state.parent.rglob("*"))
    rows = runner.preview()
    assert len(rows) == 19
    assert {row["status"] for row in rows} == {"pending"}
    assert set(runner.state.parent.rglob("*")) == before


def test_formal_review_is_specific_to_method_and_setting(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "run_job", lambda *args, **kwargs: calls.append(args))
    _save_job(runner, _job(runner))
    for reviews, methods, settings in (
        ([], ["offline_mlp"], ["multi_scale"]),
        ([("multi_scale", "offline_mlp")], ["coms"], ["multi_scale"]),
        ([("multi_scale", "offline_mlp")], ["offline_mlp"], ["fixed_1b"]),
    ):
        with pytest.raises((ValueError, RuntimeError)):
            runner.run(
                methods=methods,
                settings=settings,
                phase="formal",
                reviewed_pilots=reviews,
            )
    assert not calls


def test_review_does_not_replace_missing_successful_pilot(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "run_job", lambda *args, **kwargs: calls.append(args))
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(
            methods=["offline_mlp"],
            phase="formal",
            reviewed_pilots=[("multi_scale", "offline_mlp")],
        )
    assert not calls


def test_verified_completed_pilot_is_skipped_without_dispatch(runner, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("verified success must not launch another child")

    monkeypatch.setattr(batch, "run_job", forbidden)
    job = _job(runner)
    directory, _ = _save_job(runner, job, journal=False)
    _write_journal_success(runner.state, runner.backups, job)
    original = {path: path.read_bytes() for path in directory.iterdir()}
    preview = runner.preview(methods=["offline_mlp"])
    assert preview[0]["status"] == "complete"
    result = runner.run(methods=["offline_mlp"])
    assert result[0]["status"] == "skipped"
    assert all(path.read_bytes() == content for path, content in original.items())


def test_skip_still_checks_drive_journal_provenance(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "run_job", lambda *args, **kwargs: calls.append(args))
    job = _job(runner)
    _save_job(runner, job, journal=False)
    local_completion = _write_journal_success(runner.state, runner.backups, job)
    local_completion.unlink()
    with pytest.raises(RuntimeError, match="successful prior job"):
        runner.run(methods=["offline_mlp"])
    assert not calls


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
@pytest.mark.parametrize("name", ["candidates.npz", "evaluation.npz"])
def test_missing_or_corrupt_success_artifacts_never_rerun(
    runner, monkeypatch, kind, name
):
    calls = []
    monkeypatch.setattr(batch, "run_job", lambda *args, **kwargs: calls.append(args))
    directory, _ = _save_job(runner, _job(runner))
    path = directory / name
    if kind == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(methods=["offline_mlp"])
    assert not calls


@pytest.mark.parametrize(
    "field",
    [
        "raw_min_loss",
        "raw_median_loss",
        "raw_mean_loss",
        "unique_candidate_count",
        "unique_candidate_fraction",
    ],
)
def test_matching_csv_and_result_with_wrong_statistics_still_block(
    runner, monkeypatch, field
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid statistics must not dispatch or retrain")

    monkeypatch.setattr(batch, "run_job", forbidden)
    directory, row = _save_job(runner, _job(runner))
    row[field] += 1
    _write_json(directory / "result.json", row)
    csv_path = runner.state / "pilot" / "method_seed_results.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, records = reader.fieldnames, list(reader)
    records[0][field] = str(row[field])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    before = {
        path: path.read_bytes() if path.is_file() else None
        for path in runner.state.parent.rglob("*")
    }
    preview = runner.preview(methods=["offline_mlp"])[0]
    assert preview["status"] == "blocked" and field in preview["error"]
    report = runner.pilot_report(methods=["offline_mlp"])[0]
    assert report["status"] == "blocked" and field in report["error"]
    with pytest.raises(RuntimeError, match=field):
        runner.run(methods=["offline_mlp"])
    after = {
        path: path.read_bytes() if path.is_file() else None
        for path in runner.state.parent.rglob("*")
    }
    assert after == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"device": "cpu"},
        {"method_config_json": "{}"},
        {"provenance_json": '{"plan_id": "stale-plan"}'},
        {"train_size": 26},
    ],
)
def test_stale_or_mismatched_result_never_counts_as_complete(
    runner, monkeypatch, overrides
):
    calls = []
    monkeypatch.setattr(batch, "run_job", lambda *args, **kwargs: calls.append(args))
    _save_job(runner, _job(runner), logical_overrides=overrides)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(methods=["offline_mlp"])
    assert not calls


@pytest.mark.parametrize("phase", ["pilot", "formal"])
def test_launch_uses_exact_identity_frozen_cli_and_absolute_paths(
    runner, monkeypatch, phase
):
    calls = []
    if phase == "formal":
        _save_job(runner, _job(runner))

    def fake_run(command, state, backups, identity, **options):
        calls.append((command, state, backups, identity, options))
        _save_job(runner, identity)
        return _journal_result(runner, identity)

    monkeypatch.setattr(batch, "run_job", fake_run)
    result = runner.run(
        methods=["offline_mlp"],
        phase=phase,
        reviewed_pilots=[("multi_scale", "offline_mlp")] if phase == "formal" else [],
    )
    assert len(calls) == (8 if phase == "formal" else 1)
    assert {row["status"] for row in result} == {"completed"}
    for index, (command, state, backups, identity, options) in enumerate(calls):
        seed = 38 + index if phase == "formal" else 0
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
        assert "--infrastructure-retry-reason" not in command
        for flag in ("--data-recipes-root", "--data-bundle", "--plan", "--results-dir"):
            assert Path(command[command.index(flag) + 1]).is_absolute()
        assert ("--pilot-results" in command) is (phase == "formal")
        if phase == "formal":
            assert (
                Path(command[command.index("--pilot-results") + 1])
                == runner.state / "pilot"
            )
        assert identity == {
            "plan_id": runner.plan["plan_id"],
            "run_id": "offline_mlp",
            "setting": "multi_scale",
            "phase": phase,
            "seed": seed,
            "device": "cuda",
            "oracle_device": "cpu",
        }
        assert state == runner.state and backups == runner.backups
        assert options["snapshot_interval"] == 60
        assert options.get("infrastructure_retry_reason") is None


def test_completed_formal_seeds_are_preserved_and_only_missing_dispatch(
    runner, monkeypatch
):
    _save_job(runner, _job(runner))
    jobs = list(runner._jobs(["offline_mlp"], ["multi_scale"], "formal"))
    retained = []
    for job in jobs[:2]:
        directory, _ = _save_job(runner, job)
        retained.append(
            (directory / "result.json", (directory / "result.json").read_bytes())
        )
    calls = []

    def fake_run(command, state, backups, identity, **options):
        calls.append(identity["seed"])
        _save_job(runner, identity)
        return _journal_result(runner, identity)

    monkeypatch.setattr(batch, "run_job", fake_run)
    results = runner.run(
        methods=["offline_mlp"],
        phase="formal",
        reviewed_pilots=[("multi_scale", "offline_mlp")],
    )
    assert calls == list(range(40, 46))
    assert [row["status"] for row in results] == ["skipped"] * 2 + ["completed"] * 6
    assert all(path.read_bytes() == data for path, data in retained)


def test_child_failure_stops_queue_without_retry_or_later_dispatch(runner, monkeypatch):
    calls = []

    def fail_run(command, state, backups, identity, **options):
        calls.append(identity)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(batch, "run_job", fail_run)
    with pytest.raises(RuntimeError) as caught:
        runner.run(methods=["coms", "offline_mlp", "spade"])
    assert isinstance(caught.value.__cause__, subprocess.CalledProcessError)
    assert len(calls) == 1
    assert calls[0]["run_id"] == "coms"


def test_exit_zero_without_verified_artifact_stops_queue(runner, monkeypatch):
    calls = []

    def fake_success(command, state, backups, identity, **options):
        calls.append(identity)
        return {"status": "success", "returncode": 0}

    monkeypatch.setattr(batch, "run_job", fake_success)
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(methods=["coms", "offline_mlp"])
    assert len(calls) == 1


def test_old_notebook_environment_contract_is_compatible(runner):
    environment = batch._environment("cuda")
    path = runner.state / "environment-offline_mlp.json"
    _write_json(path, environment)
    before = path.read_bytes()
    runner._check_environment("offline_mlp", "cuda", write=True)
    assert path.read_bytes() == before


def test_environment_changes_need_explicit_review_but_device_never_drifts(runner):
    environment = batch._environment("cuda")
    path = runner.state / "environment-offline_mlp.json"
    _write_json(path, {**environment, "gpu": "previous-GPU"})
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        runner._check_environment("offline_mlp", "cuda", write=True)
    assert path.read_bytes() == before
    runner.allow_environment_change = True
    runner._check_environment("offline_mlp", "cuda", write=True)
    with pytest.raises(RuntimeError):
        runner._check_environment("offline_mlp", "cpu", write=True)
    assert path.read_bytes() == before


def test_unresolved_old_journal_is_not_automatically_retried(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "run_job", lambda *args, **kwargs: calls.append(args))
    job = _job(runner)
    support._publish_json(
        runner.backups / "journal" / "unfinished.intent.json",
        {
            "dispatch_id": "unfinished",
            "identity": job,
            "identity_sha256": hashlib.sha256(support._canonical(job)).hexdigest(),
        },
    )
    with pytest.raises(RuntimeError, match="prior job"):
        runner.run(methods=["offline_mlp"])
    assert not calls


def test_success_without_backup_journal_is_blocked_not_retrained(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "run_job", lambda *args, **kwargs: calls.append(args))
    _save_job(runner, _job(runner), journal=False)
    assert runner.preview(methods=["offline_mlp"])[0]["status"] == "blocked"
    with pytest.raises(RuntimeError, match="completion journal"):
        runner.run(methods=["offline_mlp"])
    assert not calls


def test_success_without_original_environment_is_blocked(runner):
    _save_job(runner, _job(runner))
    (runner.state / "environment-offline_mlp.json").unlink()
    with pytest.raises(RuntimeError, match="environment contract"):
        runner.run(methods=["offline_mlp"])


def test_pilot_report_is_read_only_and_excludes_oracle_scores(runner, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("report must not dispatch a job")

    monkeypatch.setattr(batch, "run_job", forbidden)
    _save_job(runner, _job(runner))
    before = {
        path: path.read_bytes() for path in runner.state.rglob("*") if path.is_file()
    }
    report = runner.pilot_report(methods=["offline_mlp", "coms"])
    assert [row["status"] for row in report] == ["verified", "missing"]
    assert report[0]["candidate_budget"] == 128
    assert report[0]["training_summary"] == {"final_standardized_mse": 0.01}
    assert not any(name.startswith(("raw_", "refnorm_")) for name in report[0])
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize(
    "change",
    [
        {"training_summary_json": '{"mse": NaN}'},
        {"diagnostics_json": '{"nested": [1, Infinity]}'},
        {"method_seconds": float("nan")},
        {"evaluation_seconds": None},
        {"total_seconds": float("inf")},
    ],
)
def test_nonfinite_diagnostics_or_missing_runtime_blocks_acceptance(runner, change):
    directory, row = _save_job(runner, _job(runner))
    _write_json(directory / "result.json", {**row, **change})
    assert runner.pilot_report(methods=["offline_mlp"])[0]["status"] == "blocked"
    with pytest.raises(RuntimeError):
        runner.run(methods=["offline_mlp"])


@pytest.mark.parametrize("relative", ["../outside", "runs/wrong-attempt"])
def test_stale_or_escaping_csv_artifact_path_is_rejected(runner, relative):
    _save_job(runner, _job(runner))
    path = runner.state / "pilot/method_seed_results.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["artifact_relative_dir"] = relative
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="latest attempt"):
        runner.run(methods=["offline_mlp"])


def test_newer_incomplete_attempt_cannot_fall_back_to_older_success(runner):
    directory, _ = _save_job(runner, _job(runner))
    directory.with_name("attempt-0002").mkdir()
    with pytest.raises(RuntimeError, match="latest attempt"):
        runner.run(methods=["offline_mlp"])


def test_release_hash_check_precedes_data_or_plan_loading(runner, monkeypatch):
    artifact = runner.assets / "fixture.json"
    _write_json(artifact, {"original": True})
    release = {
        "artifact_sha256": {"fixture.json": file_sha256(artifact)},
        "data_manifest_id": runner.plan["data_manifest_id"],
        "plan_id": runner.plan["plan_id"],
        "package_source_sha256": runner.plan["package_source"]["sha256"],
    }
    _write_json(runner.assets / "release.json", release)
    _write_json(artifact, {"modified": True})

    def forbidden(*args, **kwargs):
        pytest.fail("bad release hash must fail before loading dataset or plan")

    monkeypatch.setattr(batch, "load_data_manifest", forbidden)
    monkeypatch.setattr(batch, "load_method_plan", forbidden)
    with pytest.raises(ValueError, match="release artifact mismatch"):
        REAL_VALIDATE_RELEASE(runner)


@pytest.mark.parametrize(
    "key", ["data_manifest_id", "plan_id", "package_source_sha256"]
)
def test_release_identity_mismatch_is_rejected(runner, monkeypatch, key):
    release = {
        "artifact_sha256": {},
        "data_manifest_id": runner.plan["data_manifest_id"],
        "plan_id": runner.plan["plan_id"],
        "package_source_sha256": runner.plan["package_source"]["sha256"],
    }
    release[key] = "different-release"
    _write_json(runner.assets / "release.json", release)
    monkeypatch.setattr(
        batch, "load_data_manifest", lambda *args, **kwargs: runner.bundle
    )
    monkeypatch.setattr(batch, "load_method_plan", lambda *args, **kwargs: runner.plan)
    with pytest.raises(ValueError, match="identities disagree"):
        REAL_VALIDATE_RELEASE(runner)


def test_each_public_operation_refreshes_new_unresolved_journal(runner):
    job = _job(runner)
    _save_job(runner, job)
    assert runner.preview(methods=["offline_mlp"])[0]["status"] == "complete"
    support._publish_json(
        runner.backups / "journal" / "9999-new-unresolved.intent.json",
        {
            "identity": job,
            "identity_sha256": hashlib.sha256(support._canonical(job)).hexdigest(),
            "dispatch_id": "9999-new-unresolved",
        },
    )
    assert runner.preview(methods=["offline_mlp"])[0]["status"] == "blocked"
    assert runner.pilot_report(methods=["offline_mlp"])[0]["status"] == "blocked"
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run(methods=["offline_mlp"])


def test_completion_corruption_is_not_hidden_by_prior_operation_cache(runner):
    job = _job(runner)
    _save_job(runner, job)
    assert runner.preview(methods=["offline_mlp"])[0]["status"] == "complete"
    completion = next((runner.backups / "journal").glob("*.completion.json"))
    completion.write_bytes(completion.read_bytes() + b"tampered")
    assert runner.pilot_report(methods=["offline_mlp"])[0]["status"] == "blocked"
    with pytest.raises(RuntimeError, match="corrupt"):
        runner.run(methods=["offline_mlp"])


def test_cached_remote_completion_still_checks_local_completion_before_skip(
    runner, monkeypatch
):
    job = _job(runner)
    _save_job(runner, job)
    original = runner._verified_row
    calls = 0
    children = []

    def check_and_simulate_restore_loss(identity):
        nonlocal calls
        calls += 1
        if calls == 2:
            next((runner.state / "_colab_jobs").glob("*/completion.json")).unlink()
        return original(identity)

    monkeypatch.setattr(runner, "_verified_row", check_and_simulate_restore_loss)
    monkeypatch.setattr(batch, "run_job", lambda *args, **kwargs: children.append(args))
    with pytest.raises(RuntimeError, match="local completion"):
        runner.run(methods=["offline_mlp"])
    assert calls == 2 and not children


def test_corrupt_intent_does_not_publish_a_partial_trusted_cache(runner):
    _save_job(runner, _job(runner))
    corrupt = runner.backups / "journal" / "9999-corrupt.intent.json"
    _write_json(corrupt, {"bad": "missing checksum"})
    preview = runner.preview(methods=["offline_mlp", "coms", "spade"])
    assert [row["status"] for row in preview] == ["blocked"] * 3


def test_large_preview_verifies_each_journal_record_at_most_once(runner, monkeypatch):
    _save_job(runner, _job(runner))
    for job in runner._jobs(["offline_mlp"], ["multi_scale"], "formal"):
        _save_job(runner, job)
    original = support._verified_json
    reads = {}

    def counted(path):
        reads[path] = reads.get(path, 0) + 1
        return original(path)

    monkeypatch.setattr(support, "_verified_json", counted)
    rows = runner.preview(phase="formal")
    assert len(rows) == 152
    assert len([row for row in rows if row["status"] == "complete"]) == 8
    assert all(count == 1 for count in reads.values())


def test_new_child_journal_is_verified_before_advancing_to_next_seed(
    runner, monkeypatch
):
    _save_job(runner, _job(runner))
    children = []

    def corrupt_child_completion(command, state, backups, identity, **options):
        children.append(identity)
        _save_job(runner, identity)
        result = _journal_result(runner, identity)
        digest = hashlib.sha256(support._canonical(identity)).hexdigest()
        completion = (
            backups / "journal" / f"20260912-fake-{digest[:20]}.completion.json"
        )
        completion.write_bytes(completion.read_bytes() + b"tampered")
        return result

    monkeypatch.setattr(batch, "run_job", corrupt_child_completion)
    with pytest.raises(RuntimeError, match="corrupt"):
        runner.run(
            methods=["offline_mlp"],
            phase="formal",
            reviewed_pilots=[("multi_scale", "offline_mlp")],
        )
    assert len(children) == 1 and children[0]["seed"] == 38


@pytest.mark.parametrize("entry", PLAN["methods"], ids=lambda item: item["run_id"])
@pytest.mark.parametrize("setting", ["multi_scale", "fixed_1b"])
@pytest.mark.parametrize("phase", ["pilot", "formal"])
def test_real_fingerprint_matches_frozen_runner_before_any_training(
    runner, monkeypatch, entry, setting, phase
):
    """Replay the old CLI fingerprint boundary; abort before method.run/oracle."""
    from llm_design_bench.evaluation import seed_runner
    from llm_design_bench.evaluation.unified_report import (
        BenchmarkTaskSpec,
        BenchmarkTrial,
    )
    from llm_design_bench.problem import OfflineProblem, ProblemMetadata
    from llm_design_bench.spaces import SimplexSpace

    def forbidden(*args, **kwargs):
        pytest.fail("fingerprint compatibility test must not train or evaluate")

    problem = OfflineProblem(
        train_designs=torch.tensor(
            [[0.2] * 5, [0.1, 0.2, 0.3, 0.3, 0.1]], dtype=torch.float64
        ),
        train_context=torch.tensor([[20, 1000], [1000, 19500]], dtype=torch.float64),
        train_utility=torch.tensor([-3.0, -2.0], dtype=torch.float64),
        target_context=torch.tensor([1000, 19500], dtype=torch.float64),
        design_space=SimplexSpace(5),
        metadata=ProblemMetadata(
            task_name=batch.TASK_IDS[setting],
            objective_name="fixture_cross_entropy",
            extra={"utility_transform": "negative_loss", "fixture": True},
        ),
    )
    trial = BenchmarkTrial(
        evaluator_task=SimpleNamespace(at_target_fidelity=forbidden, predict=forbidden),
        problem=problem,
        reference_utility=np.array([-4.0, -1.0]),
        dataset_seed=12,
        split_seed=34,
    )
    task = BenchmarkTaskSpec(
        task_id=batch.TASK_IDS[setting],
        display_name="static-fixture",
        suite="data_mixture",
        category="data_mixture",
        category_display_name="static-fixture",
        trial_factory=lambda seed: trial,
        normalization_reference_id="unique-fixture-reference",
    )
    monkeypatch.setattr(
        batch, "make_frozen_data_recipes_task_spec", lambda *args, **kwargs: task
    )
    job = _job(runner, entry["run_id"], setting, phase)
    expected = REAL_EXPECTED_LOGICAL(runner, job)
    captured = []

    class StopBeforeTraining(BaseException):
        pass

    def capture_begin(*args, **kwargs):
        captured.append(kwargs["logical_config"])
        raise StopBeforeTraining

    original_factory = seed_runner.make_method

    def guarded_method(method_id, **kwargs):
        method = original_factory(method_id, **kwargs)
        monkeypatch.setattr(method, "run", forbidden)
        return method

    monkeypatch.setattr(seed_runner, "make_method", guarded_method)
    monkeypatch.setattr(seed_runner.RunAttempt, "begin", capture_begin)
    monkeypatch.setattr(seed_runner, "capture_environment", lambda device: {})
    config = seed_runner.SeedBenchmarkConfig(
        experiment_id=f"{runner.plan['experiment_id']}_{phase}",
        results_dir=runner.state / phase,
        save_artifacts=True,
        seeds=(job["seed"],),
        required_seeds=runner.plan["shared_settings"][f"{phase}_seeds"],
        phase=phase,
        candidate_budget=128,
        device=job["device"],
        dataset_seed=trial.dataset_seed,
        split_seed=trial.split_seed,
        task_id=task.task_id,
        normalization_reference_id=task.normalization_reference_id,
        package_commit=runner.plan["package_source"]["git_commit"],
        provenance={
            "protocol_id": runner.plan["shared_settings"]["protocol_id"],
            "plan_id": runner.plan["plan_id"],
            "data_manifest_id": runner.bundle.manifest_id,
            "package_source_sha256": runner.plan["package_source"]["sha256"],
            "oracle_device": "cpu",
        },
    )
    with pytest.raises(StopBeforeTraining):
        seed_runner.run_method_seed_benchmark(
            trial.evaluator_task,
            trial.problem,
            [
                seed_runner.MethodSpec(
                    entry["method_id"],
                    entry["kwargs"],
                    run_id=entry["run_id"],
                    dtype=getattr(torch, entry["dtype"]),
                )
            ],
            reference_utility=trial.reference_utility,
            config=config,
            write_results=False,
        )
    assert len(captured) == 1
    assert captured[0] == expected
    assert stable_fingerprint(captured[0]) == stable_fingerprint(expected)
