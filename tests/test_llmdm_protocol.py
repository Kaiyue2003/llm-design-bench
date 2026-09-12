import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from llm_design_bench.evaluation import llmdm_protocol as protocol
from llm_design_bench.llmdm_cli import app


@pytest.fixture
def bundle():
    return SimpleNamespace(manifest_id="test-manifest")


@pytest.fixture(autouse=True)
def fixed_source(monkeypatch):
    monkeypatch.setattr(
        protocol,
        "package_source_identity",
        lambda: {
            "sha256": "fixed-source",
            "git_commit": "abc123",
            "git_dirty": False,
        },
    )


def _plan(bundle):
    return protocol.freeze_method_plan(
        bundle,
        [
            {"method_id": "best_logged", "kwargs": {}},
            {"method_id": "bdi", "kwargs": {"steps": 2}},
        ],
        experiment_id="contract",
    )


def test_freeze_expands_defaults_and_locks_common_settings(bundle, tmp_path):
    plan = _plan(bundle)
    assert plan["shared_settings"]["candidate_budget"] == 128
    assert plan["shared_settings"]["formal_seeds"] == list(range(38, 46))
    assert plan["shared_settings"]["pilot_seeds"] == [0]
    best, bdi = plan["methods"]
    assert best["dtype"] == "float32"
    assert bdi["dtype"] == "float64"
    assert bdi["kwargs"]["steps"] == 2
    assert len(bdi["kwargs"]) > 1
    path = protocol.save_method_plan(plan, tmp_path / "plan.json")
    assert protocol.load_method_plan(path, bundle) == plan
    with pytest.raises(FileExistsError):
        protocol.save_method_plan(plan, path)


def test_all_registered_methods_configs_round_trip_without_training(bundle):
    from llm_design_bench.optimizers.registry import method_names

    # Test-defined classes can require special constructor values; only built-ins.
    names = [
        "best_logged",
        "random_search",
        "sobol",
        "offline_mlp",
        "standard_ga",
        "coms",
        "bdi",
        "bo_qei",
        "ga_on_gp",
        "cma_es",
        "reinforce",
        "mc_dropout",
        "roma",
        "ict",
        "tri_mentoring",
        "ltr",
        "match_opt",
        "pgs",
        "spade",
    ]
    assert set(names).issubset(method_names())
    plan = protocol.freeze_method_plan(
        bundle,
        [{"method_id": name, "kwargs": {}} for name in names],
        experiment_id="defaults-roundtrip",
    )
    protocol.validate_method_plan(plan, bundle)
    assert len(plan["methods"]) == 19


def test_plan_tampering_data_mismatch_and_source_drift_rejected(bundle, monkeypatch):
    plan = _plan(bundle)
    changed = copy.deepcopy(plan)
    changed["shared_settings"]["candidate_budget"] = 16
    with pytest.raises(ValueError, match="hash mismatch"):
        protocol.validate_method_plan(changed, bundle)
    with pytest.raises(ValueError, match="different data manifest"):
        protocol.validate_method_plan(plan, SimpleNamespace(manifest_id="other"))
    monkeypatch.setattr(
        protocol, "package_source_identity", lambda: {"sha256": "changed"}
    )
    with pytest.raises(ValueError, match="package source changed"):
        protocol.validate_method_plan(plan, bundle)


@pytest.mark.parametrize(
    "methods",
    [
        [],
        [{"method_id": "best_logged"}],
        [{"method_id": "best_logged", "kwargs": {}, "dtype": "float64"}],
        [{"method_id": "best_logged", "kwargs": {}, "run_id": "../escape"}],
        [{"method_id": "best_logged", "kwargs": []}],
        [{"method_id": "best_logged", "kwargs": {}}] * 2,
    ],
)
def test_method_plan_requires_explicit_valid_configs(bundle, methods):
    with pytest.raises((ValueError, TypeError)):
        protocol.freeze_method_plan(bundle, methods, experiment_id="test")


def test_frozen_pilot_preserves_protocol_and_per_method_precision(
    bundle, monkeypatch, tmp_path
):
    plan = _plan(bundle)
    calls = []

    def task_factory(bundle_arg, **kwargs):
        assert bundle_arg is bundle
        calls.append(kwargs)
        return SimpleNamespace(
            task_id="main" if kwargs["logged_model_scale"] is None else "1b"
        )

    monkeypatch.setattr(protocol, "make_frozen_data_recipes_task_spec", task_factory)
    captured = {}

    def run_suite(tasks, specs, *, config, metadata):
        captured.update(tasks=tasks, specs=specs, config=config, metadata=metadata)
        return "result"

    monkeypatch.setattr(protocol, "run_benchmark_suite", run_suite)
    result = protocol.run_frozen_experiment(
        bundle,
        plan,
        data_recipes_root=tmp_path,
        results_dir=tmp_path / "pilot",
        phase="pilot",
        setting="both",
    )
    assert result == "result"
    assert [call["logged_model_scale"] for call in calls] == [None, 1000.0]
    config = captured["config"]
    assert config.seeds == (0,) and config.required_seeds == (0,)
    assert config.candidate_budget == 128 and config.save_artifacts
    assert config.provenance["plan_id"] == plan["plan_id"]
    assert [spec.dtype for spec in captured["specs"]] == [torch.float32, torch.float64]


def test_formal_requires_same_plan_pilot_and_never_uses_pilot_scores(
    bundle, monkeypatch, tmp_path
):
    plan = _plan(bundle)
    monkeypatch.setattr(
        protocol,
        "make_frozen_data_recipes_task_spec",
        lambda *a, **kw: SimpleNamespace(task_id="main"),
    )
    monkeypatch.setattr(protocol, "run_benchmark_suite", lambda *a, **kw: kw["config"])
    monkeypatch.setattr(
        protocol,
        "verify_successful_attempt",
        lambda path: (
            {
                "logical_config": {
                    "phase": "pilot",
                    "method_seed": 0,
                    "candidate_budget": 128,
                    "task_id": "main",
                    "run_id": "best_logged",
                    "method_id": "best_logged",
                    "dtype": "torch.float32",
                    "method_config_json": json.dumps(plan["methods"][0]["kwargs"]),
                    "provenance_json": json.dumps({"plan_id": plan["plan_id"]}),
                },
            },
            {"status": "success"},
        ),
    )
    kwargs = {
        "data_recipes_root": tmp_path,
        "results_dir": tmp_path / "formal",
        "phase": "formal",
        "run_ids": ["best_logged"],
        "seeds": [38],
    }
    with pytest.raises(ValueError, match="require pilot_results"):
        protocol.run_frozen_experiment(bundle, plan, **kwargs)
    # No oracle score is needed or examined for pilot acceptance.
    pilot = pd.DataFrame(
        [
            {
                "task_id": "main",
                "run_id": "best_logged",
                "method_seed": 0,
                "phase": "pilot",
                "status": "success",
                "provenance_json": json.dumps({"plan_id": plan["plan_id"]}),
                "artifact_relative_dir": "runs/test/attempt-0001",
            }
        ]
    )
    pilot.to_csv(tmp_path / "method_seed_results.csv", index=False)
    config = protocol.run_frozen_experiment(
        bundle, plan, pilot_results=tmp_path, **kwargs
    )
    assert config.seeds == (38,) and config.required_seeds == tuple(range(38, 46))
    pilot.loc[0, "provenance_json"] = json.dumps({"plan_id": "old"})
    pilot.to_csv(tmp_path / "method_seed_results.csv", index=False)
    with pytest.raises(ValueError, match="different frozen plan"):
        protocol.run_frozen_experiment(bundle, plan, pilot_results=tmp_path, **kwargs)


@pytest.mark.parametrize(
    "phase,seeds", [("formal", [0]), ("pilot", [38]), ("pilot", [])]
)
def test_wrong_phase_seeds_rejected_before_task_loading(bundle, tmp_path, phase, seeds):
    with pytest.raises(ValueError, match="seeds must"):
        protocol.run_frozen_experiment(
            bundle,
            _plan(bundle),
            data_recipes_root=tmp_path,
            results_dir=tmp_path,
            phase=phase,
            seeds=seeds,
        )


def test_cli_help_separates_preparation_from_execution():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert (
        "prepare" in result.output
        and "freeze" in result.output
        and "run" in result.output
    )
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "pilot" in result.output and "formal" in result.output
