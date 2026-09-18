"""Reporting and replay continue to work through the current method API."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from llm_design_bench.evaluation import (
    MethodSpec,
    SeedBenchmarkConfig,
    make_synthetic_task_spec,
    run_benchmark_suite,
)
from llm_design_bench.evaluation.run_artifacts import verify_successful_attempt


def test_current_suite_saves_reports_replays_and_rejects_tampering(tmp_path):
    # Synthetic tasks are inexpensive fixtures for the same persistence/report
    # runtime used by LLM-DM, not a second execution entry point or formal run.
    tasks = [
        make_synthetic_task_spec(name, logged_samples=16)
        for name in ("ackley", "booth")
    ]
    methods = [
        MethodSpec("best_logged"),
        MethodSpec(
            "coms",
            {
                "epochs": 1,
                "hidden_size": 8,
                "particle_steps": 1,
                "adversarial_steps": 1,
            },
        ),
        MethodSpec("bdi", {"steps": 1}),
        MethodSpec(
            "cbas",
            {
                "epochs": 1,
                "steps": 1,
                "hidden_size": 8,
                "population_size": 8,
                "adaptation_epochs": 1,
                "ensemble_size": 2,
            },
        ),
        MethodSpec(
            "ddom", {"epochs": 1, "steps": 2, "hidden_size": 8, "diffusion_steps": 4}
        ),
    ]
    config = SeedBenchmarkConfig(
        experiment_id="current-suite-smoke",
        seeds=(38, 39),
        candidate_budget=4,
        save_artifacts=True,
        results_dir=tmp_path,
        fail_fast=True,
    )
    first = run_benchmark_suite(tasks, methods, config=config)
    assert len(first.per_seed) == 2 * 5 * 2
    assert len(first.summary) == 2 * 5
    assert set(first.per_seed["status"]) == {"success"}
    assert set(first.per_seed["method_id"]) == {method.method_id for method in methods}
    assert np.isfinite(first.per_seed["raw_max_utility"]).all()
    assert np.isfinite(first.summary["refnorm_max_score_std"]).all()
    assert np.isfinite(first.summary["refnorm_max_score_se"]).all()
    expected_reports = {
        "method_seed_results.csv",
        "method_seed_summary.csv",
        "d_best_summary.csv",
        "rank_summary.csv",
        "run_metadata.json",
        "README.md",
        "benchmark_table.tex",
    }
    assert expected_reports <= {path.name for path in tmp_path.iterdir()}
    markdown = (tmp_path / "README.md").read_text(encoding="utf-8")
    for name in ("Best Logged", "COMs", "BDI", "CbAS", "DDOM"):
        assert name in markdown

    artifacts = [Path(path) for path in first.per_seed["artifact_dir"]]
    original_results = {}
    for artifact in artifacts:
        manifest, saved = verify_successful_attempt(artifact)
        settings = json.loads(manifest["logical_config"]["method_config_json"])
        training = json.loads(saved["training_summary_json"])
        assert training["resolved_method_config"] == settings
        original_results[artifact] = (artifact / "result.json").read_bytes()

    second = run_benchmark_suite(tasks, methods, config=replace(config, resume=True))
    np.testing.assert_array_equal(
        first.per_seed["raw_max_utility"], second.per_seed["raw_max_utility"]
    )
    for artifact, original in original_results.items():
        assert (artifact / "result.json").read_bytes() == original
    assert len(list(tmp_path.rglob("result.json"))) == len(first.per_seed)

    candidates = artifacts[0] / "candidates.npz"
    candidates.write_bytes(candidates.read_bytes() + b"corrupted-test-artifact")
    with pytest.raises(ValueError, match="corrupted"):
        run_benchmark_suite(tasks, methods, config=replace(config, resume=True))
