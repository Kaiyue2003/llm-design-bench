"""End-to-end protocol checks with invented logs and a deterministic fake oracle.

The names ``pilot`` and ``formal`` below test orchestration only; these are not
scientific results and no real data-recipes checkpoint or model is trained.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from llm_design_bench.evaluation import llmdm_protocol
from llm_design_bench.evaluation.run_artifacts import verify_successful_attempt
from llm_design_bench.llmdm_cli import app
from llm_design_bench.tasks.data_recipes import DOMAIN_ORDER

CHECKPOINT = "opt_algos/data_models/test_run/checkpoints/checkpoint_latest.pt"
PROBE_MODULE = "_llmdm_workflow_test_probe"


def _invoke(arguments: list[str]):
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    return result


@pytest.fixture
def workflow(tmp_path: Path, monkeypatch):
    root = tmp_path / "upstream"
    checkpoint = root / CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    (root / "results").mkdir()
    torch.save({"fixture": True}, checkpoint)
    for sidecar in ("config.json", "feature_mask.json"):
        (checkpoint.parent.parent / sidecar).write_text("{}\n", encoding="utf-8")
    probe = ModuleType(PROBE_MODULE)
    probe.events = []
    monkeypatch.setitem(sys.modules, PROBE_MODULE, probe)
    source = (
        "from pathlib import Path\nimport torch\n"
        f"import {PROBE_MODULE} as probe\n"
        "probe.events.append('import')\n"
        "class DataModelBenchmark:\n"
        "    def __init__(self, metric_index=4, device='cpu'):\n"
        f"        feature_names = {dict(enumerate(DOMAIN_ORDER))!r}\n"
        "        probe.events.append('construct')\n"
        "        value = torch.load(Path(__file__).parent / 'data_models/test_run/checkpoints/checkpoint_latest.pt')\n"
        "        assert value == {'fixture': True}\n"
        "    def _raw_func_with_model_scale(self, z, m, x, with_exp=True):\n"
        "        assert z == 195 and m == 100 and with_exp is False\n"
        "        probe.events.append('evaluate')\n"
        "        return 1.0 + sum(float(v) * (i + 1) / 10 for i, v in enumerate(x))\n"
    )
    (root / "opt_algos/benchmarks.py").write_text(source, encoding="utf-8")
    rows = []
    for index in range(8):
        for group, offset in (("20M", 100.0), ("1B", 1.0)):
            mixture = np.array([1 + index, 2, 3, 4, 5], dtype=np.float64)
            mixture /= mixture.sum()
            rows.append(
                {
                    "group": group,
                    "token_probabilities": mixture,
                    "history": pd.DataFrame(
                        {
                            "_step": [19000 + index % 3 * 100],
                            "eval/RedPajamaStackExchange/CrossEntropyLoss": [
                                offset + index
                            ],
                        }
                    ),
                }
            )
    pd.DataFrame(rows, index=[0] * len(rows)).to_pickle(
        root / "results/data_mixing_runs.pkl"
    )
    methods = tmp_path / "methods.json"
    methods.write_text(
        json.dumps(
            [
                {"method_id": "best_logged", "kwargs": {}},
                {"method_id": "bdi", "kwargs": {"steps": 1}},
            ]
        ),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    plan = tmp_path / "plan.json"
    _invoke(
        [
            "prepare",
            "--data-recipes-root",
            str(root),
            "--oracle-checkpoint",
            str(checkpoint),
            "--output",
            str(bundle),
        ]
    )
    assert not probe.events, (
        "prepare must neither import nor initialize the upstream oracle"
    )
    _invoke(
        [
            "freeze",
            "--data-recipes-root",
            str(root),
            "--data-bundle",
            str(bundle),
            "--methods-file",
            str(methods),
            "--experiment-id",
            "test-fixture-only",
            "--output",
            str(plan),
        ]
    )
    assert not probe.events, "freeze must not touch the upstream oracle"

    def command(phase: str, output: Path, *options: str) -> list[str]:
        return [
            "run",
            "--data-recipes-root",
            str(root),
            "--data-bundle",
            str(bundle),
            "--plan",
            str(plan),
            "--phase",
            phase,
            "--results-dir",
            str(output),
            *options,
        ]

    return {
        "root": root,
        "checkpoint": checkpoint,
        "bundle": bundle,
        "plan": plan,
        "probe": probe,
        "command": command,
        "base": tmp_path,
    }


def _result_frames(directory: Path):
    return (
        pd.read_csv(directory / "method_seed_results.csv"),
        pd.read_csv(directory / "method_seed_summary.csv"),
        pd.read_csv(directory / "rank_summary.csv"),
    )


def _assert_saved_candidates(directory: Path, rows: pd.DataFrame) -> None:
    for _, row in rows.iterrows():
        relative = Path(row["artifact_relative_dir"])
        assert not relative.is_absolute()
        attempt = directory / relative
        manifest, saved = verify_successful_attempt(attempt)
        expected = (
            np.dtype("float64") if row["method_id"] == "bdi" else np.dtype("float32")
        )
        with np.load(attempt / "candidates.npz", allow_pickle=False) as candidates:
            assert candidates["candidates"].shape == (128, 5)
            assert candidates["candidates"].dtype == expected
            assert candidates["target_context"].tolist() == [1000.0, 19500.0]
        with np.load(attempt / "evaluation.npz", allow_pickle=False) as evaluation:
            assert evaluation["utility"].shape == (128,)
            np.testing.assert_array_equal(
                evaluation["raw_loss"], -evaluation["utility"]
            )
        assert saved["status"] == "success"
        assert manifest["logical_config"]["candidate_budget"] == 128


def test_cli_full_fake_workflow_freeze_pilot_append_formal_and_portable_resume(
    workflow,
):
    command = workflow["command"]
    pilot = workflow["base"] / "pilot"
    formal = workflow["base"] / "formal"
    _invoke(command("pilot", pilot, "--setting", "both"))
    pilot_rows, pilot_summary, pilot_ranks = _result_frames(pilot)
    assert len(pilot_rows) == 4
    assert (pilot_rows["method_seed"] == 0).all()
    assert (pilot_rows["status"] == "success").all()
    assert not pilot_summary["rank_eligible"].any()
    assert pilot_summary["task_rank"].isna().all()
    assert (pilot_ranks["tasks_ranked"] == 0).all()
    _assert_saved_candidates(pilot, pilot_rows)
    assert workflow["probe"].events.count("evaluate") == 4 * 128

    with np.load(workflow["bundle"] / "visible.npz", allow_pickle=False) as visible:
        main_ids = visible["row_ids"].tolist()
        fixed_ids = visible["row_ids"][visible["context"][:, 0] == 1000].tolist()
    for _, row in pilot_rows.iterrows():
        metadata = json.loads(row["problem_metadata_json"])
        expected_ids = fixed_ids if row["task_id"].endswith("_1b") else main_ids
        assert metadata["visible_row_ids"] == expected_ids

    _invoke(
        command(
            "formal",
            formal,
            "--setting",
            "both",
            "--seed",
            "38",
            "--pilot-results",
            str(pilot),
        )
    )
    initial_rows, initial_summary, initial_ranks = _result_frames(formal)
    assert len(initial_rows) == 4
    assert (initial_summary["successful_runs"] == 1).all()
    assert (initial_summary["requested_runs"] == 8).all()
    assert (initial_summary["missing_runs"] == 7).all()
    assert not initial_summary["rank_eligible"].any()
    assert (initial_ranks["tasks_ranked"] == 0).all()
    first_attempts = set(initial_rows["artifact_relative_dir"])
    extra_seeds = [part for seed in range(39, 46) for part in ("--seed", str(seed))]
    _invoke(
        command(
            "formal",
            formal,
            "--setting",
            "both",
            "--pilot-results",
            str(pilot),
            *extra_seeds,
        )
    )
    final_rows, final_summary, final_ranks = _result_frames(formal)
    assert len(final_rows) == 32
    assert set(final_rows["method_seed"]) == set(range(38, 46))
    assert (final_summary["successful_runs"] == 8).all()
    assert (final_summary["missing_runs"] == 0).all()
    assert final_summary["complete_seed_set"].all()
    assert final_summary["rank_eligible"].all()
    assert final_summary["task_rank"].notna().all()
    assert (final_ranks["tasks_ranked"] == 2).all()
    assert first_attempts.issubset(set(final_rows["artifact_relative_dir"]))
    _assert_saved_candidates(formal, final_rows)

    # Copied results must resolve their artifacts relatively, ignoring stale
    # absolute paths that naturally refer to the original machine/directory.
    portable_pilot, portable_formal = (
        workflow["base"] / "moved-pilot",
        workflow["base"] / "moved-formal",
    )
    shutil.copytree(pilot, portable_pilot)
    shutil.copytree(formal, portable_formal)
    before = len(workflow["probe"].events)
    _invoke(
        command(
            "formal",
            portable_formal,
            "--setting",
            "both",
            "--seed",
            "38",
            "--pilot-results",
            str(portable_pilot),
            "--resume",
        )
    )
    assert len(workflow["probe"].events) == before, "resume must not query or retrain"
    resumed, summary, _ = _result_frames(portable_formal)
    assert len(resumed) == 32 and summary["rank_eligible"].all()


@pytest.mark.parametrize("damage", ["missing", "truncated"])
def test_corrupted_pilot_artifacts_block_formal_gate_and_resume(workflow, damage: str):
    command = workflow["command"]
    pilot = workflow["base"] / "pilot"
    _invoke(command("pilot", pilot, "--run-id", "best_logged"))
    rows, _, _ = _result_frames(pilot)
    candidate_path = pilot / rows.iloc[0]["artifact_relative_dir"] / "candidates.npz"
    if damage == "missing":
        candidate_path.unlink()
    else:
        candidate_path.write_bytes(b"truncated")
    before = len(workflow["probe"].events)
    denied = CliRunner().invoke(
        app,
        command(
            "formal",
            workflow["base"] / "formal",
            "--run-id",
            "best_logged",
            "--seed",
            "38",
            "--pilot-results",
            str(pilot),
        ),
    )
    assert denied.exit_code != 0
    assert "missing or corrupted" in str(denied.exception)
    resumed = CliRunner().invoke(
        app, command("pilot", pilot, "--run-id", "best_logged", "--resume")
    )
    assert resumed.exit_code != 0
    assert "missing or corrupted" in str(resumed.exception)
    assert len(workflow["probe"].events) == before


def test_frozen_cli_rejects_changed_package_code_before_oracle(workflow, monkeypatch):
    monkeypatch.setattr(
        llmdm_protocol, "package_source_identity", lambda: {"sha256": "changed-code"}
    )
    denied = CliRunner().invoke(
        app, workflow["command"]("pilot", workflow["base"] / "pilot")
    )
    assert denied.exit_code != 0
    assert "package source changed" in str(denied.exception)
    assert not workflow["probe"].events


def test_frozen_cli_rejects_changed_checkpoint_before_oracle(workflow):
    workflow["checkpoint"].write_bytes(b"modified checkpoint")
    denied = CliRunner().invoke(
        app, workflow["command"]("pilot", workflow["base"] / "pilot")
    )
    assert denied.exit_code != 0
    assert "hash mismatch" in str(denied.exception)
    assert not workflow["probe"].events
