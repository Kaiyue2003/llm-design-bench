"""Schedulers must see failures for the requested shard, not unrelated history."""

from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from llm_design_bench import llmdm_cli


@pytest.mark.parametrize("status,expected_code", [("success", 0), ("failed", 1)])
def test_cli_exit_status_tracks_selected_shard(
    tmp_path, monkeypatch, status, expected_code
):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(llmdm_cli, "load_data_manifest", lambda *a, **k: object())
    monkeypatch.setattr(
        llmdm_cli,
        "load_method_plan",
        lambda *a, **k: {
            "methods": [{"run_id": "offline_mlp"}, {"run_id": "spade"}],
            "shared_settings": {
                "pilot_seeds": [0],
                "formal_seeds": list(range(38, 46)),
            },
        },
    )
    frame = pd.DataFrame(
        [
            {
                "run_id": "offline_mlp",
                "method_seed": 0,
                "task_id": "data_recipes_stack_exchange",
                "status": status,
            },
            {
                "run_id": "spade",
                "method_seed": 0,
                "task_id": "data_recipes_stack_exchange",
                "status": "failed",
            },
            {
                "run_id": "offline_mlp",
                "method_seed": 0,
                "task_id": "data_recipes_stack_exchange_1b",
                "status": "failed",
            },
        ]
    )
    monkeypatch.setattr(
        llmdm_cli,
        "run_frozen_experiment",
        lambda *a, **k: SimpleNamespace(
            per_seed=frame,
            summary=pd.DataFrame({"status": [status]}),
        ),
    )
    result = CliRunner().invoke(
        llmdm_cli.app,
        [
            "run",
            "--data-recipes-root",
            str(tmp_path),
            "--data-bundle",
            str(tmp_path),
            "--plan",
            str(plan_path),
            "--phase",
            "pilot",
            "--results-dir",
            str(tmp_path),
            "--run-id",
            "offline_mlp",
            "--setting",
            "multi_scale",
        ],
    )
    assert result.exit_code == expected_code, result.output
