"""Engineering-only container smoke checks, with no external data or Docker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_design_bench.evaluation.formal_methods import FORMAL_METHOD_IDS
from llm_design_bench.llmdm_cli import app
from llm_design_bench.optimizers.registry import make_method

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/container_smoke.py"
SPEC = importlib.util.spec_from_file_location("container_smoke_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_smoke_config_covers_exact_reviewed_roster() -> None:
    entries = smoke.smoke_methods()
    assert len(entries) == 27
    assert {entry["method_id"] for entry in entries} == set(FORMAL_METHOD_IDS)
    assert "spade" not in FORMAL_METHOD_IDS
    for entry in entries:
        make_method(entry["method_id"], **entry["kwargs"])


@pytest.mark.parametrize(
    "methods", [[], ["best_logged", "best_logged"], ["spade"], ["random_search"]]
)
def test_invalid_smoke_selection_writes_nothing(tmp_path, methods) -> None:
    output = tmp_path / "not-created"
    with pytest.raises(ValueError):
        smoke.run_smoke(output, method_ids=methods)
    assert not output.exists()


def test_smoke_checks_persistence_resume_and_partial_seed_coverage(
    tmp_path_factory, monkeypatch
) -> None:
    def invoke(_prefix, arguments, log):
        result = CliRunner().invoke(app, arguments)
        log.write_text(result.output, encoding="utf-8")
        if result.exception:
            raise result.exception
        assert result.exit_code == 0, result.output

    monkeypatch.setattr(smoke, "_run_cli", invoke)
    # Keep nested artifact paths below legacy Windows MAX_PATH in host checks.
    tmp_path = tmp_path_factory.mktemp("container")
    sentinel = tmp_path / "previous-results.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    output = tmp_path / "run"
    result = smoke.run_smoke(output, method_ids=["best_logged"])
    assert result == {
        "status": "passed",
        "scientific_result": False,
        "invented_data": True,
        "pilot_methods": 1,
        "formal_shard_rows": 1,
        "formal_missing_seeds": 7,
        "formal_rank_eligible": False,
        "resume_reused_attempt": True,
        "candidate_budget": 128,
    }
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert (output / "smoke_verification.json").is_file()
    with pytest.raises(FileExistsError):
        smoke.run_smoke(output, method_ids=["best_logged"])
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_failed_subprocess_preserves_diagnostic_log(tmp_path) -> None:
    log = tmp_path / "failed.log"
    with pytest.raises(RuntimeError, match=r"Smoke CLI failed \(7\)"):
        smoke._run_cli(
            [sys.executable, "-c", "print('smoke failure'); raise SystemExit(7)"],
            [],
            log,
        )
    assert "smoke failure" in log.read_text(encoding="utf-8")
