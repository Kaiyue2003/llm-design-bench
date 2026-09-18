from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from llm_design_bench import report_cli
from llm_design_bench.evaluation.seed_types import SeedBenchmarkConfig
from llm_design_bench.evaluation.unified_report import (
    UNIFIED_REPORT_FILENAMES,
    load_legacy_publication_results,
    write_unified_report,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "reference_results" / "publication"
RUNNER = CliRunner()


@pytest.fixture
def unified_rows() -> pd.DataFrame:
    return load_legacy_publication_results(LEGACY).iloc[[0]].copy()


@pytest.fixture
def legacy_copy(tmp_path: Path) -> Path:
    source = tmp_path / "publication"
    source.mkdir()
    for name in ("raw_runs.csv", "run_metadata.json"):
        (source / name).write_bytes((LEGACY / name).read_bytes())
    return source


def _file_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _forbid_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_write(*args, **kwargs):
        pytest.fail("input conflict must be rejected before invoking the writer")

    monkeypatch.setattr(report_cli, "write_unified_report", unexpected_write)


@pytest.mark.parametrize("filename", UNIFIED_REPORT_FILENAMES)
def test_unified_rejects_every_output_name_without_changing_files(
    tmp_path: Path,
    unified_rows: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    for name in UNIFIED_REPORT_FILENAMES:
        (tmp_path / name).write_text("existing output", encoding="utf-8")
    source = tmp_path / filename
    unified_rows.to_csv(source, index=False)
    before = _file_bytes(tmp_path)
    _forbid_writer(monkeypatch)

    result = RUNNER.invoke(
        report_cli.app,
        ["from-unified", "--input-csv", str(source), "--results-dir", str(tmp_path)],
    )

    assert result.exit_code == 2, result.output
    assert "must not overwrite input file" in result.output
    assert _file_bytes(tmp_path) == before


@pytest.mark.parametrize(
    "alias_kind", ["hardlink", "output_symlink", "input_symlink", "directory_symlink"]
)
def test_unified_rejects_filesystem_aliases(
    tmp_path: Path,
    unified_rows: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    output = tmp_path / "report"
    output.mkdir()
    source = tmp_path / "input.csv"
    unified_rows.to_csv(source, index=False)
    destination = output / "README.md"
    try:
        if alias_kind == "hardlink":
            destination.hardlink_to(source)
        elif alias_kind == "output_symlink":
            destination.symlink_to(source)
        elif alias_kind == "input_symlink":
            source.replace(destination)
            source.symlink_to(destination)
        else:
            source.replace(destination)
            source = destination
            alias = tmp_path / "report-alias"
            alias.symlink_to(output, target_is_directory=True)
            output = alias
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"filesystem cannot create {alias_kind}: {exc}")
    before = source.read_bytes()
    _forbid_writer(monkeypatch)

    result = RUNNER.invoke(
        report_cli.app,
        ["from-unified", "--input-csv", str(source), "--results-dir", str(output)],
    )

    assert result.exit_code == 2, result.output
    assert "must not overwrite input file" in result.output
    assert source.read_bytes() == before
    assert destination.read_bytes() == before
    assert {path.name for path in output.iterdir()} == {"README.md"}


def test_unified_allows_same_directory_and_repeated_generation(
    tmp_path: Path, unified_rows: pd.DataFrame
) -> None:
    source = tmp_path / "source.csv"
    unified_rows.to_csv(source, index=False)
    before = source.read_bytes()

    for _ in range(2):
        result = RUNNER.invoke(
            report_cli.app,
            [
                "from-unified",
                "--input-csv",
                str(source),
                "--results-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert source.read_bytes() == before
        assert {path.name for path in tmp_path.iterdir()} == {
            "source.csv",
            ".suite.lock",
            *UNIFIED_REPORT_FILENAMES,
        }
        assert len(pd.read_csv(tmp_path / "method_seed_results.csv")) == 1


def test_unified_rejects_normalized_parent_alias(
    tmp_path: Path,
    unified_rows: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    source = tmp_path / "benchmark_table.tex"
    unified_rows.to_csv(source, index=False)
    before = _file_bytes(tmp_path)
    _forbid_writer(monkeypatch)

    result = RUNNER.invoke(
        report_cli.app,
        [
            "from-unified",
            "--input-csv",
            str(source),
            "--results-dir",
            str(child / ".."),
        ],
    )

    assert result.exit_code == 2, result.output
    assert "must not overwrite input file" in result.output
    assert _file_bytes(tmp_path) == before


@pytest.mark.parametrize("input_name", ["raw_runs.csv", "run_metadata.json"])
@pytest.mark.parametrize("output_name", UNIFIED_REPORT_FILENAMES)
def test_legacy_rejects_aliases_to_both_actual_inputs_before_writing(
    tmp_path: Path,
    legacy_copy: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    output_name: str,
) -> None:
    output = tmp_path / "report"
    output.mkdir()
    try:
        (output / output_name).hardlink_to(legacy_copy / input_name)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"filesystem cannot create a hard link: {exc}")
    before = _file_bytes(tmp_path)
    _forbid_writer(monkeypatch)

    result = RUNNER.invoke(
        report_cli.app,
        [
            "from-legacy",
            "--publication-dir",
            str(legacy_copy),
            "--results-dir",
            str(output),
        ],
    )

    assert result.exit_code == 2, result.output
    assert "must not overwrite input file" in result.output
    assert _file_bytes(tmp_path) == before


@pytest.mark.parametrize("child", ["", "nested/report"])
def test_legacy_keeps_publication_directory_immutable(
    tmp_path: Path,
    legacy_copy: Path,
    monkeypatch: pytest.MonkeyPatch,
    child: str,
) -> None:
    before = _file_bytes(tmp_path)
    _forbid_writer(monkeypatch)
    result = RUNNER.invoke(
        report_cli.app,
        [
            "from-legacy",
            "--publication-dir",
            str(legacy_copy),
            "--results-dir",
            str(legacy_copy / child),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "results-dir must be outside publication-dir" in result.output
    assert "immutable" in result.output
    assert _file_bytes(tmp_path) == before
    assert not (legacy_copy / "nested").exists()


def test_legacy_allows_external_existing_report_directory(
    tmp_path: Path, legacy_copy: Path
) -> None:
    output = tmp_path / "report"
    output.mkdir()
    before = _file_bytes(legacy_copy)
    for _ in range(2):
        result = RUNNER.invoke(
            report_cli.app,
            [
                "from-legacy",
                "--publication-dir",
                str(legacy_copy),
                "--results-dir",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert _file_bytes(legacy_copy) == before
        assert {path.name for path in output.iterdir()} == {
            ".suite.lock",
            *UNIFIED_REPORT_FILENAMES,
        }


def test_writer_still_merges_seed_shards_and_resumes_saved_rows(
    tmp_path: Path, unified_rows: pd.DataFrame
) -> None:
    first = unified_rows.copy()
    first["method_seed"] = 38
    first["phase"] = "exploratory"
    first["required_seeds_json"] = "[38,39]"
    first["provenance_json"] = "{}"
    first["logical_fingerprint"] = "first-seed"
    first["artifact_relative_dir"] = "attempts/seed-38"
    second = first.copy()
    second["method_seed"] = 39
    second["logical_fingerprint"] = "second-seed"
    second["artifact_relative_dir"] = "attempts/seed-39"
    config = SeedBenchmarkConfig(
        experiment_id=str(first.iloc[0]["experiment_id"]),
        results_dir=tmp_path,
        seeds=(38, 39),
        required_seeds=(38, 39),
        phase="exploratory",
    )

    write_unified_report(first, tmp_path, config=config)
    merged = write_unified_report(second, tmp_path, config=config)
    resumed = write_unified_report(second, tmp_path, config=config)

    assert set(merged.per_seed["method_seed"]) == {38, 39}
    pd.testing.assert_frame_equal(merged.per_seed, resumed.per_seed)
    assert len(pd.read_csv(tmp_path / "method_seed_results.csv")) == 2
