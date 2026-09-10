import json

import pandas as pd
import pytest
import numpy as np
from dataclasses import replace
from pathlib import Path

from llm_design_bench.evaluation.publication_runner import (
    METHOD_ORDER,
    ALL_METHOD_ORDER,
    PublicationBenchmarkConfig,
    aggregate_publication_results,
    run_publication_benchmarks,
    _config_fingerprint,
)


def test_all_methods_reports_artifacts_and_tamper_detection(tmp_path):
    settings = json.loads((Path(__file__).parents[1] / "configs/all_methods_smoke.json").read_text())
    config = PublicationBenchmarkConfig(methods=ALL_METHOD_ORDER, method_configs=settings,
        seeds=(5,), functions=("booth",), include_data_mixture=False, logged_samples=16,
        recommendations=4, epochs=1, particle_steps=1, bdi_steps=1, method_steps=2, results_dir=tmp_path)
    rows = run_publication_benchmarks(config)
    assert set(rows.optimizer) == set(ALL_METHOD_ORDER)
    assert len(rows) == 14
    assert np.isfinite(rows.raw_max_utility).all()
    markdown = (tmp_path / "TABLE1_STYLE.md").read_text()
    for name in ("CbAS", "MINs", "DDOM", "GABO", "GTG", "RGD", "BONET", "DEMO", "ROOT", "SPADE"):
        assert name in markdown
    replay = run_publication_benchmarks(config)
    assert len(replay) == len(rows)
    entry = json.loads(rows.iloc[0].artifacts_json)["candidates"]
    with (tmp_path / entry["path"]).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="modified replay artifact"):
        run_publication_benchmarks(config)


def test_resume_fingerprint_covers_method_settings_and_data():
    config = PublicationBenchmarkConfig(include_data_mixture=False)
    original = _config_fingerprint(config, None, {"data": "hash1"})
    assert original != _config_fingerprint(config, None, {"data": "hash2"})
    assert original != _config_fingerprint(replace(config, methods=("cbas",)), None, {"data": "hash1"})
    assert original != _config_fingerprint(replace(config, method_configs={"coms": {"hidden_size": 64}}), None, {"data": "hash1"})


def test_publication_runner_writes_seeded_reports_and_resumes(tmp_path) -> None:
    config = PublicationBenchmarkConfig(
        seeds=(3, 5),
        functions=("ackley", "booth"),
        include_data_mixture=False,
        logged_samples=16,
        recommendations=4,
        epochs=1,
        particle_steps=1,
        bdi_steps=1,
        results_dir=tmp_path,
    )

    first = run_publication_benchmarks(config)
    second = run_publication_benchmarks(config)

    expected_rows = 2 * 2 * len(METHOD_ORDER)
    assert len(first) == expected_rows
    assert len(second) == expected_rows
    assert set(first["seed"]) == {3, 5}
    assert (first["recommendation_count"] == 4).all()
    assert (first["refnorm_d_best_score"] == 1.0).all()

    expected_files = {
        "README.md",
        "SEED_RANGES.md",
        "TABLE1_STYLE.md",
        "d_best_summary.csv",
        "rank_summary.csv",
        "raw_runs.csv",
        "run_metadata.json",
        "seed_manifest.csv",
        "seeded_benchmark_table.tex",
        "seeded_benchmark_table_se.tex",
        "seeded_benchmark_ranges.tex",
        "task_summary.csv",
    }
    assert expected_files.issubset(path.name for path in tmp_path.iterdir())

    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["seeds"] == [3, 5]
    assert metadata["recommendations"] == 4
    assert metadata["uncertainty"] == "sample_standard_deviation_ddof_1"
    assert metadata["table1_style_uncertainty"] == (
        "standard_error_over_independent_seeds"
    )

    markdown = (tmp_path / "README.md").read_text(encoding="utf-8")
    latex = (tmp_path / "seeded_benchmark_table.tex").read_text(encoding="utf-8")
    latex_se = (tmp_path / "seeded_benchmark_table_se.tex").read_text(
        encoding="utf-8"
    )
    assert "mean +/- sample SD across 2 independent seeds" in markdown
    assert "Best Logged" in markdown
    assert "COM" in markdown
    assert "BDI" in markdown
    assert "AntMorphology" not in markdown
    assert "target 1B/19,500-step fidelity" in markdown
    assert "\\begin{sidewaystable*}" in latex
    assert "target 1B/19,500-step fidelity" in latex
    assert "\\mathbf" in latex
    assert "\\underline" in latex
    assert "standard error" in latex_se
    assert "\\label{tab:seeded-publication-benchmark}" in latex
    assert "\\label{tab:seeded-publication-benchmark-se}" in latex_se


def test_publication_aggregation_uses_sample_standard_deviation() -> None:
    rows = []
    for optimizer, values in {
        "best_logged": (1.0, 1.0),
        "coms": (1.0, 3.0),
        "bdi": (0.0, 2.0),
    }.items():
        for seed, value in enumerate(values):
            rows.append(
                {
                    "suite": "synthetic",
                    "task": "ackley",
                    "display_name": "Ackley",
                    "category": "many_local_minima",
                    "category_display_name": "Many Local Minima",
                    "optimizer": optimizer,
                    "seed": seed,
                    "refnorm_max_score": value,
                    "raw_max_utility": value,
                    "refnorm_d_best_score": 1.0,
                    "d_best_utility": 1.0,
                }
            )

    summary, ranks, d_best = aggregate_publication_results(pd.DataFrame(rows))

    com = summary[summary["optimizer"] == "coms"].iloc[0]
    assert com["mean_score"] == 2.0
    assert com["std_score"] == 2**0.5
    assert com["se_score"] == 1.0
    assert com["min_score"] == 1.0
    assert com["max_score"] == 3.0
    assert com["range_score"] == 2.0
    assert ranks.loc[ranks["optimizer"] == "coms", "mean_rank"].item() == 1.0
    assert d_best["mean_score"].item() == 1.0
