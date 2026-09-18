"""Keep main's public seed tables alongside the frozen-runtime contracts."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from llm_design_bench.evaluation.seed_rendering import (
    render_seed_summary_latex,
    render_seed_summary_markdown,
)
from llm_design_bench.evaluation.seed_statistics import summarize_seed_results


def _rows(values, *, phase="exploratory", failed_seed=None):
    seeds = list(range(38, 38 + len(values)))
    required = list(range(38, 46)) if phase == "formal" else seeds
    return pd.DataFrame(
        [
            {
                "experiment_id": "rendering-contract",
                "result_source": "unified_runner",
                "phase": phase,
                "required_seeds_json": json.dumps(required),
                "run_id": "example",
                "method_id": "example",
                "display_name": "Method_A & B",
                "method_seed": seed,
                "status": "failed" if seed == failed_seed else "success",
                "raw_max_utility": value,
                "refnorm_max_score": value,
            }
            for seed, value in zip(seeds, values, strict=True)
        ]
    )


@pytest.mark.parametrize("values", [(1.0, 3.0), (2.0, 2.0), (4.0,)])
def test_summary_retains_intervals_and_observed_range(values):
    summary = summarize_seed_results(_rows(values)).iloc[0]
    for metric in ("raw_max_utility", "refnorm_max_score"):
        assert summary[f"{metric}_n"] == len(values)
        assert summary[f"{metric}_mean"] == np.mean(values)
        assert summary[f"{metric}_min"] == min(values)
        assert summary[f"{metric}_max"] == max(values)
        assert summary[f"{metric}_range"] == max(values) - min(values)
        if len(values) > 1:
            std = np.std(values, ddof=1)
            se = std / np.sqrt(len(values))
            critical = student_t.ppf(0.975, df=len(values) - 1)
            assert summary[f"{metric}_std"] == std
            assert summary[f"{metric}_se"] == se
            assert summary[f"{metric}_ci95_low"] == np.mean(values) - critical * se
            assert summary[f"{metric}_ci95_high"] == np.mean(values) + critical * se
        else:
            for suffix in ("std", "se", "ci95_low", "ci95_high"):
                assert np.isnan(summary[f"{metric}_{suffix}"])


def test_summary_intervals_exclude_failed_seeds_and_keep_formal_coverage():
    frame = _rows((1.0, 3.0, 999.0), phase="formal", failed_seed=40)
    summary = summarize_seed_results(frame).iloc[0]
    assert summary["refnorm_max_score_mean"] == 2.0
    assert summary["refnorm_max_score_min"] == 1.0
    assert summary["refnorm_max_score_max"] == 3.0
    assert summary["successful_runs"] == 2
    assert summary["failed_runs"] == 1
    assert summary["missing_runs"] == 5
    assert summary["requested_runs"] == 8
    assert not summary["rank_eligible"]
    assert not summary["complete_seed_set"]


def test_summary_without_successes_has_no_fabricated_intervals():
    summary = summarize_seed_results(_rows((999.0,), failed_seed=38)).iloc[0]
    assert summary["refnorm_max_score_n"] == 0
    for suffix in ("mean", "std", "se", "ci95_low", "ci95_high", "min", "max", "range"):
        assert np.isnan(summary[f"refnorm_max_score_{suffix}"])


def test_seed_tables_are_pure_and_retain_uncertainty_labels():
    summary = summarize_seed_results(_rows((1.0, 3.0)))
    original = summary.copy(deep=True)
    markdown = render_seed_summary_markdown(summary)
    latex = render_seed_summary_latex(summary)
    assert "Mean +/- SE" in markdown
    assert "Sample SD | 95% CI | Observed range" in markdown
    assert "2.000 +/- 1.000" in markdown
    assert "[1.000, 3.000]" in markdown
    assert "Method\\_A \\& B" in latex
    assert "2.000 $\\pm$ 1.000" in latex
    pd.testing.assert_frame_equal(summary, original)


@pytest.mark.parametrize(
    "renderer", [render_seed_summary_markdown, render_seed_summary_latex]
)
def test_seed_table_missing_fields_fail_explicitly(renderer):
    with pytest.raises(KeyError, match="missing seed-summary columns"):
        renderer(pd.DataFrame([{"experiment_id": "incomplete"}]))


@pytest.mark.parametrize(
    "renderer", [render_seed_summary_markdown, render_seed_summary_latex]
)
def test_seed_table_unknown_single_seed_uncertainty_is_not_zero(renderer):
    rendered = renderer(summarize_seed_results(_rows((2.0,))))
    assert "--" in rendered
    assert "nan" not in rendered.lower()


def test_seed_reports_are_all_written_atomically_under_one_directory_lock(
    tmp_path, monkeypatch
):
    from llm_design_bench.evaluation import seed_persistence
    from llm_design_bench.evaluation.run_artifacts import result_directory_lock
    from llm_design_bench.evaluation.seed_runner import (
        SeedBenchmarkConfig,
        run_method_seed_benchmark,
    )
    from llm_design_bench.problem import OfflineProblem
    from llm_design_bench.tasks.synthetic_functions import SyntheticFunctionTask

    published = []
    original_atomic_bytes = seed_persistence.atomic_bytes

    def checked_atomic_bytes(path, content, *, replace):
        # A second descriptor cannot obtain the OS lock during publication.
        with (
            pytest.raises(RuntimeError, match="another worker"),
            result_directory_lock(tmp_path),
        ):
            pytest.fail("the result writer did not hold its directory lock")
        assert replace is True
        published.append(path.name)
        return original_atomic_bytes(path, content, replace=replace)

    monkeypatch.setattr(seed_persistence, "atomic_bytes", checked_atomic_bytes)
    task = SyntheticFunctionTask("booth", logged_samples=8, seed=38)
    result = run_method_seed_benchmark(
        task,
        OfflineProblem.from_task(task),
        ["best_logged"],
        reference_utility=task.logged_y,
        config=SeedBenchmarkConfig(
            seeds=(38, 39), candidate_budget=2, results_dir=tmp_path
        ),
    )
    assert set(result.per_seed["status"]) == {"success"}
    assert set(published) == {
        "method_seed_results.csv",
        "method_seed_summary.csv",
        "method_seed_table.md",
        "method_seed_table.tex",
    }
    for filename in set(published):
        assert (tmp_path / filename).is_file()
    assert (tmp_path / "method_seed_table.md").read_text(encoding="utf-8") == (
        render_seed_summary_markdown(result.summary)
    )
    assert (tmp_path / "method_seed_table.tex").read_text(encoding="utf-8") == (
        render_seed_summary_latex(result.summary)
    )
    with result_directory_lock(tmp_path):
        pass  # Publication released the lock without deleting its sidecar.
