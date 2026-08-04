import pandas as pd

from llm_design_bench.evaluation.report import save_summary_plot


def test_summary_plot_is_written(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "optimizer": "example",
                "raw_max_utility": 1.0,
                "raw_median_utility": 0.5,
                "raw_mean_utility": 0.4,
                "candidate_diversity": 0.2,
                "candidate_novelty": 0.1,
            }
        ]
    )
    output = tmp_path / "summary.png"
    save_summary_plot(frame, output)
    assert output.is_file()
    assert output.stat().st_size > 0
