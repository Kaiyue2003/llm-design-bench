from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_summary_plot(frame: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    utility_columns = [
        "refnorm_max_score",
        "refnorm_median_score",
        "refnorm_mean_score",
    ]
    utility_title = "Reference-normalized score"
    if not set(utility_columns).issubset(frame.columns):
        utility_columns = ["raw_max_utility", "raw_median_utility", "raw_mean_utility"]
        utility_title = "Target-fidelity utility"
    frame.plot.bar(
        x="optimizer",
        y=utility_columns,
        ax=axes[0],
        title=utility_title,
    )
    if utility_columns[0].startswith("refnorm_"):
        low = min(0.0, float(frame[utility_columns].min().min()))
        high = max(1.0, float(frame[utility_columns].max().max()))
        if "refnorm_d_best_score" in frame:
            d_best_values = frame["refnorm_d_best_score"].dropna().unique()
            if len(d_best_values) == 1:
                d_best = float(d_best_values[0])
                low = min(low, d_best)
                high = max(high, d_best)
                axes[0].axhline(
                    d_best,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    label="D(best)",
                )
                axes[0].legend()
        margin = max(0.05, 0.05 * (high - low))
        axes[0].set_ylim(low, high + margin)
    frame.plot.bar(
        x="optimizer",
        y=["candidate_diversity", "candidate_novelty"],
        ax=axes[1],
        title="Candidate diagnostics",
    )
    for axis in axes:
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
