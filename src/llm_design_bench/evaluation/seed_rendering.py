"""Pure Markdown and LaTeX rendering for the public seeded-report API."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def render_seed_summary_markdown(
    summary: pd.DataFrame,
    *,
    metric: str = "refnorm_max_score",
) -> str:
    """Render an auditable seed table with both uncertainty and observed range."""

    _validate_summary_metric(summary, metric)
    lines = [
        "# Seeded Method Results",
        "",
        (
            "Primary estimate is mean +/- standard error across successful seeds. "
            "Sample SD, the two-sided 95% Student-t confidence interval, and the "
            "observed seed range are shown separately."
        ),
        "",
        "| Experiment | Method | Mean +/- SE | Sample SD | 95% CI | Observed range | Runs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["experiment_id"]),
                    str(row["display_name"]),
                    _mean_error_cell(row, metric, "se"),
                    _number(row[f"{metric}_std"]),
                    _interval_cell(row, metric, "ci95"),
                    _interval_cell(row, metric, "observed"),
                    f"{int(row['successful_runs'])}/{int(row['requested_runs'])}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Metric: `{metric}`. Confidence intervals describe uncertainty in the seed mean; ",
            "the observed range reports the minimum and maximum realized seed scores.",
            "",
        ]
    )
    return "\n".join(lines)


def render_seed_summary_latex(
    summary: pd.DataFrame,
    *,
    metric: str = "refnorm_max_score",
) -> str:
    """Render a compact booktabs table suitable for direct Overleaf inclusion."""

    _validate_summary_metric(summary, metric)
    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table*}[t]",
        "\\centering",
        (
            "\\caption{Seeded offline optimization results. The primary estimate is mean "
            "$\\pm$ standard error; SD is the sample standard deviation, CI is a two-sided "
            "95\\% Student-$t$ interval, and range is the observed minimum and maximum.}"
        ),
        "\\label{tab:seeded-method-results}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Experiment & Method & Mean $\\pm$ SE & SD & 95\\% CI & Range & Runs \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            " & ".join(
                (
                    _latex_escape(str(row["experiment_id"])),
                    _latex_escape(str(row["display_name"])),
                    _mean_error_cell(row, metric, "se", latex=True),
                    _number(row[f"{metric}_std"]),
                    _interval_cell(row, metric, "ci95", latex=True),
                    _interval_cell(row, metric, "observed", latex=True),
                    f"{int(row['successful_runs'])}/{int(row['requested_runs'])}",
                )
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_summary_metric(summary: pd.DataFrame, metric: str) -> None:
    required = {
        "experiment_id",
        "display_name",
        "successful_runs",
        "requested_runs",
        f"{metric}_mean",
        f"{metric}_std",
        f"{metric}_se",
        f"{metric}_ci95_low",
        f"{metric}_ci95_high",
        f"{metric}_min",
        f"{metric}_max",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise KeyError(f"missing seed-summary columns: {missing}")


def _mean_error_cell(
    row: pd.Series,
    metric: str,
    error: str,
    *,
    latex: bool = False,
) -> str:
    separator = " $\\pm$ " if latex else " +/- "
    return (
        _number(row[f"{metric}_mean"]) + separator + _number(row[f"{metric}_{error}"])
    )


def _interval_cell(
    row: pd.Series,
    metric: str,
    interval: str,
    *,
    latex: bool = False,
) -> str:
    if interval == "ci95":
        low = row[f"{metric}_ci95_low"]
        high = row[f"{metric}_ci95_high"]
    elif interval == "observed":
        low = row[f"{metric}_min"]
        high = row[f"{metric}_max"]
    else:
        raise ValueError(f"unknown interval {interval!r}")
    left, right = ("$[", "]$") if latex else ("[", "]")
    return f"{left}{_number(low)}, {_number(high)}{right}"


def _number(value: Any) -> str:
    numeric = float(value)
    return "--" if not math.isfinite(numeric) else f"{numeric:.3f}"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(character, character) for character in value)
