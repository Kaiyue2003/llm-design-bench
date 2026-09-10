from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

from llm_design_bench.evaluation.offline_runner import make_logged_percentile_split
from llm_design_bench.metrics.usefulness import usefulness_summary
from llm_design_bench.optimizers import get_method_metadata, make_method
from llm_design_bench.optimizers.additional_metadata import SOURCES
from llm_design_bench.evaluation.reproducibility import (
    atomic_replace, code_digest, data_assets_manifest, environment_versions,
    save_run_artifacts, verify_run_artifacts,
)
from llm_design_bench.problem import OfflineProblem, RunContext
from llm_design_bench.tasks.data_recipes import DataRecipesTask
from llm_design_bench.tasks.synthetic_functions import (
    CATEGORY_DISPLAY_NAMES,
    SyntheticFunctionTask,
)


DEFAULT_PUBLICATION_SEEDS = tuple(range(38, 46))
PUBLICATION_SYNTHETIC_FUNCTIONS = (
    "ackley",
    "schaffer2",
    "sum_different_powers",
    "matyas",
    "power_sum",
    "rosenbrock",
    "michalewicz",
    "hartmann6",
    "shekel",
)
METHOD_ORDER = ("best_logged", "coms", "bdi")
ALL_METHOD_ORDER = (*METHOD_ORDER, "offline_mlp", *SOURCES)
METHOD_DISPLAY_NAMES = {name: get_method_metadata(name).display_name for name in ALL_METHOD_ORDER}
DATA_MIXTURE_TASK = "data_recipes_stack_exchange"
DATA_MIXTURE_DISPLAY_NAME = "LLM-DM"
DATA_MIXTURE_CATEGORY = "data_mixture"
DATA_MIXTURE_CATEGORY_DISPLAY_NAME = "Data Mixture"


@dataclass(frozen=True)
class PublicationBenchmarkConfig:
    methods: tuple[str, ...] = METHOD_ORDER
    method_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    method_steps: int = 50
    device: str = "cpu"
    seeds: tuple[int, ...] = DEFAULT_PUBLICATION_SEEDS
    functions: tuple[str, ...] = PUBLICATION_SYNTHETIC_FUNCTIONS
    data_recipes_root: Path | None = None
    include_data_mixture: bool = True
    metric_index: int = 4
    logged_samples: int = 256
    recommendations: int = 128
    epochs: int = 100
    particle_steps: int = 100
    bdi_steps: int = 100
    train_min_percentile: float = 0.0
    train_max_percentile: float = 40.0
    deterministic: bool = True
    torch_threads: int = 1
    resume: bool = True
    results_dir: Path = Path("results/publication")


def run_publication_benchmarks(
    config: PublicationBenchmarkConfig = PublicationBenchmarkConfig(),
) -> pd.DataFrame:
    _validate_config(config)
    _configure_torch(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    data_task = None
    data_commit = None
    data_dirty = None
    if config.include_data_mixture:
        data_task = DataRecipesTask(
            data_recipes_root=config.data_recipes_root,
            metric_index=config.metric_index,
        )
        data_commit, data_dirty = _git_state(data_task.root)

    data_assets = data_assets_manifest(data_task.root if data_task is not None else None)
    fingerprint = _config_fingerprint(config, data_commit, data_assets)
    raw_path = config.results_dir / "raw_runs.csv"
    rows = _load_resumable_rows(raw_path, fingerprint, config.resume)
    completed = {
        (row["suite"], row["task"], int(row["seed"]), row["optimizer"])
        for row in rows
    }

    if data_task is not None:
        split = make_logged_percentile_split(
            data_task,
            min_percentile=config.train_min_percentile,
            max_percentile=config.train_max_percentile,
        )
        problem = OfflineProblem.from_task(split.task)
        for seed in config.seeds:
            for optimizer_name in config.methods:
                key = ("data_mixture", DATA_MIXTURE_TASK, seed, optimizer_name)
                if key in completed:
                    continue
                method_result, recommendation_utility = _run_registered_method(
                    optimizer_name,
                    split.task,
                    problem,
                    config,
                    seed,
                    dataset_seed=None,
                    split_seed=0,
                )
                rows.append(
                    _result_row(
                        fingerprint=fingerprint,
                        suite="data_mixture",
                        task=DATA_MIXTURE_TASK,
                        display_name=DATA_MIXTURE_DISPLAY_NAME,
                        category=DATA_MIXTURE_CATEGORY,
                        category_display_name=DATA_MIXTURE_CATEGORY_DISPLAY_NAME,
                        seed=seed,
                        task_seed=None,
                        optimizer_seed=seed,
                        optimizer=optimizer_name,
                        method_result=method_result,
                        recommendation_utility=recommendation_utility,
                        reference_y=split.reference_y,
                        d_best_utility=split.d_best_utility,
                        refnorm_d_best_score=split.refnorm_d_best_score,
                        train_size=split.train_size,
                        train_min_percentile=split.train_min_percentile,
                        train_max_percentile=split.train_max_percentile,
                    )
                )
                _write_raw_rows(rows, raw_path)

    for function_name in config.functions:
        for seed in config.seeds:
            task = SyntheticFunctionTask(
                function_name,
                logged_samples=config.logged_samples,
                seed=seed,
            )
            reference_y = np.asarray(task.logged_y, dtype=float)
            d_best_utility = float(reference_y.max())
            problem = OfflineProblem.from_task(task)
            for optimizer_name in config.methods:
                key = ("synthetic", task.name, seed, optimizer_name)
                if key in completed:
                    continue
                method_result, recommendation_utility = _run_registered_method(
                    optimizer_name,
                    task,
                    problem,
                    config,
                    seed,
                    dataset_seed=seed,
                    split_seed=seed,
                )
                rows.append(
                    _result_row(
                        fingerprint=fingerprint,
                        suite="synthetic",
                        task=task.name,
                        display_name=task.display_name,
                        category=task.category,
                        category_display_name=task.category_display_name,
                        seed=seed,
                        task_seed=seed,
                        optimizer_seed=seed,
                        optimizer=optimizer_name,
                        method_result=method_result,
                        recommendation_utility=recommendation_utility,
                        reference_y=reference_y,
                        d_best_utility=d_best_utility,
                        refnorm_d_best_score=1.0,
                        train_size=len(reference_y),
                        train_min_percentile=0.0,
                        train_max_percentile=100.0,
                    )
                )
                _write_raw_rows(rows, raw_path)

    raw = _ordered_raw_frame(pd.DataFrame(rows), config)
    _write_raw_rows(raw.to_dict("records"), raw_path)
    summary, ranks, d_best = aggregate_publication_results(raw)
    summary.to_csv(config.results_dir / "task_summary.csv", index=False)
    ranks.to_csv(config.results_dir / "rank_summary.csv", index=False)
    d_best.to_csv(config.results_dir / "d_best_summary.csv", index=False)
    _write_seed_manifest(raw, config.results_dir / "seed_manifest.csv")

    metadata = _metadata(
        config=config,
        fingerprint=fingerprint,
        data_commit=data_commit,
        data_dirty=data_dirty,
        data_assets=data_assets,
    )
    (config.results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = ["# Method provenance", "", "These runs use the configurations in raw_runs.csv. "
                  "The additional methods are adaptations, not validated reproductions of published tables.", ""]
    for method_id in config.methods:
        method = get_method_metadata(method_id)
        provenance.extend([f"## {method.display_name}", "", f"- Implementation: `{method.implementation_kind.value}`",
                           f"- Source: {method.source_url or 'native baseline'}",
                           f"- Source revision: `{method.source_commit or 'not pinned'}`",
                           *[f"- {adaptation}" for adaptation in method.adaptations], ""])
    (config.results_dir / "METHOD_PROVENANCE.md").write_text("\n".join(provenance), encoding="utf-8")
    (config.results_dir / "README.md").write_text(
        render_markdown_report(summary, ranks, d_best, metadata),
        encoding="utf-8",
    )
    (config.results_dir / "seeded_benchmark_table.tex").write_text(
        render_latex_table(summary, ranks, d_best, metadata),
        encoding="utf-8",
    )
    (config.results_dir / "TABLE1_STYLE.md").write_text(
        render_markdown_report(
            summary,
            ranks,
            d_best,
            metadata,
            uncertainty="se",
        ),
        encoding="utf-8",
    )
    (config.results_dir / "seeded_benchmark_table_se.tex").write_text(
        render_latex_table(
            summary,
            ranks,
            d_best,
            metadata,
            uncertainty="se",
        ),
        encoding="utf-8",
    )
    (config.results_dir / "SEED_RANGES.md").write_text(
        render_seed_range_markdown(summary, d_best),
        encoding="utf-8",
    )
    (config.results_dir / "seeded_benchmark_ranges.tex").write_text(
        render_seed_range_latex(summary, d_best, metadata),
        encoding="utf-8",
    )
    return raw


def aggregate_publication_results(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "suite",
        "task",
        "display_name",
        "category",
        "category_display_name",
        "optimizer",
        "seed",
        "refnorm_max_score",
        "raw_max_utility",
        "refnorm_d_best_score",
        "d_best_utility",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise KeyError(f"raw results are missing columns: {missing}")

    group_columns = [
        "suite",
        "task",
        "display_name",
        "category",
        "category_display_name",
        "optimizer",
    ]
    summary = (
        raw.groupby(group_columns, sort=False, as_index=False)
        .agg(
            mean_score=("refnorm_max_score", "mean"),
            std_score=("refnorm_max_score", "std"),
            min_score=("refnorm_max_score", "min"),
            max_score=("refnorm_max_score", "max"),
            mean_raw_max_utility=("raw_max_utility", "mean"),
            std_raw_max_utility=("raw_max_utility", "std"),
            min_raw_max_utility=("raw_max_utility", "min"),
            max_raw_max_utility=("raw_max_utility", "max"),
            trials=("seed", "nunique"),
        )
        .fillna({"std_score": 0.0, "std_raw_max_utility": 0.0})
    )
    summary["se_score"] = summary["std_score"] / np.sqrt(summary["trials"])
    summary["range_score"] = summary["max_score"] - summary["min_score"]
    summary["se_raw_max_utility"] = summary["std_raw_max_utility"] / np.sqrt(
        summary["trials"]
    )
    summary["range_raw_max_utility"] = (
        summary["max_raw_max_utility"] - summary["min_raw_max_utility"]
    )
    summary = _add_student_t_interval(summary, "score")
    summary = _add_student_t_interval(summary, "raw_max_utility")
    summary["task_rank"] = summary.groupby("task")["mean_score"].rank(
        method="average",
        ascending=False,
    )

    ranks = (
        summary.groupby("optimizer", sort=False, as_index=False)
        .agg(
            mean_rank=("task_rank", "mean"),
            median_rank=("task_rank", "median"),
            tasks=("task", "nunique"),
        )
    )
    ranks["method_order"] = ranks["optimizer"].map(_method_index)
    ranks = ranks.sort_values("method_order").drop(columns="method_order").reset_index(drop=True)

    d_best = (
        raw.drop_duplicates(["suite", "task", "seed"]).groupby(
            [
                "suite",
                "task",
                "display_name",
                "category",
                "category_display_name",
            ],
            sort=False,
            as_index=False,
        )
        .agg(
            mean_score=("refnorm_d_best_score", "mean"),
            std_score=("refnorm_d_best_score", "std"),
            min_score=("refnorm_d_best_score", "min"),
            max_score=("refnorm_d_best_score", "max"),
            mean_raw_utility=("d_best_utility", "mean"),
            std_raw_utility=("d_best_utility", "std"),
            trials=("seed", "nunique"),
        )
        .fillna({"std_score": 0.0, "std_raw_utility": 0.0})
    )
    d_best["se_score"] = d_best["std_score"] / np.sqrt(d_best["trials"])
    d_best["range_score"] = d_best["max_score"] - d_best["min_score"]
    d_best = _add_student_t_interval(d_best, "score")
    return summary, ranks, d_best


def render_markdown_report(
    summary: pd.DataFrame,
    ranks: pd.DataFrame,
    d_best: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    uncertainty: str = "std",
) -> str:
    error_column, uncertainty_label = _uncertainty_definition(uncertainty)
    tasks = _task_records(summary)
    header = ["Method", *[task["display_name"] for task in tasks], "Mean rank", "Median rank"]
    alignment = ["---", *["---:" for _ in tasks], "---:", "---:"]
    lines = [
        "# Seeded Publication Benchmark",
        "",
        (
            "Normalized maximum score (100th percentile of "
            f"K={metadata['recommendations']} recommendations), reported as mean +/- "
            f"{uncertainty_label} across {metadata['trial_count']} independent seeds. Higher is better."
        ),
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(alignment) + " |",
    ]

    d_best_cells = []
    for task in tasks:
        row = d_best[d_best["task"] == task["task"]].iloc[0]
        d_best_cells.append(_plain_score_cell(row["mean_score"], row[error_column]))
    lines.append("| D(best) | " + " | ".join(d_best_cells) + " | -- | -- |")

    for optimizer in _summary_method_order(summary):
        cells = []
        for task in tasks:
            row = summary[
                (summary["task"] == task["task"])
                & (summary["optimizer"] == optimizer)
            ].iloc[0]
            cells.append(
                _markdown_score_cell(
                    summary,
                    task=task["task"],
                    optimizer=optimizer,
                    mean=float(row["mean_score"]),
                    std=float(row[error_column]),
                )
            )
        rank = ranks[ranks["optimizer"] == optimizer].iloc[0]
        lines.append(
            "| "
            + METHOD_DISPLAY_NAMES[optimizer]
            + " | "
            + " | ".join(cells)
            + f" | {rank['mean_rank']:.2f} | {rank['median_rank']:.2f} |"
        )

    seeds = ", ".join(str(seed) for seed in metadata["seeds"])
    functions = ", ".join(metadata["synthetic_functions"])
    lines.extend(
        [
            "",
            "Bold is best and underlining is second best within each task, based on the mean score.",
            "",
            "## Experiment Contract",
            "",
            f"- Seeds: `{seeds}`. The synthetic logged-dataset seed and optimizer seed both equal the trial seed.",
            "- Data-mixture logged observations are fixed upstream data; only the optimizer seed varies by trial.",
            (
                "- Score: `(generated best utility - full logged minimum utility) / "
                "(full logged maximum utility - full logged minimum utility)`. Scores above 1 are allowed."
            ),
            (
                f"- Data-mixture training visibility: utility percentiles "
                f"[{metadata['train_min_percentile']:.0f}, {metadata['train_max_percentile']:.0f}]."
            ),
            (
                "- `D(best)` is the best optimizer-visible logged utility. LLM-DM logged runs retain "
                "their recorded model scale and training step, while method recommendations are "
                "evaluated at the target 1B/19,500-step fidelity."
            ),
            f"- Synthetic functions: `{functions}`.",
            f"- Displayed uncertainty: {uncertainty_label}.",
            "- `task_summary.csv` also records sample SD, standard error, 95% Student-t CI, and observed seed range.",
            f"- Candidate count: `K={metadata['recommendations']}` for every method and trial.",
            "",
            "## Selection And Method Provenance",
            "",
            metadata.get("selection_policy", "Tasks were selected by the caller."),
            "",
            "Additional methods are continuous PyTorch adaptations. Published-result parity is not established. "
            "Exact source revisions, settings and substitutions are in METHOD_PROVENANCE.md and run_metadata.json.",
            "",
            (
                "COM is a native PyTorch reimplementation of the conservative objective-model structure "
                "from [design-baselines](https://github.com/brandontrabucco/design-baselines)."
            ),
            (
                "BDI is the repository's native PyTorch/RBF-kernel adaptation of the forward/backward "
                "distillation structure in the [official BDI repository](https://github.com/GGchen1997/BDI). "
                "It does not claim numerical equivalence to the original JAX/Neural Tangents implementation."
            ),
            "",
            "## Reproduction Files",
            "",
            "- [Raw per-seed runs](raw_runs.csv)",
            "- [Aggregated task summary](task_summary.csv)",
            "- [Rank summary](rank_summary.csv)",
            "- [Seed manifest](seed_manifest.csv)",
            "- [Run metadata](run_metadata.json)",
            "- [Overleaf table](seeded_benchmark_table.tex)",
            "- [Table 1-style mean +/- SE report](TABLE1_STYLE.md)",
            "- [Table 1-style Overleaf table](seeded_benchmark_table_se.tex)",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex_table(
    summary: pd.DataFrame,
    ranks: pd.DataFrame,
    d_best: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    uncertainty: str = "std",
) -> str:
    error_column, uncertainty_label = _uncertainty_definition(uncertainty)
    table_label = (
        "tab:seeded-publication-benchmark-se"
        if uncertainty == "se"
        else "tab:seeded-publication-benchmark"
    )
    tasks = _task_records(summary)
    groups = _task_groups(tasks)
    column_spec = "l" + "c" * len(tasks) + "cc"
    group_headers = ["Method"]
    start_column = 2
    cmidrules = []
    for group_name, group_tasks in groups:
        span = len(group_tasks)
        group_headers.append(f"\\multicolumn{{{span}}}{{c}}{{{_latex_escape(group_name)}}}")
        cmidrules.append(f"\\cmidrule(lr){{{start_column}-{start_column + span - 1}}}")
        start_column += span
    group_headers.append("\\multicolumn{2}{c}{Rank}")
    cmidrules.append(f"\\cmidrule(lr){{{start_column}-{start_column + 1}}}")

    task_headers = ["", *[_latex_escape(task["display_name"]) for task in tasks], "Mean", "Median"]
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx,rotating}",
        "\\begin{sidewaystable*}[t]",
        "\\centering",
        "\\caption{Reference-normalized maximum score (100th percentile of "
        f"$K={metadata['recommendations']}$ recommendations). Values are mean "
        f"$\\pm$ {uncertainty_label} across {metadata['trial_count']} independent seeds. "
        "Higher is better; bold and underlined entries denote the best and second-best mean per task.}",
        f"\\label{{{table_label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(group_headers) + " \\\\",
        " ".join(cmidrules),
        " & ".join(task_headers) + " \\\\",
        "\\midrule",
    ]

    d_best_cells = []
    for task in tasks:
        row = d_best[d_best["task"] == task["task"]].iloc[0]
        d_best_cells.append(_latex_plain_score(row["mean_score"], row[error_column]))
    lines.append("$\\mathcal{D}$ (best) & " + " & ".join(d_best_cells) + " & -- & -- \\\\")
    lines.append("\\midrule")

    for optimizer in _summary_method_order(summary):
        cells = []
        for task in tasks:
            row = summary[
                (summary["task"] == task["task"])
                & (summary["optimizer"] == optimizer)
            ].iloc[0]
            cells.append(
                _latex_score_cell(
                    summary,
                    task=task["task"],
                    optimizer=optimizer,
                    mean=float(row["mean_score"]),
                    std=float(row[error_column]),
                )
            )
        rank = ranks[ranks["optimizer"] == optimizer].iloc[0]
        lines.append(
            _latex_escape(METHOD_DISPLAY_NAMES[optimizer])
            + " & "
            + " & ".join(cells)
            + f" & {rank['mean_rank']:.2f} & {rank['median_rank']:.2f} \\\\"
        )

    seed_text = ", ".join(str(seed) for seed in metadata["seeds"])
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\par\\vspace{0.5em}",
            "\\begin{minipage}{0.98\\textwidth}",
            "\\footnotesize",
            "Scores use the full logged reference range: "
            "$\\bigl(u_{\\max}^{\\mathrm{cand}}-u_{\\min}^{\\mathcal{D}}\\bigr) / "
            "\\bigl(u_{\\max}^{\\mathcal{D}}-u_{\\min}^{\\mathcal{D}}\\bigr)$. "
            "Scores above one indicate improvement over the logged maximum. "
            f"Seeds: {seed_text}. "
            "For LLM-DM, $\\mathcal{D}$ (best) uses the recorded logged fidelity, whereas method "
            "recommendations are evaluated at the target 1B/19,500-step fidelity. "
            "Task selection and implementation adaptations are recorded in the accompanying run manifest. "
            "Results for additional methods do not establish published-result parity.",
            "\\end{minipage}",
            "\\end{sidewaystable*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_seed_range_markdown(
    summary: pd.DataFrame,
    d_best: pd.DataFrame,
) -> str:
    tasks = _task_records(summary)
    lines = [
        "# Observed Seed Ranges",
        "",
        "Each cell is [minimum, maximum] over successful independent seed runs.",
        "",
        "| Method | " + " | ".join(task["display_name"] for task in tasks) + " |",
        "| --- | " + " | ".join("---:" for _ in tasks) + " |",
    ]
    reference_cells = [
        _plain_range_cell(
            d_best[d_best["task"] == task["task"]].iloc[0]["min_score"],
            d_best[d_best["task"] == task["task"]].iloc[0]["max_score"],
        )
        for task in tasks
    ]
    lines.append("| D(best) | " + " | ".join(reference_cells) + " |")
    for optimizer in _summary_method_order(summary):
        cells = []
        for task in tasks:
            row = summary[
                (summary["task"] == task["task"])
                & (summary["optimizer"] == optimizer)
            ].iloc[0]
            cells.append(_plain_range_cell(row["min_score"], row["max_score"]))
        lines.append(
            f"| {METHOD_DISPLAY_NAMES[optimizer]} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_seed_range_latex(
    summary: pd.DataFrame,
    d_best: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    tasks = _task_records(summary)
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx,rotating}",
        "\\begin{sidewaystable*}[t]",
        "\\centering",
        "\\caption{Observed minimum and maximum reference-normalized scores "
        f"across {metadata['trial_count']} independent seeds.}}",
        "\\label{tab:seeded-publication-ranges}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(tasks) + "}",
        "\\toprule",
        "Method & "
        + " & ".join(_latex_escape(task["display_name"]) for task in tasks)
        + " \\\\",
        "\\midrule",
    ]
    reference_cells = [
        _latex_range_cell(
            d_best[d_best["task"] == task["task"]].iloc[0]["min_score"],
            d_best[d_best["task"] == task["task"]].iloc[0]["max_score"],
        )
        for task in tasks
    ]
    lines.append(
        "$\\mathcal{D}$ (best) & " + " & ".join(reference_cells) + " \\\\"
    )
    lines.append("\\midrule")
    for optimizer in _summary_method_order(summary):
        cells = []
        for task in tasks:
            row = summary[
                (summary["task"] == task["task"])
                & (summary["optimizer"] == optimizer)
            ].iloc[0]
            cells.append(_latex_range_cell(row["min_score"], row["max_score"]))
        lines.append(
            _latex_escape(METHOD_DISPLAY_NAMES[optimizer])
            + " & "
            + " & ".join(cells)
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{sidewaystable*}",
            "",
        ]
    )
    return "\n".join(lines)


def _result_row(
    *,
    fingerprint: str,
    suite: str,
    task: str,
    display_name: str,
    category: str,
    category_display_name: str,
    seed: int,
    task_seed: int | None,
    optimizer_seed: int,
    optimizer: str,
    method_result,
    recommendation_utility: np.ndarray,
    reference_y: np.ndarray,
    d_best_utility: float,
    refnorm_d_best_score: float,
    train_size: int,
    train_min_percentile: float,
    train_max_percentile: float,
) -> dict[str, Any]:
    reference_y = np.asarray(reference_y, dtype=float)
    row: dict[str, Any] = {
        "config_fingerprint": fingerprint,
        "suite": suite,
        "task": task,
        "display_name": display_name,
        "category": category,
        "category_display_name": category_display_name,
        "seed": seed,
        "task_seed": task_seed,
        "optimizer_seed": optimizer_seed,
        "method_seed": optimizer_seed,
        "dataset_seed": task_seed,
        "split_seed": task_seed if task_seed is not None else 0,
        "optimizer": optimizer,
        "method_id": optimizer,
        "method_display_name": METHOD_DISPLAY_NAMES[optimizer],
        "implementation_kind": get_method_metadata(
            optimizer
        ).implementation_kind.value,
        "original_framework": get_method_metadata(optimizer).original_framework,
        "implementation_framework": get_method_metadata(
            optimizer
        ).implementation_framework,
        "source_url": get_method_metadata(optimizer).source_url,
        "source_commit": get_method_metadata(optimizer).source_commit,
        "paper_url": get_method_metadata(optimizer).paper_url,
        "adaptations_json": json.dumps(get_method_metadata(optimizer).adaptations),
        "artifacts_json": json.dumps(method_result.diagnostics["reproduction_artifacts"], sort_keys=True),
        "train_size": train_size,
        "train_min_percentile": train_min_percentile,
        "train_max_percentile": train_max_percentile,
        "reference_min_utility": float(reference_y.min()),
        "reference_max_utility": float(reference_y.max()),
        "d_best_utility": d_best_utility,
        "refnorm_d_best_score": refnorm_d_best_score,
        "recommendation_count": len(method_result.candidates),
        "query_count": 0,
        "cumulative_simulated_cost": 0.0,
        "training_summary_json": json.dumps(
            method_result.training_summary,
            sort_keys=True,
            default=str,
        ),
        "resolved_method_config_json": json.dumps(
            method_result.training_summary.get("resolved_method_config", {}),
            sort_keys=True,
            default=str,
        ),
        "diagnostics_json": json.dumps(
            method_result.diagnostics,
            sort_keys=True,
            default=str,
        ),
    }
    row.update(
        usefulness_summary(
            recommendation_utility,
            reference_y,
            oracle_best=float(reference_y.max()),
        )
    )
    if suite == "synthetic":
        row["best_objective"] = -row["raw_max_utility"]
    else:
        row["best_objective"] = np.nan
    return row


def _run_registered_method(
    method_id: str,
    evaluator_task,
    problem: OfflineProblem,
    config: PublicationBenchmarkConfig,
    seed: int,
    *,
    dataset_seed: int | None,
    split_seed: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    kwargs: dict[str, Any] = {}
    if method_id in {"coms", "offline_mlp"}:
        kwargs = {
            "epochs": config.epochs,
            "particle_steps": config.particle_steps,
        }
    elif method_id == "bdi":
        kwargs = {"steps": config.bdi_steps}
    elif method_id in SOURCES:
        kwargs = {"epochs": config.epochs, "steps": config.method_steps}
    elif method_id != "best_logged":
        raise KeyError(f"unknown publication optimizer: {method_id}")
    kwargs.update(config.method_configs.get(method_id, {}))
    result = make_method(method_id, **kwargs).run(
        problem,
        RunContext(
            method_seed=seed,
            candidate_budget=config.recommendations,
            device=config.device,
            dataset_seed=dataset_seed,
            split_seed=split_seed,
        ),
    )
    candidates = result.candidates.detach().cpu().numpy()
    batch = evaluator_task.at_target_fidelity(candidates)
    utility = np.asarray(evaluator_task.predict(batch), dtype=float)
    if utility.shape != (config.recommendations,) or not np.isfinite(utility).all():
        raise ValueError("evaluator returned invalid recommendation utilities")
    artifacts = save_run_artifacts(config.results_dir, problem, result, utility,
                                   method_id=method_id, method_seed=seed, dataset_seed=dataset_seed)
    result = replace(result, diagnostics={**result.diagnostics, "reproduction_artifacts": artifacts})
    return result, utility


def _validate_config(config: PublicationBenchmarkConfig) -> None:
    if not config.methods or len(set(config.methods)) != len(config.methods):
        raise ValueError("methods must be nonempty and unique")
    if set(config.methods).difference(ALL_METHOD_ORDER):
        raise ValueError("unknown publication method")
    if set(config.method_configs).difference(config.methods):
        raise ValueError("method_configs contains an unselected method")
    if torch.device(config.device).type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if torch.device(config.device).type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected but is unavailable")
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be unique")
    if not config.functions and not config.include_data_mixture:
        raise ValueError("select at least one benchmark task")
    for name in (
        "logged_samples",
        "recommendations",
        "epochs",
        "particle_steps",
        "bdi_steps",
        "method_steps",
        "torch_threads",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= config.train_min_percentile <= config.train_max_percentile <= 100:
        raise ValueError("training percentiles must satisfy 0 <= min <= max <= 100")


def _configure_torch(config: PublicationBenchmarkConfig) -> None:
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(config.deterministic)


def _config_fingerprint(
    config: PublicationBenchmarkConfig,
    data_commit: str | None,
    data_assets: dict[str, str] | None = None,
) -> str:
    payload = asdict(config)
    payload["results_dir"] = "<excluded>"
    payload["resume"] = "<excluded>"
    payload["data_recipes_root"] = "<external>" if config.data_recipes_root else None
    payload["data_recipes_commit"] = data_commit
    payload["methods"] = config.methods
    payload["code_sha256"] = code_digest()
    payload["dependency_versions"] = environment_versions()
    payload["data_assets"] = data_assets or {}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _load_resumable_rows(
    path: Path,
    fingerprint: str,
    resume: bool,
) -> list[dict[str, Any]]:
    if not resume or not path.is_file():
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    fingerprints = set(frame.get("config_fingerprint", pd.Series(dtype=str)).dropna())
    if fingerprints != {fingerprint}:
        raise ValueError(
            "existing raw_runs.csv was produced by a different configuration; "
            "use --no-resume or choose another results directory"
        )
    if "artifacts_json" not in frame:
        raise ValueError("existing results have no replay artifacts; choose a new directory")
    for artifacts in frame["artifacts_json"]:
        verify_run_artifacts(path.parent, artifacts)
    return frame.to_dict(orient="records")


def _write_raw_rows(rows: list[dict[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    atomic_replace(temporary, path)


def _ordered_raw_frame(
    frame: pd.DataFrame,
    config: PublicationBenchmarkConfig,
) -> pd.DataFrame:
    task_order = []
    if config.include_data_mixture:
        task_order.append(DATA_MIXTURE_TASK)
    task_order.extend(config.functions)
    ordered = frame.copy()
    ordered["_task_order"] = ordered["task"].map(
        {task: index for index, task in enumerate(task_order)}
    )
    ordered["_method_order"] = ordered["optimizer"].map(_method_index)
    ordered = ordered.sort_values(["_task_order", "seed", "_method_order"])
    return ordered.drop(columns=["_task_order", "_method_order"]).reset_index(drop=True)


def _write_seed_manifest(raw: pd.DataFrame, path: Path) -> None:
    columns = [
        "suite",
        "task",
        "display_name",
        "seed",
        "task_seed",
        "optimizer_seed",
    ]
    manifest = raw[columns].drop_duplicates().sort_values(["suite", "task", "seed"])
    manifest.to_csv(path, index=False)


def _metadata(
    *,
    config: PublicationBenchmarkConfig,
    fingerprint: str,
    data_commit: str | None,
    data_dirty: bool | None,
    data_assets: dict[str, str] | None = None,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    code_commit, code_dirty = _git_state(repository_root)
    if code_commit is None:
        code_commit = os.environ.get("LLM_DESIGN_BENCH_VCS_REF")
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "seeds": list(config.seeds),
        "trial_count": len(config.seeds),
        "methods": list(config.methods),
        "method_configs": config.method_configs,
        "method_steps": config.method_steps,
        "device": config.device,
        "code_sha256": code_digest(),
        "data_asset_sha256": data_assets or {},
        "synthetic_functions": list(config.functions),
        "include_data_mixture": config.include_data_mixture,
        "data_mixture_metric_index": config.metric_index,
        "data_mixture_metric": "stack_exchange_cross_entropy" if config.metric_index == 4 else None,
        "train_min_percentile": config.train_min_percentile,
        "train_max_percentile": config.train_max_percentile,
        "logged_samples": config.logged_samples,
        "recommendations": config.recommendations,
        "epochs": config.epochs,
        "particle_steps": config.particle_steps,
        "bdi_steps": config.bdi_steps,
        "deterministic_algorithms": config.deterministic,
        "torch_threads": config.torch_threads,
        "uncertainty": "sample_standard_deviation_ddof_1",
        "table1_style_uncertainty": "standard_error_over_independent_seeds",
        "confidence_interval": "two_sided_95_percent_student_t",
        "observed_range": "minimum_and_maximum_seed_estimates",
        "selection_policy": (
            "Performance-selected union of prior single-seed COM and BDI category winners; descriptive comparison only."
            if config.functions == PUBLICATION_SYNTHETIC_FUNCTIONS
            else "Explicit caller-selected synthetic tasks; no claim of an unbiased all-task comparison."
        ),
        "code_git_commit": code_commit,
        "code_git_dirty": code_dirty,
        "data_recipes_git_commit": data_commit,
        "data_recipes_git_dirty": data_dirty,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": {
            name: _package_version(name)
            for name in (
                "llm-design-bench",
                "torch",
                "numpy",
                "pandas",
                "scikit-learn",
                "bayeso-benchmarks",
            )
        },
        "method_provenance": {name: asdict(get_method_metadata(name)) for name in config.methods},
    }


def _git_state(path: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _add_student_t_interval(
    frame: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    result = frame.copy()
    standard_error = result[f"se_{metric}"]
    critical = result["trials"].map(
        lambda count: (
            float(student_t.ppf(0.975, df=int(count) - 1))
            if int(count) > 1
            else 0.0
        )
    )
    result[f"ci95_low_{metric}"] = result[f"mean_{metric}"] - (
        critical * standard_error
    )
    result[f"ci95_high_{metric}"] = result[f"mean_{metric}"] + (
        critical * standard_error
    )
    return result


def _uncertainty_definition(uncertainty: str) -> tuple[str, str]:
    if uncertainty == "std":
        return "std_score", "sample SD"
    if uncertainty == "se":
        return "se_score", "standard error"
    raise ValueError("uncertainty must be 'std' or 'se'")


def _task_records(summary: pd.DataFrame) -> list[dict[str, str]]:
    return (
        summary[["task", "display_name", "category", "category_display_name"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )


def _task_groups(tasks: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]]]]:
    groups: list[tuple[str, list[dict[str, str]]]] = []
    for task in tasks:
        if groups and groups[-1][0] == task["category_display_name"]:
            groups[-1][1].append(task)
        else:
            groups.append((task["category_display_name"], [task]))
    return groups


def _method_index(name: str) -> int:
    try:
        return ALL_METHOD_ORDER.index(name)
    except ValueError:
        return len(ALL_METHOD_ORDER)


def _summary_method_order(summary: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(summary["optimizer"].unique(), key=_method_index))


def _plain_score_cell(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


def _plain_range_cell(minimum: float, maximum: float) -> str:
    return f"[{minimum:.3f}, {maximum:.3f}]"


def _markdown_score_cell(
    summary: pd.DataFrame,
    *,
    task: str,
    optimizer: str,
    mean: float,
    std: float,
) -> str:
    value = _plain_score_cell(mean, std)
    position = _rank_position(summary, task, optimizer)
    if position == 1:
        return f"**{value}**"
    if position == 2:
        return f"<u>{value}</u>"
    return value


def _latex_plain_score(mean: float, std: float) -> str:
    return f"${mean:.3f} \\pm {std:.3f}$"


def _latex_range_cell(minimum: float, maximum: float) -> str:
    return "$[" + f"{minimum:.3f},\\,{maximum:.3f}" + "]$"


def _latex_score_cell(
    summary: pd.DataFrame,
    *,
    task: str,
    optimizer: str,
    mean: float,
    std: float,
) -> str:
    value = f"{mean:.3f} \\pm {std:.3f}"
    position = _rank_position(summary, task, optimizer)
    if position == 1:
        return f"$\\mathbf{{{value}}}$"
    if position == 2:
        return f"$\\underline{{{value}}}$"
    return f"${value}$"


def _rank_position(summary: pd.DataFrame, task: str, optimizer: str) -> int:
    subset = summary[summary["task"] == task].sort_values(
        ["mean_score", "optimizer"],
        ascending=[False, True],
    )
    ordered = list(subset["optimizer"])
    return ordered.index(optimizer) + 1


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
    return "".join(replacements.get(character, character) for character in str(value))
