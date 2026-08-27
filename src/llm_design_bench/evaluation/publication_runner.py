from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from llm_design_bench.evaluation.offline_runner import make_logged_percentile_split
from llm_design_bench.metrics.usefulness import usefulness_summary
from llm_design_bench.optimizers.bdi import BackwardDistillationOptimizer
from llm_design_bench.optimizers.best_logged import BestLoggedOptimizer
from llm_design_bench.optimizers.coms import ConservativeObjectiveModelOptimizer
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
METHOD_DISPLAY_NAMES = {
    "best_logged": "Best Logged",
    "coms": "COM",
    "bdi": "BDI",
}
DATA_MIXTURE_TASK = "data_recipes_stack_exchange"
DATA_MIXTURE_DISPLAY_NAME = "LLM-DM"
DATA_MIXTURE_CATEGORY = "data_mixture"
DATA_MIXTURE_CATEGORY_DISPLAY_NAME = "Data Mixture"


@dataclass(frozen=True)
class PublicationBenchmarkConfig:
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

    fingerprint = _config_fingerprint(config, data_commit)
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
        for seed in config.seeds:
            for optimizer_name in METHOD_ORDER:
                key = ("data_mixture", DATA_MIXTURE_TASK, seed, optimizer_name)
                if key in completed:
                    continue
                optimizer = _make_optimizer(optimizer_name, config, seed)
                trace = optimizer.optimize(split.task)
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
                        trace=trace,
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
            for optimizer_name in METHOD_ORDER:
                key = ("synthetic", task.name, seed, optimizer_name)
                if key in completed:
                    continue
                optimizer = _make_optimizer(optimizer_name, config, seed)
                trace = optimizer.optimize(task)
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
                        trace=trace,
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
    raw.to_csv(raw_path, index=False)
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
    )
    (config.results_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.results_dir / "README.md").write_text(
        render_markdown_report(summary, ranks, d_best, metadata),
        encoding="utf-8",
    )
    (config.results_dir / "seeded_benchmark_table.tex").write_text(
        render_latex_table(summary, ranks, d_best, metadata),
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
            mean_raw_max_utility=("raw_max_utility", "mean"),
            std_raw_max_utility=("raw_max_utility", "std"),
            trials=("seed", "nunique"),
        )
        .fillna({"std_score": 0.0, "std_raw_max_utility": 0.0})
    )
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
        raw.groupby(
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
            mean_raw_utility=("d_best_utility", "mean"),
            std_raw_utility=("d_best_utility", "std"),
            trials=("seed", "nunique"),
        )
        .fillna({"std_score": 0.0, "std_raw_utility": 0.0})
    )
    return summary, ranks, d_best


def render_markdown_report(
    summary: pd.DataFrame,
    ranks: pd.DataFrame,
    d_best: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    tasks = _task_records(summary)
    header = ["Method", *[task["display_name"] for task in tasks], "Mean rank", "Median rank"]
    alignment = ["---", *["---:" for _ in tasks], "---:", "---:"]
    lines = [
        "# Seeded Publication Benchmark",
        "",
        (
            "Normalized maximum score (100th percentile of "
            f"K={metadata['recommendations']} recommendations), reported as mean +/- "
            f"sample SD across {metadata['trial_count']} independent seeds. Higher is better."
        ),
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(alignment) + " |",
    ]

    d_best_cells = []
    for task in tasks:
        row = d_best[d_best["task"] == task["task"]].iloc[0]
        d_best_cells.append(_plain_score_cell(row["mean_score"], row["std_score"]))
    lines.append("| D(best) | " + " | ".join(d_best_cells) + " | -- | -- |")

    for optimizer in METHOD_ORDER:
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
                    std=float(row["std_score"]),
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
            f"- Synthetic functions: `{functions}`.",
            "- Uncertainty: sample standard deviation, not standard error.",
            f"- Candidate count: `K={metadata['recommendations']}` for every method and trial.",
            "",
            "## Selection And Method Provenance",
            "",
            (
                "The synthetic subset is the union of the previously committed single-seed COM and BDI "
                "winner in each test-problem category. It is intentionally performance-selected for a "
                "compact descriptive table and must not be presented as an unbiased all-task comparison."
            ),
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
            "",
        ]
    )
    return "\n".join(lines)


def render_latex_table(
    summary: pd.DataFrame,
    ranks: pd.DataFrame,
    d_best: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
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
        f"$\\pm$ sample standard deviation across {metadata['trial_count']} independent seeds. "
        "Higher is better; bold and underlined entries denote the best and second-best mean per task.}",
        "\\label{tab:seeded-publication-benchmark}",
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
        d_best_cells.append(_latex_plain_score(row["mean_score"], row["std_score"]))
    lines.append("$\\mathcal{D}$ (best) & " + " & ".join(d_best_cells) + " & -- & -- \\\\")
    lines.append("\\midrule")

    for optimizer in METHOD_ORDER:
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
                    std=float(row["std_score"]),
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
            "The synthetic subset is performance-selected from the prior exploratory sweep; "
            "the table is descriptive rather than an unbiased all-task comparison. "
            "COM and BDI are native PyTorch implementations; BDI uses the repository's RBF-kernel adaptation.",
            "\\end{minipage}",
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
    trace,
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
        "optimizer": optimizer,
        "method_display_name": METHOD_DISPLAY_NAMES[optimizer],
        "train_size": train_size,
        "train_min_percentile": train_min_percentile,
        "train_max_percentile": train_max_percentile,
        "reference_min_utility": float(reference_y.min()),
        "reference_max_utility": float(reference_y.max()),
        "d_best_utility": d_best_utility,
        "refnorm_d_best_score": refnorm_d_best_score,
        "recommendation_count": len(trace.recommendations),
        "query_count": len(trace.queried),
        "cumulative_simulated_cost": trace.cumulative_cost,
    }
    row.update(
        usefulness_summary(
            trace.recommendation_utility,
            reference_y,
            oracle_best=float(reference_y.max()),
        )
    )
    if suite == "synthetic":
        row["best_objective"] = -row["raw_max_utility"]
    else:
        row["best_objective"] = np.nan
    return row


def _make_optimizer(
    optimizer: str,
    config: PublicationBenchmarkConfig,
    seed: int,
):
    if optimizer == "best_logged":
        return BestLoggedOptimizer(recommendations=config.recommendations)
    if optimizer == "coms":
        return ConservativeObjectiveModelOptimizer(
            recommendations=config.recommendations,
            seed=seed,
            epochs=config.epochs,
            particle_steps=config.particle_steps,
        )
    if optimizer == "bdi":
        return BackwardDistillationOptimizer(
            recommendations=config.recommendations,
            seed=seed,
            steps=config.bdi_steps,
        )
    raise KeyError(f"unknown publication optimizer: {optimizer}")


def _validate_config(config: PublicationBenchmarkConfig) -> None:
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
        "torch_threads",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= config.train_min_percentile <= config.train_max_percentile <= 100:
        raise ValueError("training percentiles must satisfy 0 <= min <= max <= 100")


def _configure_torch(config: PublicationBenchmarkConfig) -> None:
    torch.set_num_threads(config.torch_threads)
    if config.deterministic:
        torch.use_deterministic_algorithms(True)


def _config_fingerprint(
    config: PublicationBenchmarkConfig,
    data_commit: str | None,
) -> str:
    payload = asdict(config)
    payload["results_dir"] = "<excluded>"
    payload["resume"] = "<excluded>"
    payload["data_recipes_root"] = "<external>" if config.data_recipes_root else None
    payload["data_recipes_commit"] = data_commit
    payload["methods"] = METHOD_ORDER
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
    return frame.to_dict(orient="records")


def _write_raw_rows(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


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
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    code_commit, code_dirty = _git_state(repository_root)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "seeds": list(config.seeds),
        "trial_count": len(config.seeds),
        "methods": list(METHOD_ORDER),
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
        "selection_policy": (
            "union of the prior single-seed top COM and top BDI task in each synthetic category"
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
        "method_provenance": {
            "coms": "https://github.com/brandontrabucco/design-baselines",
            "bdi": "https://github.com/GGchen1997/BDI",
        },
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
        return METHOD_ORDER.index(name)
    except ValueError:
        return len(METHOD_ORDER)


def _plain_score_cell(mean: float, std: float) -> str:
    return f"{mean:.3f} +/- {std:.3f}"


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
