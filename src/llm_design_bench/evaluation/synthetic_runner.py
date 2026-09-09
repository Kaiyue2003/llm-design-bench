from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from llm_design_bench.metrics.diversity import pairwise_diversity
from llm_design_bench.metrics.novelty import candidate_novelty
from llm_design_bench.metrics.usefulness import (
    add_reference_normalized_score_columns,
    usefulness_summary,
)
from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import OfflineProblem, RunContext
from llm_design_bench.tasks.synthetic_functions import (
    CATEGORY_DISPLAY_NAMES,
    DEFAULT_SYNTHETIC_FUNCTIONS,
    SYNTHETIC_CATEGORIES,
    SyntheticFunctionTask,
)


OPTIMIZER_DISPLAY_NAMES = {
    "best_logged": "Best Logged",
    "offline_mlp": "Offline MLP",
    "coms": "COM",
    "com": "COM",
    "bdi": "BDI",
}


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    functions: tuple[str, ...] = DEFAULT_SYNTHETIC_FUNCTIONS
    logged_samples: int = 256
    recommendations: int = 64
    seed: int = 38
    epochs: int = 100
    particle_steps: int = 100
    bdi_steps: int = 100
    results_dir: Path = Path("results")


def run_synthetic_bo_benchmarks(
    config: SyntheticBenchmarkConfig = SyntheticBenchmarkConfig(),
    optimizers=None,
) -> pd.DataFrame:
    rows = []
    for task_index, function_name in enumerate(config.functions):
        task = SyntheticFunctionTask(
            function_name,
            logged_samples=config.logged_samples,
            seed=config.seed + task_index,
        )
        problem = OfflineProblem.from_task(task)
        task_optimizers = optimizers
        if task_optimizers is None:
            task_optimizers = ("best_logged", "coms", "bdi")

        logged_normalized = task.normalize_designs(task.logged_x.mixtures)
        for optimizer in task_optimizers:
            if isinstance(optimizer, str):
                (
                    optimizer_name,
                    recommendations,
                    recommendation_utility,
                    training_summary_json,
                ) = _run_unified_synthetic_method(
                    optimizer,
                    task,
                    problem,
                    config,
                    dataset_seed=config.seed + task_index,
                )
                query_count = 0
                cumulative_cost = 0.0
            else:
                trace = optimizer.optimize(task)
                optimizer_name = trace.name
                recommendations = trace.recommendations
                recommendation_utility = trace.recommendation_utility
                training_summary_json = None
                query_count = len(trace.queried)
                cumulative_cost = trace.cumulative_cost
            row = {
                "task": task.name,
                "display_name": task.display_name,
                "category": task.category,
                "category_display_name": task.category_display_name,
                "source": task.spec.source,
                "dimension": task.mixture_dim,
                "logged_samples": len(task.logged_y),
                "optimizer": optimizer_name,
                "global_minimum_objective": task.spec.global_minimum_value,
                "oracle_utility": task.oracle_utility,
                "method_seed": config.seed,
                "dataset_seed": config.seed + task_index,
                "split_seed": config.seed + task_index,
                "query_count": query_count,
                "recommendation_count": len(recommendations),
                "cumulative_simulated_cost": cumulative_cost,
                "training_summary_json": training_summary_json,
            }
            row.update(
                usefulness_summary(
                    recommendation_utility,
                    task.logged_y,
                    task.oracle_utility,
                )
            )
            row["best_objective"] = -row["raw_max_utility"]
            row["median_objective"] = -row["raw_median_utility"]
            row["mean_objective"] = -row["raw_mean_utility"]

            recommendations_normalized = task.normalize_designs(
                recommendations.mixtures
            )
            row["candidate_diversity"] = pairwise_diversity(recommendations_normalized)
            row["candidate_novelty"] = candidate_novelty(
                recommendations_normalized,
                logged_normalized,
            )
            rows.append(row)

    frame = add_reference_normalized_score_columns(pd.DataFrame(rows))
    config.results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.results_dir / "synthetic_bo_results.csv", index=False)
    save_synthetic_bo_summary_plot(frame, config.results_dir / "synthetic_bo_summary.png")
    save_synthetic_bo_category_plots(frame, config.results_dir)
    return frame


def _run_unified_synthetic_method(
    method_id: str,
    task,
    problem: OfflineProblem,
    config: SyntheticBenchmarkConfig,
    *,
    dataset_seed: int,
):
    kwargs = {}
    if method_id == "coms":
        kwargs = {
            "epochs": config.epochs,
            "particle_steps": config.particle_steps,
        }
    elif method_id == "bdi":
        kwargs = {"steps": config.bdi_steps}
    elif method_id != "best_logged":
        raise KeyError(f"unknown synthetic optimizer: {method_id}")
    result = make_method(method_id, **kwargs).run(
        problem,
        RunContext(
            method_seed=config.seed,
            candidate_budget=config.recommendations,
            dataset_seed=dataset_seed,
            split_seed=dataset_seed,
        ),
    )
    candidates = result.candidates.detach().cpu().numpy()
    recommendations = task.at_target_fidelity(candidates)
    utility = np.asarray(task.predict(recommendations), dtype=float)
    return (
        method_id,
        recommendations,
        utility,
        json.dumps(result.training_summary, sort_keys=True, default=str),
    )


def save_synthetic_bo_category_plots(frame: pd.DataFrame, results_dir: Path) -> None:
    category_dir = results_dir / "synthetic_categories"
    category_dir.mkdir(parents=True, exist_ok=True)
    for index, category in enumerate(SYNTHETIC_CATEGORIES, start=1):
        subset = frame[frame["category"] == category]
        if subset.empty:
            continue
        compatibility_path = results_dir / f"synthetic_bo_{category}_summary.png"
        save_synthetic_bo_summary_plot(
            subset,
            compatibility_path,
            title=CATEGORY_DISPLAY_NAMES[category],
        )
        shutil.copyfile(
            compatibility_path,
            category_dir / f"{index:02d}_{category}.png",
        )
    save_synthetic_bo_category_overview(
        frame,
        category_dir / "00_all_categories_overview.png",
    )
    save_synthetic_bo_top_task_plots(frame, results_dir, top_n=1)


def select_top_synthetic_tasks(
    frame: pd.DataFrame,
    top_n: int = 2,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    records = []
    optimizers = [
        name
        for name in dict.fromkeys(frame["optimizer"])
        if name != "best_logged"
    ]
    for category in SYNTHETIC_CATEGORIES:
        subset = frame[frame["category"] == category]
        tasks = list(dict.fromkeys(subset["display_name"]))
        for optimizer in optimizers:
            ranked = []
            for task in tasks:
                score = _score_for(subset, task, optimizer)
                ranked.append((score, task))
            ranked.sort(key=lambda item: item[0], reverse=True)
            for rank, (score, task) in enumerate(
                ranked[:top_n],
                start=1,
            ):
                task_row = subset[subset["display_name"] == task].iloc[0]
                records.append(
                    {
                        "category": category,
                        "category_display_name": CATEGORY_DISPLAY_NAMES[category],
                        "optimizer": optimizer,
                        "rank": rank,
                        "task": task_row["task"],
                        "display_name": task,
                        "normalized_best_utility": score,
                    }
                )
    return pd.DataFrame(records)


def save_synthetic_bo_top_task_plots(
    frame: pd.DataFrame,
    results_dir: Path,
    top_n: int = 2,
) -> None:
    selection = select_top_synthetic_tasks(frame, top_n=top_n)
    if selection.empty:
        return

    output_dir = results_dir / f"synthetic_categories_top{top_n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / f"top{top_n}_selection.csv", index=False)
    selection_label = "Best Task" if top_n == 1 else f"Top {top_n} Tasks"

    selected_pairs = set(zip(selection["category"], selection["display_name"]))
    selected_frame = frame[
        [
            (category, display_name) in selected_pairs
            for category, display_name in zip(
                frame["category"],
                frame["display_name"],
            )
        ]
    ].copy()

    for index, category in enumerate(SYNTHETIC_CATEGORIES, start=1):
        category_selection = selection[selection["category"] == category]
        if category_selection.empty:
            continue
        save_synthetic_bo_selection_vertical_plot(
            selected_frame,
            category_selection,
            output_dir / f"{index:02d}_{category}_top{top_n}.png",
            title=f"{CATEGORY_DISPLAY_NAMES[category]}: {selection_label} per Method",
        )

    save_synthetic_bo_selection_vertical_plot(
        selected_frame,
        selection,
        output_dir / f"00_all_categories_top{top_n}_overview.png",
        title=(
            f"{selection_label} for COM and BDI within Each "
            "Test-Problem Category"
        ),
    )


def save_synthetic_bo_selection_vertical_plot(
    frame: pd.DataFrame,
    selection: pd.DataFrame,
    path: Path,
    title: str,
) -> None:
    comparison_optimizers = [
        name
        for name in dict.fromkeys(frame["optimizer"])
        if name != "best_logged"
    ]
    panel_count = len(selection)
    if panel_count == 0 or not comparison_optimizers:
        return

    colors = dict(zip(comparison_optimizers, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    figure, axes = plt.subplots(
        panel_count,
        1,
        figsize=(11, max(3.0 * panel_count, 4.5)),
        squeeze=False,
    )
    for axis, (_, selected) in zip(axes.flat, selection.iterrows()):
        task = selected["display_name"]
        scores = [
            _score_for(frame, task, optimizer)
            for optimizer in comparison_optimizers
        ]
        y_positions = np.arange(len(comparison_optimizers))
        axis.scatter(
            scores,
            y_positions,
            color=[colors[optimizer] for optimizer in comparison_optimizers],
            s=110,
            zorder=3,
        )
        for y_position, score, optimizer in zip(
            y_positions,
            scores,
            comparison_optimizers,
        ):
            axis.plot(
                [1.0, score],
                [y_position, y_position],
                color=colors[optimizer],
                linewidth=3,
                alpha=0.7,
                zorder=2,
            )
            axis.annotate(
                f"{score:.6g}",
                (score, y_position),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        axis.axvline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Logged Maximum = 1",
        )
        axis.set_title(
            f"{selected['category_display_name']} | "
            f"Selected for {_optimizer_display_name(selected['optimizer'])}: {task}",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(
            [_optimizer_display_name(optimizer) for optimizer in comparison_optimizers]
        )
        axis.set_xlabel("Normalized Best Utility (Higher Is Better)")
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.grid(axis="x", alpha=0.25)
        minimum = min(min(scores), 1.0)
        maximum = max(max(scores), 1.0)
        padding = max((maximum - minimum) * 0.25, 1e-6)
        axis.set_xlim(
            minimum - padding,
            maximum + padding,
        )

    figure.suptitle(title, y=0.998, fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.99), h_pad=2.0)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_synthetic_bo_category_overview(
    frame: pd.DataFrame,
    path: Path,
    title: str = (
        "COM and BDI Performance Differences within Each "
        "Test-Problem Category"
    ),
) -> None:
    optimizers = list(dict.fromkeys(frame["optimizer"]))
    comparison_optimizers = [name for name in optimizers if name != "best_logged"]
    if not comparison_optimizers:
        comparison_optimizers = optimizers

    figure, axes = plt.subplots(3, 2, figsize=(18, 18))
    for axis, category in zip(axes.flat, SYNTHETIC_CATEGORIES):
        subset = frame[frame["category"] == category]
        tasks = list(dict.fromkeys(subset["display_name"]))
        y_positions = np.arange(len(tasks), dtype=float)
        height = 0.8 / max(len(comparison_optimizers), 1)
        for offset, optimizer in enumerate(comparison_optimizers):
            scores = [
                _score_for(subset, task, optimizer)
                for task in tasks
            ]
            positions = y_positions + (
                offset - (len(comparison_optimizers) - 1) / 2.0
            ) * height
            axis.barh(
                positions,
                scores,
                height=height,
                label=_optimizer_display_name(optimizer),
            )

        axis.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(CATEGORY_DISPLAY_NAMES[category])
        axis.set_yticks(y_positions)
        axis.set_yticklabels(tasks)
        axis.invert_yaxis()
        axis.set_xlabel("Normalized Best Utility (Higher Is Better)")
        axis.grid(axis="x", alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=len(labels),
        )
    figure.suptitle(title, y=0.998)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _score_for(frame: pd.DataFrame, task: str, optimizer: str) -> float:
    subset = frame[
        (frame["display_name"] == task) & (frame["optimizer"] == optimizer)
    ]
    return float(subset["refnorm_max_score"].iloc[0]) if len(subset) else np.nan


def _optimizer_display_name(optimizer: str) -> str:
    optimizer = str(optimizer)
    return OPTIMIZER_DISPLAY_NAMES.get(optimizer, optimizer.replace("_", " ").title())


def save_synthetic_bo_summary_plot(
    frame: pd.DataFrame,
    path: Path,
    title: str = "All Synthetic BO Tasks",
) -> None:
    tasks = list(dict.fromkeys(frame["display_name"]))
    optimizers = list(dict.fromkeys(frame["optimizer"]))
    x_positions = np.arange(len(tasks), dtype=float)
    width = 0.8 / max(len(optimizers), 1)

    figure_width = max(12, 0.65 * len(tasks) * max(len(optimizers), 1))
    figure, axes = plt.subplots(1, 2, figsize=(figure_width, 4.8))
    figure.suptitle(title)
    for offset, optimizer in enumerate(optimizers):
        values = []
        objective_values = []
        for task in tasks:
            score = _score_for(frame, task, optimizer)
            values.append(score)
            subset = frame[
                (frame["display_name"] == task)
                & (frame["optimizer"] == optimizer)
            ]
            objective_values.append(
                float(subset["best_objective"].iloc[0])
                if len(subset)
                else np.nan
            )
        positions = x_positions + (offset - (len(optimizers) - 1) / 2.0) * width
        label = _optimizer_display_name(optimizer)
        axes[0].bar(positions, values, width=width, label=label)
        axes[1].bar(positions, objective_values, width=width, label=label)

    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Normalized Best Utility")
    axes[0].set_ylabel("Higher Is Better")
    axes[1].set_title("Best Objective Value")
    axes[1].set_ylabel("Lower Is Better")
    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(tasks, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
