import pandas as pd
import matplotlib.image as mpimg

from llm_design_bench.evaluation.synthetic_runner import (
    SyntheticBenchmarkConfig,
    run_synthetic_bo_benchmarks,
    save_synthetic_bo_top_task_plots,
    select_top_synthetic_tasks,
)
from llm_design_bench.optimizers.best_logged import BestLoggedOptimizer
from llm_design_bench.tasks.synthetic_functions import SYNTHETIC_CATEGORIES


def test_synthetic_runner_writes_results(tmp_path) -> None:
    frame = run_synthetic_bo_benchmarks(
        SyntheticBenchmarkConfig(
            functions=("ackley", "booth"),
            logged_samples=24,
            recommendations=4,
            seed=11,
            results_dir=tmp_path,
        ),
        optimizers=(BestLoggedOptimizer(recommendations=4),),
    )

    assert set(frame["task"]) == {"ackley", "booth"}
    assert "refnorm_max_score" in frame.columns
    assert "best_objective" in frame.columns
    assert (tmp_path / "synthetic_bo_results.csv").is_file()
    assert (tmp_path / "synthetic_bo_summary.png").is_file()
    assert (tmp_path / "synthetic_bo_many_local_minima_summary.png").is_file()
    assert (tmp_path / "synthetic_bo_plate_shaped_summary.png").is_file()
    category_dir = tmp_path / "synthetic_categories"
    assert (category_dir / "00_all_categories_overview.png").is_file()
    assert (category_dir / "01_many_local_minima.png").is_file()
    assert (category_dir / "03_plate_shaped.png").is_file()


def test_synthetic_runner_writes_one_plot_per_category(tmp_path) -> None:
    functions = tuple(names[0] for names in SYNTHETIC_CATEGORIES.values())
    run_synthetic_bo_benchmarks(
        SyntheticBenchmarkConfig(
            functions=functions,
            logged_samples=16,
            recommendations=2,
            seed=13,
            results_dir=tmp_path,
        ),
        optimizers=(BestLoggedOptimizer(recommendations=2),),
    )

    for index, category in enumerate(SYNTHETIC_CATEGORIES, start=1):
        assert (tmp_path / f"synthetic_bo_{category}_summary.png").is_file()
        assert (
            tmp_path / "synthetic_categories" / f"{index:02d}_{category}.png"
        ).is_file()


def test_top_task_report_selects_two_per_method_and_category(tmp_path) -> None:
    rows = []
    for category, tasks in (
        ("many_local_minima", ("A", "B", "C")),
        ("bowl_shaped", ("D", "E", "F")),
    ):
        for task_index, task in enumerate(tasks):
            for optimizer, gain in (
                ("best_logged", 0.0),
                ("coms", float(task_index)),
                ("bdi", float(2 - task_index)),
            ):
                rows.append(
                    {
                        "task": task.lower(),
                        "display_name": task,
                        "category": category,
                        "optimizer": optimizer,
                        "refnorm_max_score": 1.0 + gain,
                    }
                )
    frame = pd.DataFrame(rows)

    selection = select_top_synthetic_tasks(frame, top_n=2)
    assert len(selection) == 8
    assert set(
        selection[
            (selection["category"] == "many_local_minima")
            & (selection["optimizer"] == "coms")
        ]["display_name"]
    ) == {"B", "C"}
    assert set(
        selection[
            (selection["category"] == "many_local_minima")
            & (selection["optimizer"] == "bdi")
        ]["display_name"]
    ) == {"A", "B"}
    assert "normalized_best_utility" in selection.columns
    assert "gain_vs_best_logged" not in selection.columns

    save_synthetic_bo_top_task_plots(frame, tmp_path, top_n=2)
    output_dir = tmp_path / "synthetic_categories_top2"
    assert (output_dir / "top2_selection.csv").is_file()
    overview_path = output_dir / "00_all_categories_top2_overview.png"
    assert overview_path.is_file()
    overview = mpimg.imread(overview_path)
    assert overview.shape[0] > overview.shape[1]
