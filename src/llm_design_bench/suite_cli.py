from pathlib import Path

import typer

from llm_design_bench.evaluation.seed_runner import (
    DEFAULT_METHOD_SEEDS,
    SeedBenchmarkConfig,
)
from llm_design_bench.evaluation.task_specs import (
    make_data_recipes_task_spec,
    make_synthetic_task_spec,
)
from llm_design_bench.evaluation.unified_report import run_benchmark_suite


app = typer.Typer(add_completion=False)


@app.command()
def benchmark(
    method: list[str] | None = typer.Option(
        None,
        "--method",
        "-m",
        help="Registered method ID. Repeat to select multiple methods.",
    ),
    seed: list[int] | None = typer.Option(
        None,
        "--seed",
        help="Paired method/dataset seed. Repeat to select multiple seeds.",
    ),
    function: list[str] | None = typer.Option(
        None,
        "--function",
        "-f",
        help="Synthetic function. Repeat to select multiple functions.",
    ),
    include_data_mixture: bool = typer.Option(
        False,
        "--data-mixture/--no-data-mixture",
        help="Include the data-recipes LLM-DM task.",
    ),
    data_recipes_root: Path | None = typer.Option(
        None,
        help="Path to the data-recipes clone.",
    ),
    fixed_1b: bool = typer.Option(
        False,
        "--fixed-1b/--multi-scale",
        help="Expose only 1B logged observations; target remains 1B/19,500 steps.",
    ),
    metric_index: int = typer.Option(4, min=0, max=10),
    train_min_percentile: float = typer.Option(0.0, min=0.0, max=100.0),
    train_max_percentile: float = typer.Option(40.0, min=0.0, max=100.0),
    logged_samples: int = typer.Option(256, min=1),
    candidate_budget: int = typer.Option(128, "--candidate-budget", "-k", min=1),
    experiment_id: str = typer.Option("unified_benchmark"),
    device: str = typer.Option("cpu"),
    results_dir: Path = typer.Option(Path("results/unified_benchmark")),
) -> None:
    if train_min_percentile > train_max_percentile:
        raise typer.BadParameter(
            "train-min-percentile must not exceed train-max-percentile"
        )
    if fixed_1b and not include_data_mixture:
        raise typer.BadParameter("--fixed-1b requires --data-mixture")
    tasks = [
        make_synthetic_task_spec(name, logged_samples=logged_samples)
        for name in (function or [])
    ]
    if include_data_mixture:
        tasks.append(
            make_data_recipes_task_spec(
                data_recipes_root=data_recipes_root,
                metric_index=metric_index,
                logged_model_scale=1000.0 if fixed_1b else None,
                train_min_percentile=train_min_percentile,
                train_max_percentile=train_max_percentile,
            )
        )
    if not tasks:
        raise typer.BadParameter(
            "select at least one --function or enable --data-mixture"
        )
    selected_methods = tuple(
        method
        or [
            "best_logged",
            "random_search",
            "sobol",
            "offline_mlp",
            "coms",
            "bdi",
        ]
    )
    run_metadata = {"synthetic_logged_samples": logged_samples}
    if include_data_mixture:
        run_metadata.update(
            {
                "data_mixture_setting": (
                    "fixed_1b" if fixed_1b else "multi_scale"
                ),
                "data_mixture_metric_index": metric_index,
                "train_min_percentile": train_min_percentile,
                "train_max_percentile": train_max_percentile,
            }
        )
    result = run_benchmark_suite(
        tasks,
        selected_methods,
        config=SeedBenchmarkConfig(
            experiment_id=experiment_id,
            seeds=tuple(seed or DEFAULT_METHOD_SEEDS),
            candidate_budget=candidate_budget,
            device=device,
            results_dir=results_dir,
        ),
        metadata=run_metadata,
    )
    typer.echo(result.summary.to_string(index=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
