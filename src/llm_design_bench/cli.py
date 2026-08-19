from pathlib import Path

import typer

from llm_design_bench.evaluation.runner import BenchmarkConfig, run_baselines
from llm_design_bench.registry import make

app = typer.Typer(add_completion=False)


@app.command()
def benchmark(
    data_recipes_root: Path | None = typer.Option(None, help="Path to the data-recipes clone."),
    logged_model_scale: float | None = typer.Option(
        None,
        help="Optional model scale filter for logged data, e.g. 1000 for the 1B-only ablation.",
    ),
    queries: int = typer.Option(256, min=1, help="Queries per search baseline."),
    reference_queries: int = typer.Option(2048, min=1, help="Queries for regret reference."),
    recommendations: int = typer.Option(128, min=1, help="Final candidate batch size."),
    seed: int = typer.Option(38, help="Random seed."),
    results_dir: Path = typer.Option(Path("results"), help="Output directory."),
) -> None:
    task = make(
        "data-recipes",
        data_recipes_root=data_recipes_root,
        logged_model_scale=logged_model_scale,
    )
    frame = run_baselines(
        task,
        BenchmarkConfig(
            queries=queries,
            reference_queries=reference_queries,
            recommendations=recommendations,
            seed=seed,
            results_dir=results_dir,
        ),
    )
    typer.echo(frame.to_string(index=False))
