from pathlib import Path

import typer

from llm_design_bench.evaluation.offline_runner import (
    OfflineBenchmarkConfig,
    run_offline_benchmarks,
)
from llm_design_bench.registry import make

app = typer.Typer(add_completion=False)


@app.command()
def benchmark(
    data_recipes_root: Path | None = typer.Option(None, help="Path to the data-recipes clone."),
    reference_queries: int = typer.Option(2048, min=1, help="Queries for regret reference."),
    recommendations: int = typer.Option(128, min=1, help="Final candidate batch size."),
    epochs: int = typer.Option(100, min=1, help="MLP and COM training epochs."),
    particle_steps: int = typer.Option(100, min=1, help="MLP and COM search steps."),
    bdi_steps: int = typer.Option(100, min=1, help="BDI distillation steps."),
    train_min_percentile: float = typer.Option(
        0.0,
        min=0.0,
        max=100.0,
        help="Lowest logged utility percentile visible to offline optimizers.",
    ),
    train_max_percentile: float = typer.Option(
        40.0,
        min=0.0,
        max=100.0,
        help="Highest logged utility percentile visible to offline optimizers.",
    ),
    seed: int = typer.Option(38, help="Random seed."),
    results_dir: Path = typer.Option(Path("results"), help="Output directory."),
) -> None:
    task = make("data-recipes", data_recipes_root=data_recipes_root)
    frame = run_offline_benchmarks(
        task,
        OfflineBenchmarkConfig(
            reference_queries=reference_queries,
            recommendations=recommendations,
            seed=seed,
            epochs=epochs,
            particle_steps=particle_steps,
            bdi_steps=bdi_steps,
            train_min_percentile=train_min_percentile,
            train_max_percentile=train_max_percentile,
            results_dir=results_dir,
        ),
    )
    typer.echo(frame.to_string(index=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
