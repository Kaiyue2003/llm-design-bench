from pathlib import Path
import json

import typer

from llm_design_bench.evaluation.publication_runner import (
    DEFAULT_PUBLICATION_SEEDS,
    ALL_METHOD_ORDER,
    METHOD_ORDER,
    PUBLICATION_SYNTHETIC_FUNCTIONS,
    PublicationBenchmarkConfig,
    run_publication_benchmarks,
)


app = typer.Typer(add_completion=False)


@app.command()
def benchmark(
    method: list[str] | None = typer.Option(None, "--method", help="Registered method ID; repeat to select several."),
    all_methods: bool = typer.Option(False, "--all-methods", help="Run all fourteen integrated PyTorch methods."),
    method_config: Path | None = typer.Option(None, "--method-config", help="JSON object mapping method IDs to constructor settings."),
    method_steps: int = typer.Option(50, min=1, help="Search/sampling steps for the ten additional methods."),
    device: str = typer.Option("cpu", help="PyTorch method device: cpu or cuda."),
    seed: list[int] | None = typer.Option(
        None,
        "--seed",
        help="Independent trial seed. Repeat to select multiple seeds.",
    ),
    function: list[str] | None = typer.Option(
        None,
        "--function",
        "-f",
        help="Synthetic function to run. Repeat to override the publication subset.",
    ),
    data_recipes_root: Path | None = typer.Option(
        None,
        help="Path to the data-recipes clone.",
    ),
    include_data_mixture: bool = typer.Option(
        True,
        "--data-mixture/--no-data-mixture",
        help="Include the Data Recipes LLM data-mixture task.",
    ),
    metric_index: int = typer.Option(4, min=0, max=10, help="Data Recipes metric index."),
    logged_samples: int = typer.Option(256, min=8, help="Logged samples per synthetic task."),
    recommendations: int = typer.Option(128, min=1, help="Final candidate batch size K."),
    epochs: int = typer.Option(100, min=1, help="Training epochs for neural methods."),
    particle_steps: int = typer.Option(100, min=1, help="COM particle search steps."),
    bdi_steps: int = typer.Option(100, min=1, help="BDI distillation steps."),
    train_min_percentile: float = typer.Option(
        0.0,
        min=0.0,
        max=100.0,
        help="Lowest logged utility percentile visible for data-mixture optimization.",
    ),
    train_max_percentile: float = typer.Option(
        40.0,
        min=0.0,
        max=100.0,
        help="Highest logged utility percentile visible for data-mixture optimization.",
    ),
    torch_threads: int = typer.Option(1, min=1, help="PyTorch CPU thread count."),
    deterministic: bool = typer.Option(
        True,
        "--deterministic/--no-deterministic",
        help="Require deterministic PyTorch algorithms.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume compatible rows from raw_runs.csv.",
    ),
    results_dir: Path = typer.Option(
        Path("results/publication"),
        help="Output directory for raw data, summaries, Markdown, LaTeX, and metadata.",
    ),
) -> None:
    if all_methods and method:
        raise typer.BadParameter("use either --all-methods or --method")
    configs = json.loads(method_config.read_text(encoding="utf-8")) if method_config else {}
    if not isinstance(configs, dict) or any(not isinstance(value, dict) for value in configs.values()):
        raise typer.BadParameter("method config must map method IDs to objects")
    frame = run_publication_benchmarks(
        PublicationBenchmarkConfig(
            methods=ALL_METHOD_ORDER if all_methods else tuple(method) if method else METHOD_ORDER,
            method_configs=configs,
            method_steps=method_steps,
            device=device,
            seeds=tuple(seed) if seed else DEFAULT_PUBLICATION_SEEDS,
            functions=tuple(function) if function else PUBLICATION_SYNTHETIC_FUNCTIONS,
            data_recipes_root=data_recipes_root,
            include_data_mixture=include_data_mixture,
            metric_index=metric_index,
            logged_samples=logged_samples,
            recommendations=recommendations,
            epochs=epochs,
            particle_steps=particle_steps,
            bdi_steps=bdi_steps,
            train_min_percentile=train_min_percentile,
            train_max_percentile=train_max_percentile,
            deterministic=deterministic,
            torch_threads=torch_threads,
            resume=resume,
            results_dir=results_dir,
        )
    )
    typer.echo(frame.to_string(index=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
