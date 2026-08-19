from pathlib import Path

import typer

from llm_design_bench.evaluation.synthetic_runner import (
    SyntheticBenchmarkConfig,
    run_synthetic_bo_benchmarks,
)
from llm_design_bench.tasks.synthetic_functions import DEFAULT_SYNTHETIC_FUNCTIONS

app = typer.Typer(add_completion=False)


@app.command()
def benchmark(
    function: list[str] | None = typer.Option(
        None,
        "--function",
        "-f",
        help="Synthetic function to run. Repeat to select multiple functions.",
    ),
    logged_samples: int = typer.Option(256, min=8, help="Offline logged samples per function."),
    recommendations: int = typer.Option(64, min=1, help="Final candidate batch size."),
    epochs: int = typer.Option(100, min=1, help="COM training epochs."),
    particle_steps: int = typer.Option(100, min=1, help="COM particle search steps."),
    bdi_steps: int = typer.Option(100, min=1, help="BDI distillation steps."),
    seed: int = typer.Option(38, help="Random seed."),
    results_dir: Path = typer.Option(Path("results"), help="Output directory."),
) -> None:
    functions = tuple(function) if function else DEFAULT_SYNTHETIC_FUNCTIONS
    frame = run_synthetic_bo_benchmarks(
        SyntheticBenchmarkConfig(
            functions=functions,
            logged_samples=logged_samples,
            recommendations=recommendations,
            seed=seed,
            epochs=epochs,
            particle_steps=particle_steps,
            bdi_steps=bdi_steps,
            results_dir=results_dir,
        )
    )
    typer.echo(frame.to_string(index=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
