from pathlib import Path

import pandas as pd
import typer

from llm_design_bench.evaluation.unified_report import (
    load_legacy_publication_results,
    write_unified_report,
)


app = typer.Typer(add_completion=False)


@app.command("from-unified")
def from_unified(
    input_csv: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Unified method_seed_results.csv to summarize.",
    ),
    results_dir: Path = typer.Option(
        Path("results/report"),
        help="New directory for summaries, Markdown, LaTeX, and copied raw rows.",
    ),
) -> None:
    destination = results_dir.resolve() / "method_seed_results.csv"
    if destination == input_csv.resolve():
        raise typer.BadParameter("results-dir must not overwrite the input CSV")
    result = write_unified_report(pd.read_csv(input_csv), results_dir)
    typer.echo(
        f"wrote {len(result.per_seed)} per-seed rows and "
        f"{len(result.summary)} task/method summaries to {results_dir}"
    )


@app.command("from-legacy")
def from_legacy(
    publication_dir: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        readable=True,
        help="Frozen publication-v1 directory containing raw_runs.csv.",
    ),
    results_dir: Path = typer.Option(
        Path("results/publication_v1_unified"),
        help="New directory for converted unified artifacts.",
    ),
    experiment_id: str = typer.Option(
        "publication_v1",
        help="Experiment identifier assigned to converted rows.",
    ),
) -> None:
    source = publication_dir.resolve()
    destination = results_dir.resolve()
    if destination == source or source in destination.parents:
        raise typer.BadParameter(
            "results-dir must be outside publication-dir; publication v1 is immutable"
        )
    converted = load_legacy_publication_results(
        publication_dir,
        experiment_id=experiment_id,
    )
    result = write_unified_report(
        converted,
        results_dir,
        metadata={"legacy_publication_dir": str(publication_dir.resolve())},
    )
    typer.echo(
        f"converted {len(result.per_seed)} legacy rows into {results_dir}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
