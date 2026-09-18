from pathlib import Path

import pandas as pd
import typer

from llm_design_bench.evaluation.unified_report import (
    UNIFIED_REPORT_FILENAMES,
    load_legacy_publication_results,
    write_unified_report,
)


app = typer.Typer(add_completion=False)


def _protect_report_inputs(inputs: tuple[Path, ...], results_dir: Path) -> None:
    """Reject output/input aliases before the report writer changes any files.

    Resolve symbolic links and parent-directory aliases; compare existing files
    by identity as well so hard links are covered. This is a CLI input boundary,
    not a restriction on the writer's intentional incremental report updates.
    """
    destinations = tuple(
        (results_dir / name).resolve() for name in UNIFIED_REPORT_FILENAMES
    )
    for input_path in inputs:
        source = input_path.resolve()
        for destination in destinations:
            aliases_input = destination == source
            if not aliases_input:
                try:
                    aliases_input = destination.samefile(source)
                except FileNotFoundError:
                    # A not-yet-created output cannot alias an existing input.
                    aliases_input = False
            if aliases_input:
                raise typer.BadParameter(
                    f"results-dir must not overwrite input file {input_path}: "
                    f"report output {destination} refers to the same file"
                )


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
        help="Directory for summaries, Markdown, LaTeX, and copied raw rows.",
    ),
) -> None:
    _protect_report_inputs((input_csv,), results_dir)
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
        help="Directory outside publication-dir for converted unified artifacts.",
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
    _protect_report_inputs(
        (publication_dir / "raw_runs.csv", publication_dir / "run_metadata.json"),
        results_dir,
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
    typer.echo(f"converted {len(result.per_seed)} legacy rows into {results_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
