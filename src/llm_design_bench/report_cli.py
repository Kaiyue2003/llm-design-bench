from pathlib import Path

import pandas as pd
import typer

from llm_design_bench.evaluation.unified_report import (
    UNIFIED_REPORT_FILENAMES,
    write_unified_report,
)


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def report() -> None:
    """Regenerate reports from current method_seed_results.csv files."""


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
