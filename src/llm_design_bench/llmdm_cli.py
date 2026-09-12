"""Dedicated entry point for the consensus LLM-DM experiment protocol."""

import json
from pathlib import Path
from typing import Annotated

import torch
import typer

from llm_design_bench.evaluation.data_manifest import (
    load_data_manifest,
    prepare_data_manifest,
    save_data_manifest,
)
from llm_design_bench.evaluation.llmdm_protocol import (
    freeze_method_plan,
    load_method_plan,
    run_frozen_experiment,
    save_method_plan,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def prepare(
    data_recipes_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    oracle_checkpoint: Annotated[
        list[Path],
        typer.Option(
            "--oracle-checkpoint",
            exists=True,
            dir_okay=False,
            help="Trusted checkpoint file. Repeat for each checkpoint loaded by the oracle.",
        ),
    ],
    output: Annotated[
        Path, typer.Option(help="New shared data directory; must not exist.")
    ],
) -> None:
    """Freeze the shared rows and oracle provenance, without running the oracle.

    Only use a trusted data-recipes checkout: its logged dataset is a pickle.
    """
    bundle = prepare_data_manifest(
        data_recipes_root, [path.resolve() for path in oracle_checkpoint]
    )
    save_data_manifest(bundle, output)
    typer.echo(f"Prepared shared data: {output}\nData manifest: {bundle.manifest_id}")
    typer.echo(
        f"Logged rows: {len(bundle.reference_utility)}; "
        f"main visible rows: {len(bundle.utility)}; "
        f"fixed-1B visible rows: {int((bundle.context[:, 0] == 1000).sum())}"
    )


@app.command()
def freeze(
    data_recipes_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    data_bundle: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    methods_file: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    experiment_id: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option(help="New frozen plan JSON; must not exist.")],
) -> None:
    """Expand each selected method's settings and freeze them, without training."""
    bundle = load_data_manifest(data_bundle, data_recipes_root=data_recipes_root)
    methods = json.loads(methods_file.read_text(encoding="utf-8"))
    if not isinstance(methods, list):
        raise typer.BadParameter("methods-file must contain a list of method objects")
    plan = freeze_method_plan(bundle, methods, experiment_id=experiment_id)
    save_method_plan(plan, output)
    typer.echo(f"Frozen method plan: {output}\nPlan ID: {plan['plan_id']}")


@app.command()
def run(
    data_recipes_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    data_bundle: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    plan: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    phase: Annotated[str, typer.Option(help="pilot (seed 0) or formal (seeds 38-45).")],
    results_dir: Annotated[Path, typer.Option()],
    setting: Annotated[
        str, typer.Option(help="multi_scale, fixed_1b, or both.")
    ] = "multi_scale",
    run_id: Annotated[list[str] | None, typer.Option("--run-id")] = None,
    seed: Annotated[list[int] | None, typer.Option("--seed")] = None,
    device: Annotated[
        str, typer.Option(help="Method device, e.g. cpu or cuda.")
    ] = "cpu",
    oracle_device: Annotated[str, typer.Option(help="Evaluator device.")] = "cpu",
    pilot_results: Annotated[
        Path | None, typer.Option(help="Required for formal runs.")
    ] = None,
    resume: Annotated[
        bool, typer.Option(help="Reuse an identical successful run.")
    ] = False,
    infrastructure_retry_reason: Annotated[
        str | None,
        typer.Option(
            help="Explicit infrastructure interruption reason; same seed/config only.",
        ),
    ] = None,
    torch_threads: Annotated[int, typer.Option(min=1)] = 1,
) -> None:
    """Run frozen methods. This is the only command that trains or calls oracle."""
    bundle = load_data_manifest(data_bundle, data_recipes_root=data_recipes_root)
    frozen = load_method_plan(plan, bundle)
    torch.set_num_threads(torch_threads)
    result = run_frozen_experiment(
        bundle,
        frozen,
        data_recipes_root=data_recipes_root,
        results_dir=results_dir,
        phase=phase,
        setting=setting,
        run_ids=run_id,
        seeds=seed,
        device=device,
        oracle_device=oracle_device,
        pilot_results=pilot_results,
        resume=resume,
        infrastructure_retry_reason=infrastructure_retry_reason,
    )
    typer.echo(result.summary.to_string(index=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
