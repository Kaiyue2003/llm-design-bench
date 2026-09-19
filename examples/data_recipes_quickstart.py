"""Propose from frozen visible data; no oracle calls or formal result writes.

For formal experiments use ``llm-design-bench prepare/freeze/run`` instead.
This deliberately small API demonstration consumes an already prepared bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_design_bench.evaluation.data_manifest import (
    load_data_manifest,
    make_frozen_data_recipes_task_spec,
)
from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import RunContext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-recipes-root", type=Path, required=True)
    parser.add_argument("--data-bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_data_manifest(
        args.data_bundle, data_recipes_root=args.data_recipes_root
    )
    task_spec = make_frozen_data_recipes_task_spec(
        bundle, data_recipes_root=args.data_recipes_root
    )
    trial = task_spec.trial_factory(0)
    result = make_method("random_search").run(
        trial.problem, RunContext(method_seed=0, candidate_budget=4)
    )
    print("Four demonstration mixtures; not formal benchmark results:")
    print(result.candidates.detach().cpu().numpy())
    print("No oracle evaluation was performed.")


if __name__ == "__main__":
    main()
