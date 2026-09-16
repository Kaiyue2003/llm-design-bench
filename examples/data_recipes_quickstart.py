"""Minimal offline API demo, not a frozen pilot or formal benchmark run.

Install this checkout first (python -m pip install -e .). Running the example
needs a trusted data-recipes checkout with its logged data and oracle checkpoint;
importing it or requesting --help does not load either. See docs/QUICKSTART.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Demo only: propose random mixtures offline, then evaluate at 1B / "
            "19500 steps. Not a frozen pilot or formal benchmark run."
        )
    )
    parser.add_argument(
        "--data-recipes-root",
        type=Path,
        help="Trusted checkout; otherwise use DATA_RECIPES_ROOT or adapter discovery.",
    )
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=128,
        help="Number of final candidates to evaluate (default: 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random sampling seed (default: 0; this is not a pilot run).",
    )
    args = parser.parse_args(argv)
    if args.candidate_budget < 1:
        parser.error("--candidate-budget must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    # Keep module import and --help independent of the ML/data runtime.
    import numpy as np
    import torch

    from llm_design_bench.evaluation import make_data_recipes_task_spec
    from llm_design_bench.optimizers import make_method
    from llm_design_bench.problem import RunContext

    print("Demo only: not a frozen pilot or formal benchmark run.")
    # Reuse the public adapter's per-scale low-utility split. Formal experiments
    # must instead read the shared frozen manifest and method plan.
    spec = make_data_recipes_task_spec(
        data_recipes_root=args.data_recipes_root,
        metric_index=4,  # StackExchange cross entropy; utility is already -loss.
        train_min_percentile=0.0,
        train_max_percentile=40.0,
    )
    trial = spec.trial_factory(args.seed)
    context = RunContext(
        method_seed=args.seed,
        candidate_budget=args.candidate_budget,
        device="cpu",
        dtype=torch.float32,
    )
    method = make_method("random_search")
    result = method.run(trial.problem, context)

    # Only the evaluator sees the oracle. Evaluate exactly the final batch;
    # do not query a larger pool and select candidates using oracle scores.
    task = trial.evaluator_task
    batch = task.at_target_fidelity(result.candidates.detach().cpu().numpy())
    utility = np.asarray(task.predict(batch), dtype=np.float64)
    if utility.shape != (context.candidate_budget,) or not np.isfinite(utility).all():
        raise ValueError("oracle must return one finite utility per candidate")

    print(f"Generated {len(utility)} candidate mixtures.")
    print(f"Best utility: {utility.max():.6f}")
    print(f"Lowest StackExchange cross entropy loss: {-utility.max():.6f}")


if __name__ == "__main__":
    main()
