"""Minimal current-method API demo, not a frozen benchmark experiment.

Install the package, then run ``python examples/synthetic_quickstart.py``.
Synthetic logs are generated locally; no external dataset is downloaded and no
formal experiment artifacts are written.
"""


def main() -> None:
    import numpy as np
    import torch

    from llm_design_bench import make
    from llm_design_bench.optimizers import make_method
    from llm_design_bench.problem import OfflineProblem, RunContext

    print("Demo only: not a frozen pilot or formal benchmark run.")
    task = make("synthetic-ackley", logged_samples=128, seed=38)
    problem = OfflineProblem.from_task(task, dtype=torch.float64)
    methods = (
        (make_method("coms", epochs=20, particle_steps=20), torch.float32),
        (make_method("bdi", steps=20), torch.float64),
    )

    for method, dtype in methods:
        context = RunContext(method_seed=38, candidate_budget=16, dtype=dtype)
        result = method.run(problem, context)
        # The method sees logged data only. Evaluate exactly its final candidates.
        batch = task.at_target_fidelity(result.candidates.detach().cpu().numpy())
        utility = np.asarray(task.predict(batch), dtype=np.float64)
        if (
            utility.shape != (context.candidate_budget,)
            or not np.isfinite(utility).all()
        ):
            raise ValueError("oracle must return one finite utility per candidate")
        best_utility = float(utility.max())
        print(
            f"{method.metadata.display_name}: best utility={best_utility:.6f}, "
            f"best objective={-best_utility:.6f}"
        )


if __name__ == "__main__":
    main()
