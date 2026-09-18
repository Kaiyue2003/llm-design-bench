"""Tiny Python API demonstration, not a formal LLM-DM experiment."""

from llm_design_bench import OfflineProblem, RunContext, make
from llm_design_bench.optimizers import make_method


def main() -> None:
    task = make("synthetic-ackley", logged_samples=32, seed=0)
    problem = OfflineProblem.from_task(task)
    methods = (
        ("random_search", {}),
        (
            "offline_mlp",
            {"hidden_size": 16, "epochs": 2, "particle_steps": 2},
        ),
    )
    for method_id, kwargs in methods:
        result = make_method(method_id, **kwargs).run(
            problem, RunContext(method_seed=0, candidate_budget=4, dataset_seed=0)
        )
        # Methods never receive the evaluator. Query only after they return.
        batch = task.at_target_fidelity(result.candidates.detach().cpu().numpy())
        best_utility = float(task.predict(batch).max())
        print(f"{method_id}: best utility={best_utility:.6f}")
    print("Small API demonstration only; use the frozen CLI for formal LLM-DM.")


if __name__ == "__main__":
    main()
