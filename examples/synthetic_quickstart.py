from llm_design_bench import make
from llm_design_bench.optimizers import (
    BackwardDistillationOptimizer,
    ConservativeObjectiveModelOptimizer,
)


def main() -> None:
    task = make("synthetic-ackley", logged_samples=128, seed=38)
    optimizers = (
        ConservativeObjectiveModelOptimizer(
            recommendations=16,
            seed=38,
            epochs=20,
            particle_steps=20,
        ),
        BackwardDistillationOptimizer(
            recommendations=16,
            seed=38,
            steps=20,
        ),
    )

    for optimizer in optimizers:
        trace = optimizer.optimize(task)
        best_utility = float(trace.recommendation_utility.max())
        print(
            f"{trace.name}: best utility={best_utility:.6f}, "
            f"best objective={-best_utility:.6f}"
        )


if __name__ == "__main__":
    main()
