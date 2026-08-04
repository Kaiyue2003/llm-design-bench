from __future__ import annotations

from pathlib import Path

from llm_design_bench import make
from llm_design_bench.optimizers import DirichletRandomSearch


def main() -> None:
    task = make("data-recipes", data_recipes_root=Path("../data-recipes"))
    optimizer = DirichletRandomSearch(queries=128, recommendations=32, seed=0)
    batch = optimizer.recommend(task)
    utility = task.evaluate(batch)

    print(f"Generated {len(utility)} candidate mixtures.")
    print(f"Best utility: {utility.max():.6f}")


if __name__ == "__main__":
    main()
