import numpy as np

from llm_design_bench.optimizers.base import EvaluationTrace
from llm_design_bench.types import CandidateBatch


class BestLoggedOptimizer:
    def __init__(self, recommendations: int = 128) -> None:
        self.recommendations = recommendations

    def optimize(self, task) -> EvaluationTrace:
        order = np.argsort(task.logged_y)[::-1]
        selected: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for index in order:
            mixture = task.logged_x.mixtures[index]
            key = tuple(np.round(mixture, 12))
            if key in seen:
                continue
            seen.add(key)
            selected.append(mixture)
            if len(selected) == self.recommendations:
                break

        recommendations = task.at_target_fidelity(np.vstack(selected))
        utility = task.predict(recommendations)
        empty = CandidateBatch(
            mixtures=np.empty((0, task.mixture_dim)),
            model_scales=np.empty(0),
            training_steps=np.empty(0),
        )
        return EvaluationTrace(
            name="best_logged",
            recommendations=recommendations,
            recommendation_utility=utility,
            queried=empty,
            query_utility=np.empty(0),
            query_cost=np.empty(0),
        )
