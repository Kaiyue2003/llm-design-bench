import numpy as np
from scipy.stats import qmc

from llm_design_bench.optimizers.base import EvaluationTrace, top_candidates


class SobolSearch:
    def __init__(
        self,
        queries: int = 256,
        recommendations: int = 128,
        seed: int = 0,
    ) -> None:
        self.queries = queries
        self.recommendations = recommendations
        self.seed = seed

    def optimize(self, task) -> EvaluationTrace:
        unit_cube = qmc.Sobol(d=task.mixture_dim, scramble=True, seed=self.seed).random(
            self.queries
        )
        clipped = np.clip(unit_cube, np.finfo(float).eps, None)
        mixtures = clipped / clipped.sum(axis=1, keepdims=True)
        queried = task.at_target_fidelity(mixtures)
        utility = task.predict(queried)
        recommendations, recommendation_utility = top_candidates(
            queried,
            utility,
            self.recommendations,
        )
        return EvaluationTrace(
            name="sobol",
            recommendations=recommendations,
            recommendation_utility=recommendation_utility,
            queried=queried,
            query_utility=utility,
            query_cost=task.cost(queried),
        )
