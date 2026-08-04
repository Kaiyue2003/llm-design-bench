import numpy as np

from llm_design_bench.optimizers.base import EvaluationTrace, top_candidates


class DirichletRandomSearch:
    def __init__(
        self,
        queries: int = 256,
        recommendations: int = 128,
        seed: int = 0,
        alpha: float = 1.0,
    ) -> None:
        self.queries = queries
        self.recommendations = recommendations
        self.seed = seed
        self.alpha = alpha

    def optimize(self, task) -> EvaluationTrace:
        rng = np.random.default_rng(self.seed)
        mixtures = rng.dirichlet(np.full(task.mixture_dim, self.alpha), size=self.queries)
        queried = task.at_target_fidelity(mixtures)
        utility = task.predict(queried)
        recommendations, recommendation_utility = top_candidates(
            queried,
            utility,
            self.recommendations,
        )
        return EvaluationTrace(
            name="dirichlet_random",
            recommendations=recommendations,
            recommendation_utility=recommendation_utility,
            queried=queried,
            query_utility=utility,
            query_cost=task.cost(queried),
        )
