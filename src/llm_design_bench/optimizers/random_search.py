import numpy as np
import torch

from llm_design_bench.optimizers.base import (
    EvaluationTrace,
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
    top_candidates,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


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


@register_method()
class RandomSearchMethod(OfflineBBOMethod):
    """Sample the final candidate batch without consulting the oracle."""

    metadata = MethodMetadata(
        method_id="random_search",
        display_name="Random Search",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
        description=(
            "Uniform native-space sampling: Dirichlet(1) on a simplex and "
            "uniform sampling in a box."
        ),
    )
    capabilities = MethodCapabilities(
        supports_simplex=True,
        supports_box=True,
        supports_context=True,
        stochastic=True,
    )

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        candidates = problem.design_space.sample(
            context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )
        return MethodResult(
            candidates=candidates,
            diagnostics={"sampling": "native_space_uniform"},
        )
