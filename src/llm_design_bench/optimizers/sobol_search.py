import numpy as np
from scipy.stats import qmc
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
from llm_design_bench.spaces import BoxSpace, SimplexSpace


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


@register_method()
class SobolSearchMethod(OfflineBBOMethod):
    """Generate a scrambled Sobol candidate batch with Torch."""

    metadata = MethodMetadata(
        method_id="sobol",
        display_name="Sobol",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
        description="Scrambled Torch Sobol sequence mapped into the design space.",
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
        del generator
        engine = torch.quasirandom.SobolEngine(
            dimension=problem.design_dim,
            scramble=True,
            seed=context.method_seed % (2**31 - 1),
        )
        unit = engine.draw(
            context.candidate_budget,
            dtype=context.dtype,
        ).to(context.device)

        if isinstance(problem.design_space, SimplexSpace):
            positive = unit.clamp_min(torch.finfo(context.dtype).tiny)
            candidates = positive / positive.sum(dim=1, keepdim=True)
            mapping = "positive_normalization"
        elif isinstance(problem.design_space, BoxSpace):
            bounds = problem.design_space.bounds.to(
                device=context.device,
                dtype=context.dtype,
            )
            lower, upper = bounds[:, 0], bounds[:, 1]
            candidates = lower + (upper - lower) * unit
            mapping = "affine_box"
        else:
            raise TypeError(
                f"unsupported design space for Sobol: "
                f"{type(problem.design_space).__name__}"
            )

        return MethodResult(
            candidates=candidates,
            diagnostics={"sampling": "scrambled_sobol", "mapping": mapping},
        )
