import numpy as np
import torch

from llm_design_bench.optimizers.base import (
    EvaluationTrace,
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_utils import (
    repeat_rows,
    select_top_unique_designs,
)
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext
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

        if not selected:
            raise ValueError("logged dataset does not contain any designs")
        while len(selected) < self.recommendations:
            selected.extend(selected[: self.recommendations - len(selected)])

        recommendations = task.at_target_fidelity(
            np.vstack(selected[: self.recommendations])
        )
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


@register_method()
class BestLoggedMethod(OfflineBBOMethod):
    """Return the highest-utility unique designs in the visible log."""

    metadata = MethodMetadata(
        method_id="best_logged",
        display_name="Best Logged",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
        description="Top visible logged designs ranked by maximization utility.",
    )
    capabilities = MethodCapabilities(
        supports_simplex=True,
        supports_box=True,
        supports_context=True,
        stochastic=False,
    )

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        del generator
        selected = select_top_unique_designs(
            problem.train_designs,
            problem.train_utility,
            context.candidate_budget,
        )
        candidates = repeat_rows(selected, context.candidate_budget)
        return MethodResult(
            candidates=candidates,
            training_summary={
                "train_samples": problem.sample_count,
                "unique_logged_designs_used": len(selected),
            },
            diagnostics={"selection": "descending_logged_utility"},
        )
