import torch

from llm_design_bench.optimizers.base import (
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
