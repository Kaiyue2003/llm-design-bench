import torch

from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext


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
