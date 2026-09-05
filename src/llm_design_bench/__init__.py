from llm_design_bench.problem import (
    MethodResult,
    OfflineProblem,
    ProblemMetadata,
    RunContext,
)
from llm_design_bench.registry import make
from llm_design_bench.spaces import BoxSpace, DesignSpace, SimplexSpace
from llm_design_bench.types import CandidateBatch

__version__ = "0.1.0"

__all__ = [
    "BoxSpace",
    "CandidateBatch",
    "DesignSpace",
    "MethodResult",
    "OfflineProblem",
    "ProblemMetadata",
    "RunContext",
    "SimplexSpace",
    "__version__",
    "make",
]


def main() -> None:
    from llm_design_bench.cli import app

    app()
