from llm_design_bench.data import (
    OfflineDataSplit,
    OfflineTensorDataset,
    split_offline_dataset,
)
from llm_design_bench.problem import (
    MethodResult,
    OfflineProblem,
    ProblemMetadata,
    RunContext,
)
from llm_design_bench.registry import make
from llm_design_bench.spaces import BoxSpace, DesignSpace, SimplexSpace
from llm_design_bench.types import CandidateBatch
from llm_design_bench.transforms import (
    FittedProblemTransforms,
    PreparedOfflineProblem,
    ProblemPreparationConfig,
    TensorStandardizer,
    prepare_offline_problem,
)

__version__ = "0.1.0"

__all__ = [
    "BoxSpace",
    "CandidateBatch",
    "DesignSpace",
    "MethodResult",
    "OfflineDataSplit",
    "OfflineProblem",
    "OfflineTensorDataset",
    "FittedProblemTransforms",
    "PreparedOfflineProblem",
    "ProblemPreparationConfig",
    "ProblemMetadata",
    "RunContext",
    "SimplexSpace",
    "TensorStandardizer",
    "__version__",
    "make",
    "prepare_offline_problem",
    "split_offline_dataset",
]


def main() -> None:
    from llm_design_bench.llmdm_cli import main as formal_main

    formal_main()
