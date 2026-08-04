from llm_design_bench.registry import make
from llm_design_bench.types import CandidateBatch

__version__ = "0.1.0"

__all__ = ["CandidateBatch", "__version__", "make"]


def main() -> None:
    from llm_design_bench.cli import app

    app()
