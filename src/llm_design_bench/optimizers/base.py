from dataclasses import dataclass

import numpy as np

from llm_design_bench.types import CandidateBatch


@dataclass(frozen=True)
class EvaluationTrace:
    name: str
    recommendations: CandidateBatch
    recommendation_utility: np.ndarray
    queried: CandidateBatch
    query_utility: np.ndarray
    query_cost: np.ndarray

    @property
    def cumulative_cost(self) -> float:
        return float(self.query_cost.sum())


def top_candidates(
    batch: CandidateBatch,
    utility: np.ndarray,
    count: int,
) -> tuple[CandidateBatch, np.ndarray]:
    indices = np.argsort(utility)[-min(count, len(batch)) :]
    return (
        CandidateBatch(
            mixtures=batch.mixtures[indices],
            model_scales=batch.model_scales[indices],
            training_steps=batch.training_steps[indices],
        ),
        utility[indices],
    )
