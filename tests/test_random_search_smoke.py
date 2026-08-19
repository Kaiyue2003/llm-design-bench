import numpy as np

from llm_design_bench.optimizers.random_search import DirichletRandomSearch
from llm_design_bench.types import CandidateBatch


class ToyTask:
    mixture_dim = 3

    def at_target_fidelity(self, mixtures: np.ndarray) -> CandidateBatch:
        return CandidateBatch.at_fidelity(mixtures, 1, 1)

    def predict(self, batch: CandidateBatch) -> np.ndarray:
        target = np.array([0.7, 0.2, 0.1])
        return -np.square(batch.mixtures - target).sum(axis=1)

    def cost(self, batch: CandidateBatch) -> np.ndarray:
        return np.ones(len(batch))


def test_random_search_returns_simplex_candidates() -> None:
    result = DirichletRandomSearch(queries=32, recommendations=8, seed=7).optimize(ToyTask())
    assert len(result.queried) == 32
    assert len(result.recommendations) == 8
    assert np.allclose(result.queried.mixtures.sum(axis=1), 1.0)
    assert (result.queried.mixtures >= 0).all()
    assert result.cumulative_cost == 32.0
