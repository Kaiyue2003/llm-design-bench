import numpy as np
import pytest

from llm_design_bench.optimizers.bdi import BackwardDistillationOptimizer
from llm_design_bench.optimizers.coms import ConservativeObjectiveModelOptimizer
from llm_design_bench.optimizers.mlp_surrogate import OfflineMLPOptimizer
from llm_design_bench.evaluation.offline_runner import make_logged_percentile_split
from llm_design_bench.types import CandidateBatch


class ToyOfflineTask:
    mixture_dim = 3
    target_model_scale = 1000
    target_training_steps = 19_500
    max_training_steps = 19_600

    def __init__(self) -> None:
        self.predict_calls = 0
        mixtures = np.array(
            [
                [0.70, 0.20, 0.10],
                [0.60, 0.25, 0.15],
                [0.50, 0.30, 0.20],
                [0.40, 0.35, 0.25],
                [0.30, 0.40, 0.30],
                [0.20, 0.45, 0.35],
                [0.10, 0.50, 0.40],
                [0.15, 0.35, 0.50],
            ]
        )
        self.logged_x = CandidateBatch(
            mixtures=mixtures,
            model_scales=np.array([20, 60, 150, 300, 500, 700, 1000, 1000]),
            training_steps=np.array([100, 500, 1000, 2500, 5000, 10_000, 15_000, 19_500]),
        )
        self.logged_y = self._oracle(mixtures)

    def at_target_fidelity(self, mixtures: np.ndarray) -> CandidateBatch:
        return CandidateBatch.at_fidelity(
            mixtures,
            self.target_model_scale,
            self.target_training_steps,
        )

    def predict(self, batch: CandidateBatch) -> np.ndarray:
        self.predict_calls += 1
        return self._oracle(batch.mixtures)

    @staticmethod
    def _oracle(mixtures: np.ndarray) -> np.ndarray:
        target = np.array([0.70, 0.20, 0.10])
        return -np.square(mixtures - target).sum(axis=1)


@pytest.mark.parametrize(
    "optimizer",
    [
        OfflineMLPOptimizer(
            recommendations=4,
            seed=7,
            hidden_size=16,
            epochs=2,
            batch_size=4,
            particle_steps=2,
        ),
        ConservativeObjectiveModelOptimizer(
            recommendations=4,
            seed=7,
            hidden_size=16,
            epochs=2,
            batch_size=4,
            adversarial_steps=2,
            particle_steps=2,
        ),
        BackwardDistillationOptimizer(
            recommendations=4,
            seed=7,
            steps=2,
        ),
    ],
)
def test_offline_optimizer_returns_simplex_without_search_queries(optimizer) -> None:
    task = ToyOfflineTask()
    trace = optimizer.optimize(task)
    assert trace.name in {"offline_mlp", "coms", "bdi"}
    assert len(trace.queried) == 0
    assert trace.cumulative_cost == 0.0
    assert task.predict_calls == 1
    assert len(trace.recommendations) == 4
    assert np.allclose(trace.recommendations.mixtures.sum(axis=1), 1.0)
    assert (trace.recommendations.mixtures >= 0).all()


def test_logged_percentile_split_exposes_only_low_outcomes() -> None:
    task = ToyOfflineTask()
    split = make_logged_percentile_split(task, min_percentile=0.0, max_percentile=50.0)
    assert 0 < split.train_size < len(task.logged_y)
    assert split.d_best_utility == split.task.logged_y.max()
    assert split.refnorm_d_best_score < 1.0
    assert np.all(split.task.logged_y <= np.percentile(task.logged_y, 50.0))
    assert split.reference_y.shape == task.logged_y.shape
