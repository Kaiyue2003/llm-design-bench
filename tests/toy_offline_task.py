"""Small evaluator fixture shared by offline API regression tests."""

import numpy as np

from llm_design_bench.types import CandidateBatch


class ToyOfflineTask:
    mixture_dim = 3
    target_model_scale = 1000
    target_training_steps = 19_500

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
            training_steps=np.array(
                [100, 500, 1000, 2500, 5000, 10_000, 15_000, 19_500]
            ),
        )
        self.logged_y = self._oracle(mixtures)

    def at_target_fidelity(self, mixtures: np.ndarray) -> CandidateBatch:
        return CandidateBatch.at_fidelity(
            mixtures, self.target_model_scale, self.target_training_steps
        )

    def predict(self, batch: CandidateBatch) -> np.ndarray:
        self.predict_calls += 1
        return self._oracle(batch.mixtures)

    @staticmethod
    def _oracle(mixtures: np.ndarray) -> np.ndarray:
        target = np.array([0.70, 0.20, 0.10])
        return -np.square(mixtures - target).sum(axis=1)
