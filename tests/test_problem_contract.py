import numpy as np
import pytest
import torch

from llm_design_bench.problem import OfflineProblem, ProblemMetadata
from llm_design_bench.spaces import BoxSpace, SimplexSpace
from llm_design_bench.types import CandidateBatch


class LoggedSimplexTask:
    mixture_dim = 3
    target_model_scale = 1000.0
    target_training_steps = 19_500.0

    def __init__(self) -> None:
        self.predict_calls = 0
        self.logged_x = CandidateBatch(
            mixtures=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.1, 0.2, 0.7],
                ]
            ),
            model_scales=np.array([20.0, 150.0, 1000.0]),
            training_steps=np.array([1_000.0, 5_000.0, 19_500.0]),
        )
        self.logged_y = np.array([-2.0, -1.5, -1.0])

    def predict(self, batch: CandidateBatch) -> np.ndarray:
        self.predict_calls += 1
        return batch.mixtures[:, 0]


def test_offline_problem_from_task_does_not_query_oracle() -> None:
    task = LoggedSimplexTask()

    problem = OfflineProblem.from_task(task)

    assert task.predict_calls == 0
    assert problem.sample_count == 3
    assert problem.design_dim == 3
    assert problem.context_dim == 2
    assert problem.train_features.shape == (3, 5)
    assert torch.allclose(problem.target_context, torch.tensor([1000.0, 19_500.0]))
    assert isinstance(problem.design_space, SimplexSpace)


def test_float32_problem_retains_original_box_boundary_precision() -> None:
    task = LoggedSimplexTask()
    task.design_bounds = np.array([[-32.768, 32.768]] * 3)
    problem = OfflineProblem.from_task(task, dtype=torch.float32)
    assert problem.train_designs.dtype == torch.float32
    assert np.array_equal(problem.design_space.bounds.numpy(), task.design_bounds)


def test_offline_problem_target_features_and_standardized_utility() -> None:
    problem = OfflineProblem.from_task(LoggedSimplexTask())
    designs = torch.tensor([[0.4, 0.4, 0.2]], dtype=torch.float32)

    features = problem.features_at_target(designs)
    standardized, mean, std = problem.standardized_utility()

    assert features.shape == (1, 5)
    assert torch.allclose(features[0, -2:], problem.target_context)
    assert torch.isclose(standardized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(mean, problem.train_utility.mean())
    assert std > 0


def test_simplex_space_round_trip_sampling_and_validation() -> None:
    space = SimplexSpace(3)
    generator = torch.Generator().manual_seed(7)

    samples = space.sample(8, generator=generator)
    parameters = space.to_unconstrained(samples)
    reconstructed = space.from_unconstrained(parameters)

    space.validate(samples)
    assert samples.shape == (8, 3)
    assert torch.allclose(samples.sum(dim=1), torch.ones(8))
    assert torch.allclose(samples, reconstructed, atol=1e-6)
    with pytest.raises(ValueError, match="sum to one"):
        space.validate(torch.tensor([[0.2, 0.2, 0.2]]))


def test_box_space_sampling_and_transform() -> None:
    space = BoxSpace(torch.tensor([[-2.0, 2.0], [1.0, 3.0]]))
    generator = torch.Generator().manual_seed(9)

    samples = space.sample(6, generator=generator, dtype=torch.float64)
    parameters = space.to_unconstrained(samples)
    reconstructed = space.from_unconstrained(parameters)

    space.validate(samples)
    assert samples.dtype == torch.float64
    assert torch.allclose(samples, reconstructed)


def test_problem_rejects_non_maximization_metadata() -> None:
    with pytest.raises(ValueError, match="maximization"):
        ProblemMetadata(task_name="invalid", utility_direction="minimize")
