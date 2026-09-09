import torch

from llm_design_bench.data import OfflineTensorDataset, split_offline_dataset
from llm_design_bench.problem import (
    OfflineProblem,
    ProblemMetadata,
    RunContext,
)
from llm_design_bench.spaces import BoxSpace, SimplexSpace
from llm_design_bench.transforms import (
    ProblemPreparationConfig,
    prepare_offline_problem,
)


def _problem() -> OfflineProblem:
    return OfflineProblem(
        train_designs=torch.tensor(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.2, 0.7],
                [0.4, 0.4, 0.2],
                [0.3, 0.2, 0.5],
            ],
            dtype=torch.float64,
        ),
        train_context=torch.tensor(
            [
                [20.0, 1000.0],
                [60.0, 2000.0],
                [150.0, 5000.0],
                [400.0, 10000.0],
                [1000.0, 19500.0],
            ],
            dtype=torch.float64,
        ),
        train_utility=torch.tensor(
            [-5.0, -2.0, 0.0, 4.0, 20.0],
            dtype=torch.float64,
        ),
        target_context=torch.tensor([1000.0, 19500.0], dtype=torch.float64),
        design_space=SimplexSpace(3),
        metadata=ProblemMetadata(task_name="unified-data-test"),
    )


def test_split_is_deterministic_and_retains_source_rows() -> None:
    dataset = OfflineTensorDataset.from_problem(_problem())

    first = split_offline_dataset(dataset, validation_fraction=0.4, seed=17)
    second = split_offline_dataset(dataset, validation_fraction=0.4, seed=17)

    assert torch.equal(first.train.row_indices, second.train.row_indices)
    assert torch.equal(first.validation.row_indices, second.validation.row_indices)
    assert len(first.train) == 3
    assert len(first.validation) == 2
    combined = torch.cat([first.train.row_indices, first.validation.row_indices])
    assert set(combined.tolist()) == set(range(5))


def test_preparation_fits_statistics_on_training_rows_only() -> None:
    prepared = prepare_offline_problem(
        _problem(),
        RunContext(method_seed=38, split_seed=17, candidate_budget=4),
    )

    expected_mean = prepared.split.train.utility.mean()
    assert torch.equal(prepared.transforms.utility_standardizer.mean, expected_mean)
    assert torch.allclose(
        prepared.train_utility.mean(),
        torch.tensor(0.0, dtype=torch.float64),
    )
    assert prepared.train_features.shape[1] == 4
    assert prepared.validation_features.shape[1] == 4


def test_design_model_coordinates_round_trip() -> None:
    simplex = SimplexSpace(3)
    simplex_designs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    assert torch.allclose(
        simplex.decode_from_model(simplex.encode_for_model(simplex_designs)),
        simplex_designs,
        atol=1e-6,
    )

    box = BoxSpace(torch.tensor([[-5.0, 5.0], [2.0, 6.0]]))
    box_designs = torch.tensor([[-2.0, 3.0], [5.0, 6.0]])
    encoded = box.encode_for_model(box_designs)
    assert torch.all((0.0 <= encoded) & (encoded <= 1.0))
    assert torch.allclose(box.decode_from_model(encoded), box_designs)


def test_preparation_accepts_a_problem_without_context() -> None:
    problem = OfflineProblem(
        train_designs=torch.tensor([[0.6, 0.4], [0.2, 0.8]]),
        train_context=torch.empty((2, 0)),
        train_utility=torch.tensor([0.0, 1.0]),
        target_context=torch.empty(0),
        design_space=SimplexSpace(2),
        metadata=ProblemMetadata(task_name="no-context"),
    )

    prepared = prepare_offline_problem(
        problem,
        RunContext(method_seed=1, candidate_budget=1),
        ProblemPreparationConfig(validation_fraction=0.0),
    )

    assert prepared.train_features.shape == (2, 1)
