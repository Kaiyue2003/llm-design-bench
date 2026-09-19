from copy import deepcopy

import pytest
import torch

from llm_design_bench.optimizers.base import OfflineBBOMethod
from llm_design_bench.problem import (
    MethodResult,
    OfflineProblem,
    ProblemMetadata,
    RunContext,
)
from llm_design_bench.spaces import SimplexSpace


def _extra() -> dict:
    return {
        "utility_transform": "negative_loss",
        "domain_order": ["Wikipedia", "StackExchange", "GitHub"],
        "settings": {
            "normalization": {"enabled": True, "offset": None, "scale": 1.0},
            "layers": [{"widths": [16, 8]}],
        },
        "tuple_nested": ({"labels": ["original"]},),
    }


def _problem(*, empty_extra: bool = False) -> OfflineProblem:
    return OfflineProblem(
        train_designs=torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]]),
        train_context=torch.tensor([[150.0, 5_000.0], [1000.0, 19_500.0]]),
        train_utility=torch.tensor([-2.0, -1.5]),
        target_context=torch.tensor([1000.0, 19_500.0]),
        design_space=SimplexSpace(3),
        metadata=ProblemMetadata(
            task_name="metadata_isolation",
            objective_name="CrossEntropyLoss",
            source="test_fixture",
            extra={} if empty_extra else _extra(),
        ),
    )


def _mutate_metadata(problem: OfflineProblem) -> None:
    extra = problem.metadata.extra
    extra["utility_transform"] = "identity"
    extra["domain_order"].reverse()
    extra["settings"]["normalization"]["enabled"] = False
    extra["settings"]["layers"][0]["widths"].append(4)
    extra["tuple_nested"][0]["labels"].append("mutated")


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_copy_isolates_nested_metadata_and_preserves_conversion(dtype) -> None:
    problem = _problem()
    original_metadata = deepcopy(problem.metadata)

    copied = problem.to("cpu", dtype=dtype, copy=True)

    assert copied.metadata == original_metadata
    assert copied.metadata is not problem.metadata
    assert copied.metadata.extra is not problem.metadata.extra
    assert copied.design_space is not problem.design_space
    expected_dtype = dtype or torch.float32
    for name in (
        "train_designs",
        "train_context",
        "train_utility",
        "target_context",
    ):
        original_tensor = getattr(problem, name)
        copied_tensor = getattr(copied, name)
        assert copied_tensor.device == torch.device("cpu")
        assert copied_tensor.dtype == expected_dtype
        assert copied_tensor.data_ptr() != original_tensor.data_ptr()
        torch.testing.assert_close(copied_tensor, original_tensor.to(expected_dtype))

    _mutate_metadata(copied)

    assert problem.metadata == original_metadata
    assert copied.metadata.extra != original_metadata.extra


def test_independent_copies_and_original_cannot_mutate_each_other() -> None:
    problem = _problem()
    first = problem.to("cpu", copy=True)
    second = problem.to("cpu", copy=True)
    expected = deepcopy(problem.metadata)

    _mutate_metadata(first)
    assert second.metadata == expected
    assert problem.metadata == expected

    problem.metadata.extra["settings"]["layers"][0]["widths"].append(2)
    assert second.metadata == expected
    assert first.metadata.extra["settings"]["layers"][0]["widths"] == [16, 8, 4]


def test_copy_isolates_empty_extra_mapping() -> None:
    problem = _problem(empty_extra=True)
    copied = problem.to("cpu", copy=True)

    copied.metadata.extra["new"] = {"mutable": []}

    assert problem.metadata.extra == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_copy_false_retains_shared_metadata_semantics(dtype) -> None:
    problem = _problem()

    converted = problem.to("cpu", dtype=dtype, copy=False)

    assert converted.metadata is problem.metadata
    assert converted.design_space is problem.design_space
    assert converted.train_designs.dtype == dtype
    _mutate_metadata(converted)
    assert problem.metadata.extra["utility_transform"] == "identity"
    assert problem.metadata.extra["settings"]["layers"][0]["widths"] == [16, 8, 4]


class _MetadataMutatingMethod(OfflineBBOMethod):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.observed_metadata = []

    def optimize(self, problem, *, context, generator):
        self.observed_metadata.append(deepcopy(problem.metadata))
        _mutate_metadata(problem)
        if self.fail:
            raise RuntimeError("intentional method failure")
        return MethodResult(
            candidates=problem.train_designs[:1].repeat(context.candidate_budget, 1)
        )


@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_method_mutation_cannot_pollute_original_or_next_method(
    fail_first, dtype
) -> None:
    problem = _problem()
    expected = deepcopy(problem.metadata)
    first = _MetadataMutatingMethod(fail=fail_first)
    first_context = RunContext(method_seed=38, candidate_budget=2, dtype=dtype)

    if fail_first:
        with pytest.raises(RuntimeError, match="intentional method failure"):
            first.run(problem, first_context)
    else:
        first.run(problem, first_context)

    assert problem.metadata == expected
    second = _MetadataMutatingMethod()
    result = second.run(
        problem, RunContext(method_seed=39, candidate_budget=2, dtype=dtype)
    )

    assert first.observed_metadata == [expected]
    assert second.observed_metadata == [expected]
    assert problem.metadata == expected
    assert result.candidates.shape == (2, 3)
    assert result.candidates.dtype == dtype


def test_reused_method_receives_pristine_metadata_for_every_seed() -> None:
    problem = _problem()
    expected = deepcopy(problem.metadata)
    method = _MetadataMutatingMethod()

    for seed in (38, 39, 40):
        method.run(problem, RunContext(method_seed=seed, candidate_budget=2))

    assert method.observed_metadata == [expected, expected, expected]
    assert problem.metadata == expected
