import pytest
import torch

from llm_design_bench.optimizers.base import (
    FitThenProposeMethod,
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.optimizers.registry import (
    get_method_metadata,
    list_methods,
    make_method,
    method_names,
    register_method,
)
from llm_design_bench.problem import (
    MethodResult,
    OfflineProblem,
    ProblemMetadata,
    RunContext,
)
from llm_design_bench.spaces import SimplexSpace


def _problem() -> OfflineProblem:
    return OfflineProblem(
        train_designs=torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.2, 0.7],
            ]
        ),
        train_context=torch.tensor(
            [
                [20.0, 1_000.0],
                [150.0, 5_000.0],
                [1000.0, 19_500.0],
            ]
        ),
        train_utility=torch.tensor([-2.0, -1.5, -1.0]),
        target_context=torch.tensor([1000.0, 19_500.0]),
        design_space=SimplexSpace(3),
        metadata=ProblemMetadata(task_name="toy"),
    )


@register_method("contract_random_simplex")
class ContractRandomSimplex(FitThenProposeMethod):
    metadata = MethodMetadata(
        method_id="contract_random_simplex",
        display_name="Contract Random Simplex",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
    )
    capabilities = MethodCapabilities(stochastic=True)

    def __init__(self, marker: str = "default") -> None:
        self.marker = marker

    def fit(self, problem, *, context, generator):
        return {"train_samples": problem.sample_count, "marker": self.marker}

    def propose(self, problem, *, context, generator):
        return problem.design_space.sample(
            context.candidate_budget,
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )

    def diagnostics(self):
        return {"kind": "contract_test"}


class InvalidBatchMethod(OfflineBBOMethod):
    metadata = MethodMetadata(
        method_id="invalid_batch_contract",
        display_name="Invalid Batch Contract",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
    )

    def optimize(self, problem, *, context, generator):
        return MethodResult(candidates=problem.train_designs[:1])


class MutatingMethod(OfflineBBOMethod):
    metadata = MethodMetadata(
        method_id="mutating_contract",
        display_name="Mutating Contract",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
    )

    def optimize(self, problem, *, context, generator):
        problem.train_designs[0, 0] = 0.5
        return MethodResult(
            candidates=problem.design_space.sample(
                context.candidate_budget,
                generator=generator,
                device=context.device,
                dtype=context.dtype,
            )
        )


def test_registered_method_factory_and_metadata() -> None:
    method = make_method("contract_random_simplex", marker="created")

    assert isinstance(method, ContractRandomSimplex)
    assert method.marker == "created"
    assert "contract_random_simplex" in method_names()
    assert get_method_metadata("contract_random_simplex") == method.metadata
    assert method.metadata in list_methods()


def test_fit_then_propose_method_is_reproducible_for_same_seed() -> None:
    context = RunContext(method_seed=38, candidate_budget=8)

    first = make_method("contract_random_simplex").run(_problem(), context)
    second = make_method("contract_random_simplex").run(_problem(), context)

    assert torch.equal(first.candidates, second.candidates)
    assert first.training_summary["train_samples"] == 3
    assert first.diagnostics["kind"] == "contract_test"


def test_different_method_seeds_change_stochastic_candidates() -> None:
    method = make_method("contract_random_simplex")

    first = method.run(_problem(), RunContext(method_seed=38, candidate_budget=8))
    second = method.run(_problem(), RunContext(method_seed=39, candidate_budget=8))

    assert not torch.equal(first.candidates, second.candidates)


def test_method_base_validates_candidate_budget() -> None:
    with pytest.raises(ValueError, match="expected"):
        InvalidBatchMethod().run(
            _problem(),
            RunContext(method_seed=38, candidate_budget=4),
        )


def test_method_run_isolates_original_problem_tensors() -> None:
    problem = _problem()
    original = problem.train_designs.clone()

    MutatingMethod().run(
        problem,
        RunContext(method_seed=38, candidate_budget=4),
    )

    assert torch.equal(problem.train_designs, original)


def test_registry_rejects_duplicate_ids() -> None:
    with pytest.raises(KeyError, match="already registered"):
        register_method("contract_random_simplex")(ContractRandomSimplex)


def test_unknown_method_lists_available_ids() -> None:
    with pytest.raises(KeyError, match="contract_random_simplex"):
        make_method("does_not_exist")
