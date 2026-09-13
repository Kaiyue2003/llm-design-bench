import pytest
import torch

from llm_design_bench.optimizers import (
    get_method_metadata,
    make_method,
    method_names,
    planned_method_names,
    list_method_blueprints,
)
from llm_design_bench.problem import (
    OfflineProblem,
    ProblemMetadata,
    RunContext,
)
from llm_design_bench.spaces import BoxSpace


def _box_problem() -> OfflineProblem:
    designs = torch.tensor(
        [
            [-1.0, -1.0],
            [-0.7, 0.4],
            [-0.2, 0.8],
            [0.1, -0.4],
            [0.4, 0.2],
            [0.7, -0.6],
            [0.8, 0.7],
            [1.0, 1.0],
        ]
    )
    utility = -torch.square(designs - torch.tensor([0.5, -0.25])).sum(dim=1)
    context = torch.ones((len(designs), 2))
    return OfflineProblem(
        train_designs=designs,
        train_context=context,
        train_utility=utility,
        target_context=torch.ones(2),
        design_space=BoxSpace(torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])),
        metadata=ProblemMetadata(task_name="box-method-test"),
    )


@pytest.mark.parametrize(
    ("method_id", "kwargs"),
    [
        ("best_logged", {}),
        (
            "offline_mlp",
            {"epochs": 1, "particle_steps": 1, "hidden_size": 8},
        ),
        (
            "coms",
            {
                "epochs": 1,
                "particle_steps": 1,
                "adversarial_steps": 1,
                "hidden_size": 8,
            },
        ),
        ("bdi", {"steps": 1, "max_support_points": 8}),
    ],
)
def test_registered_pytorch_methods_run_on_box_space(method_id, kwargs) -> None:
    result = make_method(method_id, **kwargs).run(
        _box_problem(),
        RunContext(method_seed=38, split_seed=7, candidate_budget=4),
    )

    assert result.candidates.shape == (4, 2)
    assert torch.isfinite(result.candidates).all()
    _box_problem().design_space.validate(result.candidates)
    assert get_method_metadata(method_id).implementation_framework == "pytorch"


def test_additional_catalog_tracks_implemented_adaptations() -> None:
    assert {blueprint.method_id for blueprint in list_method_blueprints()} == {
        "bonet",
        "cbas",
        "ddom",
        "demo",
        "gabo",
        "gtg",
        "mins",
        "rgd",
        "root",
        "spade",
    }
    assert {"best_logged", "offline_mlp", "coms", "bdi"}.issubset(method_names())
    assert planned_method_names() == ()
    assert "spade" in method_names()
    assert all(blueprint.status.value == "implemented_adaptation" for blueprint in list_method_blueprints())


def test_tensorflow_origin_is_disclosed_for_com_port() -> None:
    metadata = get_method_metadata("coms")
    assert metadata.original_framework == "TensorFlow"
    assert metadata.implementation_framework == "pytorch"
    assert metadata.implementation_kind.value == "lightweight_adaptation"
