"""Execution inputs are integers; CSV representation compatibility is separate."""

import re
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import torch

from llm_design_bench._integer_parameters import require_integer
from llm_design_bench.evaluation.seed_contracts import seed_group_contract
from llm_design_bench.evaluation.seed_types import SeedBenchmarkConfig
from llm_design_bench.problem import RunContext


EXECUTION_FIELDS = [
    ("config", "seeds"),
    ("config", "required_seeds"),
    ("context", "method_seed"),
    *[
        (api, field)
        for api in ("config", "context")
        for field in ("candidate_budget", "dataset_seed", "split_seed")
    ],
]


def _construct(api, field, value):
    if api == "config":
        kwargs = {"seeds": (38,), "required_seeds": (38,)}
        kwargs[field] = (value,) if field in {"seeds", "required_seeds"} else value
        return SeedBenchmarkConfig(**kwargs)
    return RunContext(**{"method_seed": 38, "candidate_budget": 128, field: value})


@pytest.mark.parametrize(("api", "field"), EXECUTION_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        38.0,
        38.9,
        np.float64(38.0),
        np.float32(38.9),
        "38",
        float("nan"),
        float("inf"),
        float("-inf"),
        Decimal("38"),
        Fraction(38, 1),
    ],
)
def test_execution_fields_reject_non_integer_types(api, field, value):
    with pytest.raises(TypeError, match=re.escape(field)):
        _construct(api, field, value)


@pytest.mark.parametrize(("api", "field"), EXECUTION_FIELDS)
@pytest.mark.parametrize("kind", [int, np.int32, np.int64, np.uint32, np.uint64])
def test_python_and_numpy_integers_are_normalized_without_changing_values(
    api, field, kind
):
    result = _construct(api, field, kind(38))
    actual = getattr(result, field)
    if isinstance(actual, tuple):
        assert actual == (38,)
        assert all(type(value) is int for value in actual)
    else:
        assert actual == 38
        assert type(actual) is int


@pytest.mark.parametrize(("api", "field"), EXECUTION_FIELDS)
def test_negative_integer_values_remain_invalid(api, field):
    with pytest.raises(ValueError, match=re.escape(field)):
        _construct(api, field, np.int64(-1))


@pytest.mark.parametrize("api", ["config", "context"])
def test_candidate_budget_still_requires_a_positive_integer(api):
    with pytest.raises(ValueError, match="candidate_budget must be positive"):
        _construct(api, "candidate_budget", 0)


@pytest.mark.parametrize("api", ["config", "context"])
@pytest.mark.parametrize("field", ["dataset_seed", "split_seed"])
def test_optional_metadata_seeds_preserve_none_and_allow_zero(api, field):
    assert getattr(_construct(api, field, None), field) is None
    assert getattr(_construct(api, field, np.int64(0)), field) == 0


@pytest.mark.parametrize(
    ("api", "field"),
    [
        ("config", "seeds"),
        ("config", "required_seeds"),
        ("context", "method_seed"),
        ("config", "candidate_budget"),
        ("context", "candidate_budget"),
    ],
)
def test_required_integer_fields_do_not_accept_none(api, field):
    with pytest.raises(TypeError, match=re.escape(field)):
        _construct(api, field, None)


def test_required_seed_default_still_follows_the_validated_selected_seeds():
    config = SeedBenchmarkConfig(seeds=(np.int64(38), np.uint32(39)))
    assert config.seeds == config.required_seeds == (38, 39)
    assert all(type(seed) is int for seed in config.required_seeds)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seeds": ()}, "seeds must not be empty"),
        ({"seeds": (38, np.int64(38))}, "seeds must be unique"),
        ({"required_seeds": ()}, "required_seeds must be non-empty"),
        (
            {"required_seeds": (38, np.int64(38))},
            "required_seeds must be non-empty, unique",
        ),
        ({"seeds": (39,), "required_seeds": (38,)}, "seeds must be a subset"),
        ({"phase": "formal", "seeds": (38,)}, "formal required_seeds must be exactly"),
        ({"phase": "pilot", "seeds": (38,)}, "pilot required_seeds must be"),
    ],
)
def test_existing_seed_set_and_phase_rules_are_preserved(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SeedBenchmarkConfig(**kwargs)


def test_agreed_formal_and_pilot_configs_remain_identical_for_numpy_integers():
    formal = SeedBenchmarkConfig(
        phase="formal", seeds=tuple(np.arange(38, 46)), candidate_budget=np.int64(128)
    )
    assert formal == SeedBenchmarkConfig(phase="formal")
    pilot = SeedBenchmarkConfig(
        phase="pilot", seeds=(np.int64(0),), candidate_budget=np.uint32(128)
    )
    assert pilot == SeedBenchmarkConfig(phase="pilot", seeds=(0,))


def test_seed_normalization_does_not_change_torch_generator_outputs():
    context = RunContext(method_seed=np.int64(38), candidate_budget=np.int64(128))
    expected = torch.Generator().manual_seed(38)
    actual = context.make_generator()
    assert torch.equal(
        torch.rand(8, generator=actual), torch.rand(8, generator=expected)
    )


def test_type_validation_does_not_add_a_new_rng_or_budget_upper_limit():
    # This verifies only constructor policy, not whether huge values are usable
    # by a generator or affordable to allocate. Range policy is outside this fix.
    value = 2**80
    config = SeedBenchmarkConfig(seeds=(value,), candidate_budget=value)
    context = RunContext(method_seed=value, candidate_budget=value)
    assert config.seeds == (value,)
    assert context.method_seed == context.candidate_budget == value


def test_integer_like_objects_are_not_converted_before_validation():
    class IntLike:
        def __int__(self):
            pytest.fail("arbitrary int conversion must not precede type validation")

    with pytest.raises(TypeError, match="probe must be an integer"):
        require_integer(IntLike(), name="probe")


@pytest.mark.parametrize(("api", "field"), EXECUTION_FIELDS)
def test_invalid_parameter_is_rejected_during_construction_before_any_output(
    api, field, tmp_path
):
    with pytest.raises(TypeError, match=re.escape(field)):
        if api == "config":
            kwargs = {"results_dir": tmp_path / "unused", "save_artifacts": True}
            kwargs[field] = (38.9,) if field in {"seeds", "required_seeds"} else 38.9
            SeedBenchmarkConfig(**kwargs)
        else:
            _construct(api, field, 38.9)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("seed", [38.0, "38", "38.0"])
def test_csv_observed_seed_representation_compatibility_is_unchanged(seed):
    group = pd.DataFrame(
        [
            {
                "result_source": "unified_runner",
                "phase": "exploratory",
                "required_seeds_json": "[38]",
                "method_seed": seed,
                "status": "success",
            }
        ]
    )
    contract = seed_group_contract(group)
    assert contract.observed_seeds == {38}
    assert contract.rank_eligible
