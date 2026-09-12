"""Constructor-only checks for our agreed first-batch LLM-DM budgets."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_design_bench.evaluation import llmdm_protocol as protocol
from llm_design_bench.optimizers.registry import make_method, method_names

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "llmdm_methods.forward_v1.json"
)
METHODS = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
BY_ID = {entry["method_id"]: entry["kwargs"] for entry in METHODS}


def test_forward_batch_contains_exactly_our_methods():
    expected = {
        "best_logged",
        "random_search",
        "sobol",
        "offline_mlp",
        "standard_ga",
        "coms",
        "bdi",
        "ga_on_gp",
        "bo_qei",
        "cma_es",
        "reinforce",
        "mc_dropout",
        "roma",
        "ict",
        "tri_mentoring",
        "ltr",
        "match_opt",
        "pgs",
        "spade",
    }
    assert set(BY_ID) == expected
    assert len(METHODS) == len(BY_ID) == 19
    assert expected.issubset(method_names())
    for entry in METHODS:
        assert set(entry) == {"method_id", "kwargs"}
        # These values belong to the common run protocol, not method constructors.
        assert not {
            "seed",
            "device",
            "dtype",
            "candidate_budget",
            "recommendations",
        } & set(entry["kwargs"])


@pytest.mark.parametrize("entry", METHODS, ids=lambda entry: entry["method_id"])
def test_first_batch_kwargs_are_valid_and_retained(entry):
    method = make_method(entry["method_id"], **entry["kwargs"])
    inspect.signature(type(method)).bind(**entry["kwargs"])
    for name, expected in entry["kwargs"].items():
        assert getattr(method, name) == expected


@pytest.mark.parametrize(
    ("method_id", "expected"),
    [
        ("offline_mlp", {"epochs": 200, "particle_steps": 200}),
        ("standard_ga", {"surrogate_epochs": 200, "solver_steps": 200}),
        ("coms", {"epochs": 200, "adversarial_steps": 20, "particle_steps": 200}),
        ("bdi", {"steps": 200}),
        ("ga_on_gp", {"gp_training_steps": 200, "solver_steps": 200}),
        (
            "bo_qei",
            {"gp_training_steps": 200, "acquisition_steps": 200, "mc_samples": 128},
        ),
        (
            "cma_es",
            {
                "ensemble_size": 5,
                "surrogate_epochs": 200,
                "population_size": 16,
                "generations": 100,
            },
        ),
        (
            "reinforce",
            {
                "ensemble_size": 5,
                "surrogate_epochs": 200,
                "iterations": 200,
                "reinforce_batch_size": 512,
            },
        ),
        (
            "mc_dropout",
            {
                "epochs": 200,
                "dropout_probability": 0.1,
                "mc_samples": 32,
                "particle_steps": 200,
            },
        ),
        (
            "roma",
            {
                "surrogate_epochs": 100,
                "weight_perturbation_steps": 5,
                "adaptation_steps": 10,
                "solver_steps": 100,
            },
        ),
        (
            "ict",
            {
                "surrogate_epochs": 200,
                "surrogate_learning_rate": 0.001,
                "adaptation_steps": 100,
                "solver_steps": 100,
            },
        ),
        (
            "tri_mentoring",
            {
                "surrogate_epochs": 200,
                "surrogate_learning_rate": 0.001,
                "solver_steps": 100,
                "neighbor_samples": 10,
            },
        ),
        (
            "ltr",
            {
                "surrogate_epochs": 100,
                "list_length": 32,
                "lists_per_epoch": 256,
                "batch_size": 32,
                "solver_steps": 200,
            },
        ),
        (
            "match_opt",
            {
                "embedding_dim": 8,
                "surrogate_epochs": 200,
                "bucket_count": 32,
                "quadrature_nodes": 5,
                "solver_steps": 200,
            },
        ),
        ("pgs", {"surrogate_epochs": 200, "rl_steps": 10000, "solver_steps": 50}),
        (
            "spade",
            {
                "diff_hidden": 128,
                "diff_t_dim": 32,
                "diff_epochs": 100,
                "diff_batch": 64,
                "diff_steps": 100,
                "calib_mc_samples": 4,
                "calib_mc_steps": 10,
                "acq_mc_samples": 64,
                "acq_mc_steps": 50,
                "ea_pop": 128,
                "ea_elite": 64,
                "ea_gens": 100,
            },
        ),
    ],
)
def test_agreed_budgets_are_explicit(method_id, expected):
    assert expected.items() <= BY_ID[method_id].items()


def test_widths_batches_and_controls_are_explicit():
    for method_id in ("best_logged", "random_search", "sobol"):
        assert BY_ID[method_id] == {}
    for method_id, kwargs in BY_ID.items():
        if "hidden_size" in kwargs:
            assert kwargs["hidden_size"] == (64 if method_id == "roma" else 128)
        if "batch_size" in kwargs:
            assert kwargs["batch_size"] == (32 if method_id == "ltr" else 64)


def test_entire_batch_freezes_expands_defaults_and_round_trips_without_running(
    monkeypatch,
):
    monkeypatch.setattr(
        protocol,
        "package_source_identity",
        lambda: {
            "sha256": "budget-test-source",
            "git_commit": "test",
            "git_dirty": False,
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("freezing a budget must not train, search or evaluate")

    for entry in METHODS:
        method_class = type(make_method(entry["method_id"], **entry["kwargs"]))
        for operation in ("run", "fit", "propose", "optimize"):
            if hasattr(method_class, operation):
                monkeypatch.setattr(method_class, operation, forbidden)

    bundle = SimpleNamespace(manifest_id="budget-test-manifest")
    plan = protocol.freeze_method_plan(bundle, METHODS, experiment_id="forward_v1")
    protocol.validate_method_plan(plan, bundle)
    shared = plan["shared_settings"]
    assert shared["candidate_budget"] == 128
    assert shared["pilot_seeds"] == [0]
    assert shared["formal_seeds"] == list(range(38, 46))
    assert shared["mixed_precision"] is False
    for entry in plan["methods"]:
        assert entry["requested_kwargs"] == BY_ID[entry["method_id"]]
        expected_dtype = (
            "float64"
            if entry["method_id"] in {"bo_qei", "ga_on_gp", "bdi"}
            else "float32"
        )
        assert entry["dtype"] == expected_dtype
        constructor = inspect.signature(
            type(make_method(entry["method_id"], **entry["kwargs"]))
        )
        expected_names = {
            name
            for name, parameter in constructor.parameters.items()
            if parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        }
        assert set(entry["kwargs"]) == expected_names

    resolved = {entry["method_id"]: entry["kwargs"] for entry in plan["methods"]}
    assert resolved["cma_es"]["num_layers"] == 1
    assert resolved["reinforce"]["num_layers"] == 1
    assert resolved["standard_ga"]["num_layers"] == 2
    assert resolved["ict"]["surrogate_learning_rate"] == 0.001
    assert resolved["tri_mentoring"]["surrogate_learning_rate"] == 0.001
    assert resolved["pgs"]["backup_entropy"] is False
    assert resolved["spade"]["support_transform"] is False
    assert resolved["spade"]["acq_beta"] == 0.1
