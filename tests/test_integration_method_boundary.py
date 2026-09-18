"""The main methods must obey the newly integrated freeze/execution boundary."""

import json
import inspect
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_design_bench.evaluation import llmdm_protocol as protocol
from llm_design_bench.evaluation.formal_methods import FORMAL_METHOD_IDS
from llm_design_bench.evaluation.seed_runner import (
    MethodSpec,
    SeedBenchmarkConfig,
    resolved_method_config,
    run_method_seed_benchmark,
)
from llm_design_bench.optimizers import make_method
from llm_design_bench.optimizers.base import OfflineBBOMethod
from llm_design_bench.optimizers.diffusion_methods import DDOM
from llm_design_bench.problem import (
    MethodResult,
    OfflineProblem,
    ProblemMetadata,
    RunContext,
)
from llm_design_bench.spaces import SimplexSpace
from llm_design_bench.types import CandidateBatch


@pytest.fixture
def frozen_bundle(monkeypatch):
    monkeypatch.setattr(
        protocol,
        "package_source_identity",
        lambda: {
            "sha256": "integration-source",
            "git_commit": None,
            "git_dirty": False,
        },
    )
    return SimpleNamespace(manifest_id="integration-visible-data")


@pytest.fixture
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _visible_problem():
    generator = torch.Generator().manual_seed(7)
    space = SimplexSpace(3)
    designs = space.sample(10, generator=generator)
    context = torch.tensor([[20.0, 1000.0]] * 5 + [[1000.0, 19500.0]] * 5)
    return OfflineProblem(
        train_designs=designs,
        train_context=context,
        train_utility=torch.linspace(-3.0, -1.0, 10),
        target_context=torch.tensor([1000.0, 19500.0]),
        design_space=space,
        metadata=ProblemMetadata(task_name="integration-visible-only"),
    )


def test_all_non_spade_methods_freeze_complete_defaults_and_validate(
    frozen_bundle, tmp_path
):
    plan = protocol.freeze_method_plan(
        frozen_bundle,
        [{"method_id": name, "kwargs": {}} for name in FORMAL_METHOD_IDS],
        experiment_id="integration-defaults",
    )
    assert len(plan["methods"]) == 27
    for entry in plan["methods"]:
        method = make_method(entry["method_id"])
        # Inspect constructor attributes independently of the resolver. Derived
        # public state (e.g. PGS's transition_config) is not a constructor input.
        constructor_names = {
            name
            for cls in type(method).__mro__
            if cls is not object
            for name, parameter in inspect.signature(cls.__init__).parameters.items()
            if name != "self"
            and parameter.kind
            in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        }
        expected = {
            name: value
            for name, value in vars(method).items()
            if name in constructor_names
        }
        assert entry["kwargs"] == expected
        assert method.configuration() == expected
        assert resolved_method_config(method, {}) == expected
        assert entry["requested_kwargs"] == {}
        assert (
            make_method(entry["method_id"], **entry["kwargs"]).configuration()
            == expected
        )
    protocol.validate_method_plan(plan, frozen_bundle)
    saved = protocol.save_method_plan(plan, tmp_path / "plan.json")
    assert protocol.load_method_plan(saved, frozen_bundle) == plan


@pytest.mark.parametrize(
    "method_id,requested",
    [
        (
            "cbas",
            {
                "epochs": 3,
                "steps": 4,
                "validation_fraction": 0.1,
                "latent_dim": 7,
                "adaptation_epochs": 2,
            },
        ),
        (
            "rgd",
            {
                "epochs": 3,
                "steps": 4,
                "validation_fraction": 0.1,
                "diffusion_steps": 8,
                "guidance": 3.0,
                "target_margin": 0.25,
                "refinement_rounds": 3,
                "likelihood_steps": 5,
            },
        ),
    ],
)
def test_inherited_and_middle_layer_overrides_survive_frozen_plan(
    frozen_bundle, method_id, requested
):
    plan = protocol.freeze_method_plan(
        frozen_bundle,
        [{"method_id": method_id, "kwargs": requested}],
        experiment_id="integration-overrides",
    )
    entry = plan["methods"][0]
    assert entry["requested_kwargs"] == requested
    assert requested.items() <= entry["kwargs"].items()
    assert entry["kwargs"]["batch_size"] == 64
    assert entry["kwargs"]["hidden_size"] == 128
    assert entry["kwargs"]["learning_rate"] == 0.001
    assert make_method(method_id, **entry["kwargs"]).configuration() == entry["kwargs"]
    protocol.validate_method_plan(plan, frozen_bundle)


def test_real_prepared_method_uses_only_visible_data_and_oracle_after_proposal(
    monkeypatch, tmp_path, single_torch_thread
):
    visible = _visible_problem()
    original = {
        name: getattr(visible, name).clone()
        for name in (
            "train_designs",
            "train_context",
            "train_utility",
            "target_context",
        )
    }
    kwargs = {
        "epochs": 1,
        "steps": 2,
        "batch_size": 4,
        "hidden_size": 8,
        "diffusion_steps": 4,
        "validation_fraction": 0.2,
    }
    expected_config = make_method("ddom", **kwargs).configuration()
    events = []
    actual_fit = DDOM.fit_prepared
    actual_propose = DDOM.propose_prepared

    def inspect_fit(self, prepared, *, context, generator):
        assert events == []
        assert {field.name for field in fields(prepared.problem)} == {
            "train_designs",
            "train_context",
            "train_utility",
            "target_context",
            "design_space",
            "metadata",
        }
        assert not hasattr(prepared.problem, "predict")
        assert not hasattr(prepared.problem, "reference_utility")
        assert prepared.problem.metadata.extra == {}
        assert prepared.problem.sample_count == 10
        indices = []
        for subset in (prepared.split.train, prepared.split.validation):
            ids = subset.row_indices
            indices.extend(ids.tolist())
            assert torch.equal(subset.designs, original["train_designs"][ids])
            assert torch.equal(subset.context, original["train_context"][ids])
            assert torch.equal(subset.utility, original["train_utility"][ids])
        assert sorted(indices) == list(range(10))
        assert len(prepared.split.train) == 8
        assert len(prepared.split.validation) == 2
        train = prepared.split.train
        expected_context = train.context.clone()
        expected_context[:, 0] = expected_context[:, 0].log1p()
        assert torch.equal(
            prepared.transforms.utility_standardizer.mean, train.utility.mean()
        )
        assert torch.equal(
            prepared.transforms.context_standardizer.mean, expected_context.mean(0)
        )
        target = prepared.features_at_target(original["train_designs"][:4])
        transformed_target = prepared.transforms.transform_context(
            original["target_context"].unsqueeze(0)
        )
        assert torch.equal(target[:, -2:], transformed_target.expand(4, -1))
        events.append("fit")
        return actual_fit(self, prepared, context=context, generator=generator)

    def inspect_propose(self, prepared, *, context, generator):
        assert events == ["fit"]
        candidates = actual_propose(
            self, prepared, context=context, generator=generator
        )
        events.append("proposed")
        return candidates

    monkeypatch.setattr(DDOM, "fit_prepared", inspect_fit)
    monkeypatch.setattr(DDOM, "propose_prepared", inspect_propose)

    class Evaluator:
        def __init__(self):
            self.evaluated = []

        def at_target_fidelity(self, designs):
            assert events == ["fit", "proposed"]
            return CandidateBatch.at_fidelity(designs, 1000.0, 19500.0)

        def predict(self, batch):
            assert events == ["fit", "proposed"]
            assert np.all(batch.model_scales == 1000.0)
            assert np.all(batch.training_steps == 19500.0)
            self.evaluated.append(batch.mixtures.copy())
            events.append("oracle")
            return -np.square(batch.mixtures - np.array([0.5, 0.3, 0.2])).sum(1)

    evaluator = Evaluator()
    summaries = []
    for trial, hidden_reference in enumerate(([-10.0, 10.0], [-1000.0, 1000.0])):
        events.clear()
        result = run_method_seed_benchmark(
            evaluator,
            visible,
            [MethodSpec("ddom", kwargs)],
            reference_utility=np.array(hidden_reference),
            config=SeedBenchmarkConfig(
                seeds=(0,),
                split_seed=7,
                candidate_budget=4,
                results_dir=tmp_path / f"trial-{trial}",
                fail_fast=True,
            ),
        )
        assert events == ["fit", "proposed", "oracle"]
        row = result.per_seed.iloc[0]
        assert row["status"] == "success"
        assert json.loads(row["method_config_json"]) == expected_config
        assert json.loads(row["requested_method_config_json"]) == kwargs
        summary = json.loads(row["training_summary_json"])
        assert summary["train_samples"] == 8
        assert summary["validation_samples"] == 2
        assert summary["split_seed"] == 7
        assert summary["resolved_method_config"] == expected_config
        summaries.append(summary)
    np.testing.assert_array_equal(evaluator.evaluated[0], evaluator.evaluated[1])
    assert summaries[0] == summaries[1]
    for name, value in original.items():
        assert torch.equal(getattr(visible, name), value)


def test_public_learned_state_does_not_enter_configuration_and_capture_is_pre_run():
    class LearnsPublicState(OfflineBBOMethod):
        def __init__(self, *, epochs=3, settings=None):
            self.epochs = epochs
            self.settings = {"widths": [4]} if settings is None else settings

        def optimize(self, problem, *, context, generator):
            self.final_loss = 0.5
            self.model = torch.nn.Linear(3, 1)
            self.training_history = [0.7, 0.5]
            self.epochs += 1
            self.settings["widths"].append(8)
            return MethodResult(
                candidates=problem.train_designs[: context.candidate_budget],
                training_summary={"resolved_method_config": {"wrong": "method-state"}},
            )

    method = LearnsPublicState()
    expected = {"epochs": 3, "settings": {"widths": [4]}}
    assert resolved_method_config(method, {}) == expected
    result = method.run(
        _visible_problem(), RunContext(method_seed=0, candidate_budget=4)
    )
    assert result.training_summary["resolved_method_config"] == expected
    assert method.configuration() == {"epochs": 4, "settings": {"widths": [4, 8]}}
    assert not {"final_loss", "model", "training_history"} & set(method.configuration())
