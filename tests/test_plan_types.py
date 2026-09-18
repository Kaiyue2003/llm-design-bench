"""Plan typing must not rewrite frozen identities or bypass runtime validation."""

import copy
import json
from types import MappingProxyType, SimpleNamespace

import pytest

from llm_design_bench.evaluation import llmdm_protocol as protocol
from llm_design_bench.evaluation.plan_types import read_plan_shape


@pytest.fixture
def frozen_plan(monkeypatch):
    monkeypatch.setattr(
        protocol,
        "package_source_identity",
        lambda: {"sha256": "fixed-source", "git_commit": "abc123", "git_dirty": False},
    )
    bundle = SimpleNamespace(manifest_id="test-manifest")
    plan = protocol.freeze_method_plan(
        bundle,
        [
            {"method_id": "best_logged", "kwargs": {}},
            {"method_id": "bdi", "kwargs": {"steps": 2}},
        ],
        experiment_id="contract",
    )
    return bundle, plan


def _rehash(plan):
    payload = {key: value for key, value in plan.items() if key != "plan_id"}
    plan["plan_id"] = protocol._digest(payload)


def test_typed_plan_preserves_integration_fixture_identity(frozen_plan, tmp_path):
    bundle, plan = frozen_plan
    # Canonical BDI now uses our forward constructor, restoring its invented
    # fixture identity. No real archived plan is rewritten during integration.
    assert (
        plan["plan_id"]
        == "b7a4701874d9b18af37a7399f8d033b12dff604f1f7481abd004f5694ed09925"
    )
    before = protocol._canonical(plan)
    assert read_plan_shape(plan) is plan
    assert protocol._canonical(plan) == before
    path = protocol.save_method_plan(plan, tmp_path / "plan.json")
    assert protocol._canonical(protocol.load_method_plan(path, bundle)) == before


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("package_source", "git_dirty"), 0),
        (("package_source", "git_commit"), []),
        (("methods", 0, "requested_kwargs"), []),
        (("methods", 0, "run_id"), 123),
        (("shared_settings", "formal_seeds", 0), 38.0),
    ],
)
def test_rehashed_invalid_field_types_are_rejected(frozen_plan, path, value, tmp_path):
    bundle, original = frozen_plan
    plan = copy.deepcopy(original)
    parent = plan
    for component in path[:-1]:
        parent = parent[component]
    parent[path[-1]] = value
    _rehash(plan)
    file = tmp_path / "malformed.json"
    file.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(TypeError):
        protocol.load_method_plan(file, bundle)


def test_unknown_metadata_is_preserved_and_covered_by_hash(frozen_plan):
    bundle, plan = frozen_plan
    plan["extra_audit_note"] = {"preserve": [1, "two", None]}
    _rehash(plan)
    before = copy.deepcopy(plan)
    protocol.validate_method_plan(plan, bundle)
    assert read_plan_shape(plan) is plan
    assert plan == before
    plan["extra_audit_note"]["preserve"].append(3)
    with pytest.raises(ValueError, match="hash mismatch"):
        protocol.validate_method_plan(plan, bundle)


def test_shape_check_alone_is_not_semantic_or_source_verification(frozen_plan):
    bundle, plan = frozen_plan
    plan["package_source"]["sha256"] = "another-source"
    _rehash(plan)
    assert read_plan_shape(plan) is plan
    with pytest.raises(ValueError, match="package source changed"):
        protocol.validate_method_plan(plan, bundle)


@pytest.mark.parametrize("nested", [False, True])
def test_shape_reader_does_not_claim_read_only_mappings_are_mutable(
    frozen_plan, nested
):
    _, plan = frozen_plan
    if nested:
        plan["methods"][0]["kwargs"] = MappingProxyType({})
    else:
        plan = MappingProxyType(plan)
    with pytest.raises(TypeError, match="JSON object"):
        read_plan_shape(plan)


def test_semantic_validator_preserves_read_only_outer_mapping_support(frozen_plan):
    bundle, plan = frozen_plan
    protocol.validate_method_plan(MappingProxyType(plan), bundle)
