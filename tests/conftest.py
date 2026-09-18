"""Opt-in test references for orchestration independent of method migration."""

import pytest

from llm_design_bench.optimizers import registry
from llm_design_bench.optimizers.base import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    OfflineBBOMethod,
)
from llm_design_bench.problem import MethodResult


class ArtifactReference(OfflineBBOMethod):
    metadata = MethodMetadata(
        method_id="fixture_random",
        display_name="Artifact Reference (test only)",
        family=MethodFamily.REFERENCE,
        implementation_kind=ImplementationKind.NATIVE_BASELINE,
    )
    capabilities = MethodCapabilities(supports_box=True)

    def optimize(self, problem, *, context, generator):
        return MethodResult(
            candidates=problem.design_space.sample(
                context.candidate_budget,
                generator=generator,
                device=context.device,
                dtype=context.dtype,
            )
        )


@pytest.fixture
def artifact_reference_registry(monkeypatch):
    """Do not register a test-only method globally or alter the built-in catalog."""
    name = ArtifactReference.metadata.method_id
    monkeypatch.setitem(registry._METHODS, name, ArtifactReference)
    monkeypatch.setitem(registry._METADATA, name, ArtifactReference.metadata)
    monkeypatch.setitem(registry._CAPABILITIES, name, ArtifactReference.capabilities)
