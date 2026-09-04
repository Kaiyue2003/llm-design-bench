from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Mapping

import numpy as np
import torch

from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext
from llm_design_bench.types import CandidateBatch


class MethodFamily(str, Enum):
    STANDARD = "standard"
    FORWARD_SURROGATE = "forward_surrogate"
    INVERSE_GENERATIVE = "inverse_generative"
    REFERENCE = "reference"


class ImplementationKind(str, Enum):
    OFFICIAL_WRAPPER = "official_wrapper"
    FAITHFUL_PYTORCH_PORT = "faithful_pytorch_port"
    MULTI_FIDELITY_ADAPTATION = "multi_fidelity_adaptation"
    LIGHTWEIGHT_ADAPTATION = "lightweight_adaptation"
    NATIVE_BASELINE = "native_baseline"


@dataclass(frozen=True)
class MethodCapabilities:
    supports_simplex: bool = True
    supports_box: bool = False
    supports_discrete: bool = False
    supports_context: bool = True
    stochastic: bool = True
    requires_gpu: bool = False


@dataclass(frozen=True)
class MethodMetadata:
    method_id: str
    display_name: str
    family: MethodFamily
    implementation_kind: ImplementationKind
    source_url: str | None = None
    source_commit: str | None = None
    description: str = ""
    adaptations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.method_id:
            raise ValueError("method_id must not be empty")
        if not self.display_name:
            raise ValueError("display_name must not be empty")
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in self.method_id
        ):
            raise ValueError(
                "method_id may contain only lowercase letters, digits, and underscores"
            )


class OfflineBBOMethod(ABC):
    """Minimal common boundary for an offline BBO method.

    Subclasses may hold PyTorch modules but this class intentionally does not
    inherit from ``torch.nn.Module``: references and classical optimizers need
    the same public contract.
    """

    metadata: ClassVar[MethodMetadata]
    capabilities: ClassVar[MethodCapabilities] = MethodCapabilities()

    def run(self, problem: OfflineProblem, context: RunContext) -> MethodResult:
        # Isolate runs from in-place changes made by a method. Frozen
        # dataclasses do not otherwise prevent mutation of tensor contents.
        prepared = problem.to(context.device, context.dtype, copy=True)
        generator = context.make_generator()
        result = self.optimize(prepared, context=context, generator=generator)
        if not isinstance(result, MethodResult):
            raise TypeError("optimize must return MethodResult")
        self._validate_result(result, prepared, context)
        return result

    @abstractmethod
    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult: ...

    @classmethod
    def resolved_metadata(cls) -> MethodMetadata:
        try:
            metadata = cls.metadata
        except AttributeError as exc:
            raise TypeError(f"{cls.__name__} must declare method metadata") from exc
        if not isinstance(metadata, MethodMetadata):
            raise TypeError("metadata must be a MethodMetadata instance")
        return metadata

    @classmethod
    def resolved_capabilities(cls) -> MethodCapabilities:
        if not isinstance(cls.capabilities, MethodCapabilities):
            raise TypeError("capabilities must be a MethodCapabilities instance")
        return cls.capabilities

    @staticmethod
    def _validate_result(
        result: MethodResult,
        problem: OfflineProblem,
        context: RunContext,
    ) -> None:
        expected = (context.candidate_budget, problem.design_dim)
        if tuple(result.candidates.shape) != expected:
            raise ValueError(
                f"method returned candidates with shape {tuple(result.candidates.shape)}; "
                f"expected {expected}"
            )
        if result.candidates.device != problem.train_designs.device:
            raise ValueError("method candidates must be on the run-context device")
        if result.candidates.dtype != context.dtype:
            raise ValueError("method candidates must use the run-context dtype")
        problem.design_space.validate(result.candidates)


class FitThenProposeMethod(OfflineBBOMethod):
    """Template for the common train-a-model, then propose-candidates flow."""

    def optimize(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> MethodResult:
        summary = self.fit(
            problem,
            context=context,
            generator=generator,
        )
        candidates = self.propose(
            problem,
            context=context,
            generator=generator,
        )
        return MethodResult(
            candidates=candidates,
            training_summary=dict(summary or {}),
            diagnostics=dict(self.diagnostics()),
        )

    @abstractmethod
    def fit(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> Mapping[str, Any] | None: ...

    @abstractmethod
    def propose(
        self,
        problem: OfflineProblem,
        *,
        context: RunContext,
        generator: torch.Generator,
    ) -> torch.Tensor: ...

    def diagnostics(self) -> Mapping[str, Any]:
        return {}


@dataclass(frozen=True)
class EvaluationTrace:
    name: str
    recommendations: CandidateBatch
    recommendation_utility: np.ndarray
    queried: CandidateBatch
    query_utility: np.ndarray
    query_cost: np.ndarray

    @property
    def cumulative_cost(self) -> float:
        return float(self.query_cost.sum())


def top_candidates(
    batch: CandidateBatch,
    utility: np.ndarray,
    count: int,
) -> tuple[CandidateBatch, np.ndarray]:
    indices = np.argsort(utility)[-min(count, len(batch)) :]
    return (
        CandidateBatch(
            mixtures=batch.mixtures[indices],
            model_scales=batch.model_scales[indices],
            training_steps=batch.training_steps[indices],
        ),
        utility[indices],
    )
