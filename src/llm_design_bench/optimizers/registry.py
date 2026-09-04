from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import TypeVar

from llm_design_bench.optimizers.base import (
    MethodCapabilities,
    MethodMetadata,
    OfflineBBOMethod,
)


MethodType = TypeVar("MethodType", bound=type[OfflineBBOMethod])
MethodFactory = Callable[..., OfflineBBOMethod]

_METHODS: dict[str, MethodFactory] = {}
_METADATA: dict[str, MethodMetadata] = {}
_CAPABILITIES: dict[str, MethodCapabilities] = {}


def register_method(
    method_id: str | None = None,
) -> Callable[[MethodType], MethodType]:
    """Register an offline method class and return it unchanged."""

    def decorator(method_class: MethodType) -> MethodType:
        if not isinstance(method_class, type) or not issubclass(
            method_class, OfflineBBOMethod
        ):
            raise TypeError("registered methods must subclass OfflineBBOMethod")
        if inspect.isabstract(method_class):
            raise TypeError("registered method classes must be concrete")
        metadata = method_class.resolved_metadata()
        capabilities = method_class.resolved_capabilities()
        resolved_id = method_id or metadata.method_id
        if resolved_id != metadata.method_id:
            raise ValueError("registry id must match metadata.method_id")
        if resolved_id in _METHODS:
            raise KeyError(f"method {resolved_id!r} is already registered")
        _METHODS[resolved_id] = method_class
        _METADATA[resolved_id] = metadata
        _CAPABILITIES[resolved_id] = capabilities
        return method_class

    return decorator


def make_method(method_id: str, **kwargs) -> OfflineBBOMethod:
    try:
        factory = _METHODS[method_id]
    except KeyError as exc:
        available = ", ".join(method_names()) or "<none>"
        raise KeyError(
            f"unknown method {method_id!r}; available methods: {available}"
        ) from exc
    method = factory(**kwargs)
    if not isinstance(method, OfflineBBOMethod):
        raise TypeError(f"factory for {method_id!r} did not return OfflineBBOMethod")
    return method


def method_names() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))


def list_methods() -> tuple[MethodMetadata, ...]:
    return tuple(_METADATA[name] for name in method_names())


def get_method_metadata(method_id: str) -> MethodMetadata:
    try:
        return _METADATA[method_id]
    except KeyError as exc:
        available = ", ".join(method_names()) or "<none>"
        raise KeyError(
            f"unknown method {method_id!r}; available methods: {available}"
        ) from exc


def get_method_capabilities(method_id: str) -> MethodCapabilities:
    try:
        return _CAPABILITIES[method_id]
    except KeyError as exc:
        available = ", ".join(method_names()) or "<none>"
        raise KeyError(
            f"unknown method {method_id!r}; available methods: {available}"
        ) from exc
