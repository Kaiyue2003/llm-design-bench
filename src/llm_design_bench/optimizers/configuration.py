"""Export reconstructible constructor settings, including forwarded defaults."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

_MISSING = object()


def resolve_constructor_configuration(
    method: object,
    requested: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve keyword constructor arguments without exporting learned state.

    Follow cooperative ``**kwargs`` forwarding through constructor definitions
    in MRO order. A constructor without ``**kwargs`` closes that boundary.
    Actual named attributes take precedence over requested values and defaults,
    preserving constructor normalization. Unusual constructors should override
    ``configuration()``; frozen-plan creation additionally checks reconstruction.
    """

    supplied = dict(requested or {})
    resolved: dict[str, Any] = {}
    accepted: set[str] = set()
    for owner in type(method).__mro__:
        if owner is object:
            break
        constructor = owner.__dict__.get("__init__")
        if constructor is None:
            continue
        try:
            signature = inspect.signature(constructor)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Cannot inspect {owner.__name__}.__init__; override configuration() "
                "to return complete reconstructible constructor arguments"
            ) from exc
        parameters = list(signature.parameters.values())
        if not parameters or parameters[0].kind not in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            raise TypeError(
                f"Unsupported constructor signature for {owner.__name__}; "
                "override configuration()"
            )
        forwards_keywords = False
        for parameter in parameters[1:]:
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                forwards_keywords = True
                continue
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.POSITIONAL_ONLY,
            }:
                raise TypeError(
                    f"Cannot reconstruct {owner.__name__} argument "
                    f"{parameter.name!r} as a keyword; override configuration()"
                )
            name = parameter.name
            accepted.add(name)
            if name in resolved:
                continue
            value = getattr(method, name, _MISSING)
            if value is _MISSING:
                value = supplied.get(name, parameter.default)
            if value is inspect.Parameter.empty:
                raise ValueError(
                    f"Cannot resolve required constructor argument {name!r} for "
                    f"{type(method).__name__}; store its constructor value or "
                    "override configuration()"
                )
            resolved[name] = value
        if not forwards_keywords:
            break

    unknown = set(supplied) - accepted
    if unknown:
        raise ValueError(
            f"Unresolved constructor arguments {sorted(unknown)!r} for "
            f"{type(method).__name__}; override configuration()"
        )
    try:
        json.dumps(resolved, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Constructor configuration for {type(method).__name__} must contain "
            "finite JSON-compatible values; override configuration()"
        ) from exc
    _validate_json_keys(resolved)
    return deepcopy(resolved)


def _validate_json_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Constructor configuration mappings require string keys")
        for child in value.values():
            _validate_json_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_json_keys(child)
