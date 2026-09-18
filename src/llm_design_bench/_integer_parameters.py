"""Strict execution parameters, separate from permissive CSV number parsing."""

from numbers import Integral


def require_integer(value: object, *, name: str) -> int:
    """Accept integral scalar types without truncating or parsing other values.

    NumPy integer scalars implement Integral; Python bool is explicitly excluded.
    Normalization happens only after checking the original input's type.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not a boolean, float, or string")
    return int(value)


def optional_nonnegative_integer(value: object, *, name: str) -> int | None:
    """Retain an unspecified metadata seed, otherwise require a valid integer."""
    if value is None:
        return None
    normalized = require_integer(value, name=name)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized
