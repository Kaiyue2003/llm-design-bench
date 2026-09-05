from __future__ import annotations

import torch


def select_top_unique_designs(
    designs: torch.Tensor,
    utility: torch.Tensor,
    count: int,
    *,
    decimals: int = 12,
) -> torch.Tensor:
    """Return up to ``count`` unique designs in descending utility order."""

    if count < 1:
        raise ValueError("count must be positive")
    if designs.ndim != 2:
        raise ValueError("designs must be a matrix")
    if utility.ndim != 1 or len(utility) != len(designs):
        raise ValueError("utility must contain one value per design")
    if not len(designs):
        raise ValueError("logged dataset does not contain any designs")

    selected: list[torch.Tensor] = []
    seen: set[tuple[float, ...]] = set()
    order = torch.argsort(utility, descending=True, stable=True)
    for index in order.detach().cpu().tolist():
        row = designs[index]
        key = tuple(round(float(value), decimals) for value in row.detach().cpu())
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) == count:
            break

    return torch.stack(selected)


def repeat_rows(rows: torch.Tensor, count: int) -> torch.Tensor:
    """Repeat a non-empty matrix row-wise until it has exactly ``count`` rows."""

    if count < 1:
        raise ValueError("count must be positive")
    if rows.ndim != 2 or not len(rows):
        raise ValueError("rows must be a non-empty matrix")
    indices = torch.arange(count, device=rows.device) % len(rows)
    return rows.index_select(0, indices)
