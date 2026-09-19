from __future__ import annotations

import torch

from llm_design_bench.spaces import SimplexSpace


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


def initialize_candidate_designs(
    problem,
    *,
    candidate_budget: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int]:
    """Combine top unique logs with random feasible designs for search."""

    logged = select_top_unique_designs(
        problem.train_designs,
        problem.train_utility,
        candidate_budget,
    )
    random_count = candidate_budget - len(logged)
    if random_count:
        sampled = problem.design_space.sample(
            random_count,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        designs = torch.cat([logged, sampled], dim=0)
    else:
        designs = logged

    designs = move_designs_to_interior(designs, problem.design_space)
    return designs, len(logged), random_count


def initialize_mixed_candidate_designs(
    problem,
    *,
    candidate_budget: int,
    random_fraction: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int, int]:
    """Mix strong logged starts with random starts for deterministic surrogates."""

    if not 0.0 <= random_fraction < 1.0:
        raise ValueError("random_fraction must be in [0, 1)")
    requested_random = int(round(candidate_budget * random_fraction))
    if random_fraction > 0.0 and candidate_budget > 1:
        requested_random = max(1, requested_random)
    logged_target = max(1, candidate_budget - requested_random)
    logged = select_top_unique_designs(
        problem.train_designs,
        problem.train_utility,
        logged_target,
    )
    random_count = candidate_budget - len(logged)
    if random_count:
        sampled = problem.design_space.sample(
            random_count,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        designs = torch.cat([logged, sampled], dim=0)
    else:
        designs = logged
    designs = move_designs_to_interior(designs, problem.design_space)
    return designs, len(logged), random_count


def move_designs_to_interior(designs: torch.Tensor, design_space) -> torch.Tensor:
    """Keep simplex logits trainable when logged components contain zeros."""

    if not isinstance(design_space, SimplexSpace):
        return designs
    epsilon = max(float(torch.finfo(designs.dtype).eps), 1e-8)
    interior = designs.clamp_min(epsilon)
    return interior / interior.sum(dim=1, keepdim=True)
