"""SPADE official-core-derived diffusion and regularization primitives.

Source: HarryYoung2018/spade, revision 586151bbb56e246f93ca97ce33f79887a13161bd.
Copyright (c) 2026 SPADE Authors, MIT (see bundled third_party/SPADE_LICENSE.txt).
Changes: generator-local initialization/sampling, dtype support, Torch kNN,
validated deterministic DDIM, and direct moment-based optional support LCB.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from llm_design_bench.optimizers.spade import SpadeMethod


class LocalLinear(nn.Module):
    """Linear layer with source-equivalent uniform initialization, no global RNG."""

    def __init__(
        self,
        before: int,
        after: int,
        *,
        generator: torch.Generator,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(after, before, device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.empty(after, device=device, dtype=dtype))
        bound = 1 / math.sqrt(before)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound, generator=generator)
            self.bias.uniform_(-bound, bound, generator=generator)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight, self.bias)


class ScalarDiffusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        config: SpadeMethod,
        *,
        generator: torch.Generator,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.steps = config.diff_steps
        hidden, time_dim = config.diff_hidden, config.diff_t_dim

        def linear(before, after):
            return LocalLinear(
                before, after, generator=generator, device=device, dtype=dtype
            )

        self.time_net = nn.Sequential(
            linear(time_dim, time_dim), nn.SiLU(), linear(time_dim, time_dim)
        )
        self.x_net = nn.Sequential(
            linear(input_dim, hidden), nn.SiLU(), linear(hidden, hidden), nn.SiLU()
        )
        self.y_net = linear(1, hidden)
        self.out = nn.Sequential(
            linear(2 * hidden + time_dim, hidden),
            nn.SiLU(),
            linear(hidden, hidden),
            nn.SiLU(),
            linear(hidden, 1),
        )
        self.register_buffer(
            "frequencies",
            torch.linspace(
                0, math.log(10000), time_dim // 2, device=device, dtype=dtype
            ).exp(),
        )
        beta = torch.linspace(
            config.diff_beta_start,
            config.diff_beta_end,
            self.steps,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("alpha_bar", torch.log1p(-beta).cumsum(0).exp())

    def epsilon(
        self, y: torch.Tensor, time: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        angles = time[:, None] * self.frequencies
        embedding = self.time_net(torch.cat([angles.sin(), angles.cos()], -1))
        return self.out(torch.cat([self.x_net(x), self.y_net(y), embedding], -1))

    def q_sample(
        self, y: torch.Tensor, indices: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha = self.alpha_bar[indices, None]
        return alpha.sqrt() * y + (1 - alpha).sqrt() * noise

    def loss(
        self, x: torch.Tensor, y: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        indices = torch.randint(
            self.steps, (len(x),), device=x.device, generator=generator
        )
        noise = torch.randn(
            y.shape, device=y.device, dtype=y.dtype, generator=generator
        )
        prediction = self.epsilon(
            self.q_sample(y, indices, noise),
            (indices.to(x.dtype) + 0.5) / self.steps,
            x,
        )
        return F.mse_loss(prediction, noise)

    def ddim(
        self, x: torch.Tensor, steps: int, generator: torch.Generator
    ) -> torch.Tensor:
        if not 2 <= steps <= self.steps:
            raise ValueError("DDIM steps must be between 2 and the diffusion horizon")
        indices = torch.linspace(self.steps - 1, 0, steps, device=x.device).long()
        y = torch.randn(
            (len(x), 1), device=x.device, dtype=x.dtype, generator=generator
        )
        for position, index in enumerate(indices):
            alpha = self.alpha_bar[index]
            time = ((index.to(x.dtype) + 0.5) / self.steps).expand(len(x))
            eps = self.epsilon(y, time, x)
            clean = (y - (1 - alpha).sqrt() * eps) / (alpha.sqrt() + 1e-8)
            if position + 1 < len(indices):
                previous = self.alpha_bar[indices[position + 1]]
                y = previous.sqrt() * clean + (1 - previous).sqrt() * eps
        return clean

    def samples(
        self,
        x: torch.Tensor,
        *,
        count: int,
        steps: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Return (draws, rows); caller controls autograd, including calibration."""
        if count < 2 or batch_size < 1 or not len(x):
            raise ValueError(
                "MC sampling needs >=2 draws, positive batch size and rows"
            )
        outputs = []
        for offset in range(0, count, 16):
            draws = min(16, count - offset)
            chunks = []
            for part in x.split(batch_size):
                chunks.append(
                    self.ddim(part.repeat(draws, 1), steps, generator).reshape(
                        draws, -1
                    )
                )
            outputs.append(torch.cat(chunks, 1))
        return torch.cat(outputs)


def calibration_loss(
    mean: torch.Tensor,
    utility: torch.Tensor,
    *,
    pairs: int,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    mean, utility = mean.flatten(), utility.flatten()
    loss = F.mse_loss(mean, utility)
    if pairs and len(utility) > 1:
        left = torch.randint(
            len(utility), (pairs,), device=utility.device, generator=generator
        )
        right = torch.randint(
            len(utility), (pairs,), device=utility.device, generator=generator
        )
        ordered = utility[left] > utility[right]
        if ordered.any():
            loss = (
                loss
                + F.softplus(
                    -temperature * (mean[left[ordered]] - mean[right[ordered]])
                ).mean()
            )
    return loss


def proximity_loss(
    mean: torch.Tensor,
    sigma: torch.Tensor,
    neighbor_mean: torch.Tensor,
    radius: torch.Tensor,
    *,
    margin: float,
    floor: float,
    slope: float,
) -> torch.Tensor:
    distance = (radius + 1e-8).log()
    return (
        F.relu(mean - neighbor_mean - margin * distance)
        + F.relu(floor + slope * distance - sigma)
    ).mean()


def lower_confidence_bound(
    samples: torch.Tensor,
    beta: float,
    *,
    radius: torch.Tensor | None = None,
    margin: float = 0.02,
    floor: float = 0.02,
    slope: float = 0.005,
) -> torch.Tensor:
    mean, sigma = samples.mean(0), samples.std(0, unbiased=False)
    if radius is not None:
        distance = (radius + 1e-8).log()
        mean = mean - margin * distance
        sigma = sigma.maximum(floor + slope * distance)
    return mean - beta * sigma


class SupportNeighbors:
    """Chunked exact joint-feature kNN; stable row-index tie breaks, self included."""

    def __init__(
        self, features: torch.Tensor, utility: torch.Tensor, k: int, chunk_size: int
    ) -> None:
        if len(features) < 2 or k < 2 or chunk_size < 1:
            raise ValueError("support kNN requires >=2 rows, k>=2, positive chunks")
        self.features = features.detach()
        self.utility = utility.detach()
        self.k = min(k, len(features))
        self.chunk_size = chunk_size

    @torch.no_grad()
    def query(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means, radii = [], []
        for query in values.split(self.chunk_size):
            best_dist = query.new_empty((len(query), 0))
            best_idx = torch.empty(
                (len(query), 0), device=query.device, dtype=torch.long
            )
            for start in range(0, len(self.features), self.chunk_size):
                reference = self.features[start : start + self.chunk_size]
                distances = torch.cdist(
                    query, reference, compute_mode="donot_use_mm_for_euclid_dist"
                )
                indices = torch.arange(
                    start, start + len(reference), device=query.device
                ).expand(len(query), -1)
                distances = torch.cat([best_dist, distances], 1)
                indices = torch.cat([best_idx, indices], 1)
                # Earlier chunks/indices win exact ties because merge is stable.
                order = distances.argsort(dim=1, stable=True)[:, : self.k]
                best_dist, best_idx = (
                    distances.gather(1, order),
                    indices.gather(1, order),
                )
            means.append(self.utility[best_idx].mean(1))
            radii.append(best_dist[:, -1])
        return torch.cat(means), torch.cat(radii)
