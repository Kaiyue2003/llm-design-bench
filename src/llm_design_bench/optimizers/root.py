from __future__ import annotations

import math

import torch
from torch import nn

from llm_design_bench.optimizers.additional_metadata import additional_metadata
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_components import ContinuousTorchMethod, batches, mlp, noise_like, step_loss


def bridge_sample(high, low, m, variance, noise):
    objective = m * (low - high) + variance.sqrt() * noise
    return high + objective, objective


def bridge_posterior(x, high, low, m, next_m, variance, next_variance, noise):
    sigma2 = ((variance - next_variance * (1 - m) ** 2 / (1 - next_m) ** 2) * next_variance / variance).clamp_min(0)
    residual_scale = ((next_variance - sigma2).clamp_min(0) / variance).sqrt()
    mean = (1 - next_m) * high + next_m * low + residual_scale * (x - (1 - m) * high - m * low)
    return mean + sigma2.sqrt() * noise


class BridgeNet(nn.Module):
    def __init__(self, dimension, context_dim, hidden):
        super().__init__()
        self.net = mlp(dimension + context_dim + 3 + 16, dimension, hidden)

    def forward(self, x, m, low_y, high_y, c, present):
        frequencies = torch.exp(torch.linspace(0, math.log(1000), 8, device=x.device, dtype=x.dtype))
        phase = m * frequencies
        return self.net(torch.cat([x, low_y * present, high_y * present, present, c, phase.sin(), phase.cos()], -1))


@register_method("root")
class ROOT(ContinuousTorchMethod):
    metadata = additional_metadata("root", "linear Brownian bridge with displacement objective and analytic stochastic reverse transitions",
                                   "within-fidelity quantile pairing and compact time-conditioned MLP")

    def __init__(self, *, source_quantile=0.5, target_quantile=0.8, bridge_variance=1.0,
                 guidance=1.5, target_margin=0.5, **kwargs):
        super().__init__(**kwargs)
        if not 0 < source_quantile < target_quantile < 1 or bridge_variance <= 0 or guidance < 0 or target_margin < 0:
            raise ValueError("invalid probabilistic bridge configuration")
        self.source_quantile, self.target_quantile = float(source_quantile), float(target_quantile)
        self.bridge_variance, self.guidance, self.target_margin = float(bridge_variance), float(guidance), float(target_margin)

    def fit_prepared(self, problem, *, context, generator):
        self.prepare(problem)
        if self._c.shape[1]:
            _, inverse = torch.unique(self._c, dim=0, return_inverse=True)
        else:
            inverse = torch.zeros(len(self._x), device=self._x.device, dtype=torch.long)
        groups = []
        for group_id in inverse.unique():
            ids = torch.where(inverse == group_id)[0]
            y = self._y[ids, 0]
            low = ids[y <= torch.quantile(y, self.source_quantile)]
            high = ids[y >= torch.quantile(y, self.target_quantile)]
            groups.append((low, high))
        self._bridge = BridgeNet(self._dimension, self._c.shape[1], self.hidden_size).to(self._x)
        optimizer = torch.optim.Adam(self._bridge.parameters(), lr=self.learning_rate)
        final = 0.0
        for _ in range(self.epochs):
            for batch in batches(len(self._x), self.batch_size, generator, self._x.device):
                lows, highs = [], []
                for _ in range(len(batch)):
                    low, high = groups[int(torch.randint(len(groups), (), device=self._x.device, generator=generator))]
                    lows.append(low[torch.randint(len(low), (), device=low.device, generator=generator)])
                    highs.append(high[torch.randint(len(high), (), device=high.device, generator=generator)])
                li, hi = torch.stack(lows), torch.stack(highs)
                m = 0.001 + 0.998 * torch.rand((len(li), 1), device=self._x.device, dtype=self._x.dtype, generator=generator)
                variance = 2 * self.bridge_variance * m * (1 - m)
                xt, target = bridge_sample(self._x[hi], self._x[li], m, variance, noise_like(self._x[li], generator))
                present = (torch.rand((len(li), 1), device=self._x.device, generator=generator) >= 0.1).to(self._x)
                prediction = self._bridge(xt, m, self._y[li], self._y[hi], self._c[li], present)
                final = step_loss((prediction - target).square().mean(), optimizer, self._bridge.parameters())
        self._low_ids = torch.cat([group[0] for group in groups]).unique()
        return {"bridge_loss": final, "source_stratum_size": len(self._low_ids), "paired_fidelity_groups": len(groups)}

    def propose_prepared(self, problem, *, context, generator):
        count = context.candidate_budget
        ids = self._low_ids[torch.randint(len(self._low_ids), (count,), device=self._x.device, generator=generator)]
        low, low_y = self._x[ids], self._y[ids]
        high_y, c = self.target_utility(count, self.target_margin), self.target_context(count)
        schedule = torch.linspace(0.999, 0.001, self.steps, device=low.device, dtype=low.dtype)
        x = low.clone()
        with torch.no_grad():
            for index, m in enumerate(schedule):
                time = m.expand(count, 1)
                conditional = self._bridge(x, time, low_y, high_y, c, torch.ones_like(low_y))
                unconditional = self._bridge(x, time, low_y, high_y, c, torch.zeros_like(low_y))
                objective = unconditional + self.guidance * (conditional - unconditional)
                high = self.project((x - objective).clamp(-8, 8), problem)
                if index + 1 == len(schedule):
                    x = high
                else:
                    next_m = schedule[index + 1]
                    variance = 2 * self.bridge_variance * m * (1 - m)
                    next_variance = 2 * self.bridge_variance * next_m * (1 - next_m)
                    x = bridge_posterior(x, high, low, m, next_m, variance, next_variance, noise_like(x, generator))
        return self.decode(x, problem)
