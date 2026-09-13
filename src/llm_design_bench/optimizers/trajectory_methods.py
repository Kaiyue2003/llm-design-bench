from __future__ import annotations

import math

import torch
from torch import nn

from llm_design_bench.optimizers.additional_metadata import additional_metadata
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_components import (
    ContinuousTorchMethod, Diffusion, GaussianEnsemble, batches,
    classifier_free_condition, gaussian_log_prob, noise_like, step_loss, train_diffusion,
)


def construct_trajectories(x, y, c, *, count, length, generator, neighbors=None):
    """Build improving trajectories within a single observed fidelity.

    BONET uses bootstrap-and-sort. GTG walks to nearby equal-or-better
    observations. No synthetic labels or oracle calls enter either builder.
    """
    if c.shape[1]:
        _, inverse = torch.unique(c, dim=0, return_inverse=True)
    else:
        inverse = torch.zeros(len(x), device=x.device, dtype=torch.long)
    groups = [torch.where(inverse == group)[0] for group in inverse.unique()]
    trajectories = []
    for _ in range(count):
        group = groups[int(torch.randint(len(groups), (), device=x.device, generator=generator))]
        if neighbors is None:
            choices = group[torch.randint(len(group), (length,), device=x.device, generator=generator)]
            ids = choices[y[choices, 0].argsort(stable=True)]
        else:
            current = group[torch.randint(len(group), (), device=x.device, generator=generator)]
            path = [current]
            for _ in range(length - 1):
                candidates = group[y[group, 0] >= y[current, 0]]
                unseen = candidates[~torch.isin(candidates, torch.stack(path))]
                if len(unseen):
                    candidates = unseen
                distance = (x[candidates] - x[current]).square().sum(-1)
                nearest = candidates[distance.argsort(stable=True)[:neighbors]]
                current = nearest[torch.randint(len(nearest), (), device=x.device, generator=generator)]
                path.append(current)
            ids = torch.stack(path)
        trajectories.append(ids)
    indices = torch.stack(trajectories)
    return x[indices], y[indices], c[indices], indices


def regret_to_go(utility, ceiling):
    regret = (ceiling - utility).clamp_min(0)
    return regret.flip(1).cumsum(1).flip(1)


class RegretTransformer(nn.Module):
    def __init__(self, dimension, context_dim, hidden, length):
        super().__init__()
        self.inputs = nn.Linear(dimension + context_dim + 1, hidden)
        self.positions = nn.Parameter(torch.zeros(1, length, hidden))
        nn.init.normal_(self.positions, std=0.02)
        layer = nn.TransformerEncoderLayer(hidden, nhead=4 if hidden % 4 == 0 else 1,
                                           dim_feedforward=2 * hidden, dropout=0.0,
                                           batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output = nn.Linear(hidden, 2 * dimension)

    def forward(self, previous_designs, regret, context):
        length = previous_designs.shape[1]
        tokens = self.inputs(torch.cat([previous_designs, regret, context], -1)) + self.positions[:, :length]
        mask = torch.ones((length, length), device=tokens.device, dtype=torch.bool).triu(1)
        hidden = self.transformer(tokens, mask=mask)
        mean, logvar = self.output(hidden).chunk(2, -1)
        return mean, logvar.clamp(-8, 2)


@register_method("bonet")
class BONET(ContinuousTorchMethod):
    metadata = additional_metadata("bonet", "within-fidelity bootstrap-sorted trajectories and Gaussian causal transformer",
                                   "strictly offline rollout updates regret using a trained proxy instead of online oracle evaluations")

    def __init__(self, *, trajectory_length=16, trajectory_count=128,
                 regret_budget=0.0, sampling_temperature=0.1, **kwargs):
        super().__init__(**kwargs)
        if trajectory_length < 2 or trajectory_count < 1 or regret_budget < 0 or sampling_temperature < 0:
            raise ValueError("invalid trajectory configuration")
        self.trajectory_length, self.trajectory_count = int(trajectory_length), int(trajectory_count)
        self.regret_budget, self.sampling_temperature = float(regret_budget), float(sampling_temperature)

    def fit_prepared(self, problem, *, context, generator):
        self.prepare(problem)
        x, y, c, _ = construct_trajectories(self._x, self._y, self._c, count=self.trajectory_count,
                                            length=self.trajectory_length, generator=generator)
        self._ceiling = self._y.max()
        rtg = regret_to_go(y, self._ceiling) / self.trajectory_length
        previous = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
        self._transformer = RegretTransformer(self._dimension, self._c.shape[1], self.hidden_size, self.trajectory_length).to(self._x)
        optimizer = torch.optim.Adam(self._transformer.parameters(), lr=self.learning_rate)
        final = 0.0
        for _ in range(self.epochs):
            for ids in batches(len(x), self.batch_size, generator, x.device):
                mean, logvar = self._transformer(previous[ids], rtg[ids], c[ids])
                loss = -gaussian_log_prob(x[ids], mean, logvar).mean()
                final = step_loss(loss, optimizer, self._transformer.parameters())
        self._transformer.eval()
        self._proxy = GaussianEnsemble(self._dimension + self._c.shape[1], self.hidden_size).to(self._x)
        self._proxy.fit(torch.cat([self._x, self._c], -1), self._y, epochs=self.epochs,
                        batch_size=self.batch_size, lr=self.learning_rate, generator=generator)
        return {"trajectory_nll": final, "trajectory_count": len(x), "trajectory_length": self.trajectory_length}

    def propose_prepared(self, problem, *, context, generator):
        count = context.candidate_budget
        previous = self._x.new_zeros(count, 1, self._dimension)
        remaining = self._x.new_full((count, 1, 1), self.regret_budget)
        regrets = remaining / self.trajectory_length
        c = self.target_context(count)[:, None]
        with torch.no_grad():
            for index in range(self.trajectory_length):
                mean, logvar = self._transformer(previous, regrets, c.expand(-1, index + 1, -1))
                candidate = mean[:, -1] + self.sampling_temperature * (0.5 * logvar[:, -1]).exp() * noise_like(mean[:, -1], generator)
                candidate = self.project(candidate, problem)
                predicted, _ = self._proxy(torch.cat([candidate, c[:, 0]], -1))
                remaining = (remaining - (self._ceiling - predicted).clamp_min(0)[:, None]).clamp_min(0)
                if index + 1 < self.trajectory_length:
                    previous = torch.cat([previous, candidate[:, None]], 1)
                    regrets = torch.cat([regrets, remaining / self.trajectory_length], 1)
        self._diagnostics["rollout_regret_source"] = "training-only Gaussian ensemble; zero oracle calls"
        return self.decode(candidate, problem)


@register_method("gtg")
class GTG(ContinuousTorchMethod):
    metadata = additional_metadata("gtg", "within-fidelity nearest-neighbor improving trajectories",
                                   "flattened trajectory MLP diffusion instead of temporal U-Net; mean-return CFG and anchored start")

    def __init__(self, *, trajectory_length=16, trajectory_count=128, neighbors=10,
                 diffusion_steps=100, guidance=2.0, target_margin=0.5, **kwargs):
        super().__init__(**kwargs)
        if trajectory_length < 2 or min(trajectory_count, neighbors) < 1 or diffusion_steps < 2 or guidance < 0 or target_margin < 0:
            raise ValueError("invalid guided trajectory configuration")
        self.trajectory_length, self.trajectory_count, self.neighbors = int(trajectory_length), int(trajectory_count), int(neighbors)
        self.diffusion_steps, self.guidance, self.target_margin = int(diffusion_steps), float(guidance), float(target_margin)

    def fit_prepared(self, problem, *, context, generator):
        self.prepare(problem)
        x, y, c, _ = construct_trajectories(self._x, self._y, self._c, count=self.trajectory_count,
                                            length=self.trajectory_length, neighbors=self.neighbors, generator=generator)
        trajectory_context = torch.cat([c[:, 0], x[:, 0]], -1)
        returns = y.mean(1)
        self._return_max = returns.max()
        self._diffusion = Diffusion(self.trajectory_length * self._dimension,
                                    2 + trajectory_context.shape[1], self.hidden_size, self.diffusion_steps).to(self._x)
        final = train_diffusion(self, self._diffusion, x.flatten(1), returns, trajectory_context, generator)
        return {"trajectory_diffusion_loss": final, "trajectory_count": len(x), "trajectory_length": self.trajectory_length}

    def propose_prepared(self, problem, *, context, generator):
        count = context.candidate_budget
        anchor = self.initial(count)
        c = torch.cat([self.target_context(count), anchor], -1)
        returns = (self._return_max + self.target_margin).expand(count, 1)
        cond = classifier_free_condition(returns, c)
        uncond = classifier_free_condition(returns, c, unconditional=True)

        def project_trajectory(flat):
            sequence = self.project(flat.reshape(-1, self._dimension), problem).reshape(count, self.trajectory_length, -1)
            sequence = torch.cat([anchor[:, None], sequence[:, 1:]], 1)
            return sequence.flatten(1)

        with torch.no_grad():
            generated = self._diffusion.sample(cond, generator, steps=self.steps, guidance=self.guidance,
                                               unconditional=uncond, project=project_trajectory)
            sequence = generated.reshape(count, self.trajectory_length, self._dimension)
        self._diagnostics["trajectory_target_mean_return"] = float(returns[0, 0])
        return self.decode(sequence[:, -1], problem)
