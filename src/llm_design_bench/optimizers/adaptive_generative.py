from __future__ import annotations

import copy
import math

import torch
from torch.nn import functional as F

from llm_design_bench.optimizers.additional_metadata import additional_metadata
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_components import (
    ConditionalVAE, ContinuousTorchMethod, GaussianEnsemble, batches,
    gaussian_log_prob, noise_like, step_loss,
)


def cbas_log_weights(log_original, log_adaptive, mean, std, threshold):
    """log[p0(x|z,c)/pt(x|z,c) * P(Y >= threshold | x,c)]."""
    survival = torch.special.log_ndtr((mean.reshape(-1) - threshold) / std.reshape(-1).clamp_min(1e-6))
    return (log_original - log_adaptive).clamp(-20, 20) + survival


@register_method("cbas")
class CbAS(ContinuousTorchMethod):
    metadata = additional_metadata("cbas", "Gaussian conditional VAE for continuous designs",
                                   "decoder density ratio uses the shared latent draw; log ratio clipped to +/-20")

    def __init__(self, *, latent_dim=16, quantile=0.9, adaptation_epochs=5,
                 population_size=256, ensemble_size=3, **kwargs):
        super().__init__(**kwargs)
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        if min(latent_dim, adaptation_epochs, population_size, ensemble_size) < 1:
            raise ValueError("component sizes must be positive")
        self.latent_dim, self.quantile = int(latent_dim), float(quantile)
        self.adaptation_epochs, self.population_size = int(adaptation_epochs), int(population_size)
        self.ensemble_size = int(ensemble_size)

    def fit_prepared(self, problem, *, context, generator):
        self.prepare(problem)
        self._vae = ConditionalVAE(self._dimension, self._c.shape[1], self.latent_dim, self.hidden_size).to(self._x)
        optimizer = torch.optim.Adam(self._vae.parameters(), lr=self.learning_rate)
        final = 0.0
        for _ in range(self.epochs):
            for ids in batches(len(self._x), self.batch_size, generator, self._x.device):
                final = step_loss(self._vae.loss(self._x[ids], self._c[ids], generator).mean(),
                                  optimizer, self._vae.parameters())
        self._prior = copy.deepcopy(self._vae).eval().requires_grad_(False)
        self._ensemble = GaussianEnsemble(self._dimension + self._c.shape[1], self.hidden_size, self.ensemble_size).to(self._x)
        nll = self._ensemble.fit(torch.cat([self._x, self._c], -1), self._y, epochs=self.epochs,
                                batch_size=self.batch_size, lr=self.learning_rate, generator=generator)
        return {"vae_loss": final, "ensemble_nll": nll}

    def propose_prepared(self, problem, *, context, generator):
        c = self.target_context(max(self.population_size, context.candidate_budget))
        threshold = self._y.new_tensor(-torch.inf)
        optimizer = torch.optim.Adam(self._vae.parameters(), lr=self.learning_rate)
        history, effective = [], []
        for _ in range(self.steps):
            with torch.no_grad():
                x, z = self._vae.sample(c, generator)
                current_mean, current_logvar = self._vae.distribution(z, c)
                prior_mean, prior_logvar = self._prior.distribution(z, c)
                # Score the feasible design, while weighting the original latent draw.
                feasible = self.project(x, problem)
                mean, std = self._ensemble(torch.cat([feasible, c], -1))
                threshold = torch.maximum(threshold, torch.quantile(mean[:, 0], self.quantile))
                logs = cbas_log_weights(gaussian_log_prob(x, prior_mean, prior_logvar),
                                       gaussian_log_prob(x, current_mean, current_logvar), mean, std, threshold)
                probabilities = torch.softmax(logs, dim=0)
                weights = len(x) * probabilities
                effective.append(float(1 / probabilities.square().sum()))
                history.append(float(threshold))
            for _ in range(self.adaptation_epochs):
                for ids in batches(len(x), self.batch_size, generator, x.device):
                    loss = (weights[ids] * self._vae.loss(x[ids], c[ids], generator)).mean()
                    step_loss(loss, optimizer, self._vae.parameters())
        with torch.no_grad():
            samples, _ = self._vae.sample(self.target_context(context.candidate_budget), generator)
        self._diagnostics.update(threshold_history=history, effective_sample_size=effective)
        return self.decode(samples, problem)


@register_method("mins")
class MINs(ContinuousTorchMethod):
    metadata = additional_metadata("mins", "weighted conditional GAN with mismatched-pair negatives",
                                   "finite target-utility search ranked by ensemble LCB and discriminator plausibility")

    def __init__(self, *, latent_dim=16, target_margin=1.0, target_grid_size=8,
                 weight_temperature=1.0, realism_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        if latent_dim < 1 or target_grid_size < 1 or target_margin < 0 or weight_temperature <= 0 or realism_weight < 0:
            raise ValueError("invalid inverse GAN configuration")
        self.latent_dim, self.target_grid_size = int(latent_dim), int(target_grid_size)
        self.target_margin, self.weight_temperature = float(target_margin), float(weight_temperature)
        self.realism_weight = float(realism_weight)

    def fit_prepared(self, problem, *, context, generator):
        self.prepare(problem)
        cond_dim = 1 + self._c.shape[1]
        self._generator_model = self.module(self.latent_dim + cond_dim, self._dimension)
        self._discriminator = self.module(self._dimension + cond_dim, 1)
        gopt = torch.optim.Adam(self._generator_model.parameters(), lr=self.learning_rate)
        dopt = torch.optim.Adam(self._discriminator.parameters(), lr=self.learning_rate)
        weights = torch.softmax(self._y[:, 0] / self.weight_temperature, 0) * len(self._y)
        gloss = dloss = 0.0
        for _ in range(self.epochs):
            for ids in batches(len(self._x), self.batch_size, generator, self._x.device):
                x, cond = self._x[ids], torch.cat([self._y[ids], self._c[ids]], -1)
                z = torch.randn((len(ids), self.latent_dim), device=x.device, dtype=x.dtype, generator=generator)
                fake = self._generator_model(torch.cat([z, cond], -1))
                positive = self._discriminator(torch.cat([x, cond], -1))[:, 0]
                negative = self._discriminator(torch.cat([fake.detach(), cond], -1))[:, 0]
                mismatch = self._discriminator(torch.cat([x.roll(1, 0), cond], -1))[:, 0]
                loss = weights[ids] * (F.softplus(-positive) + 0.5 * F.softplus(negative) + 0.5 * F.softplus(mismatch))
                dloss = step_loss(loss.mean(), dopt, self._discriminator.parameters())
                self._discriminator.requires_grad_(False)
                fake_logits = self._discriminator(torch.cat([fake, cond], -1))[:, 0]
                gloss = step_loss((weights[ids] * F.softplus(-fake_logits)).mean(), gopt, self._generator_model.parameters())
                self._discriminator.requires_grad_(True)
        self._ensemble = GaussianEnsemble(self._dimension + self._c.shape[1], self.hidden_size).to(self._x)
        self._ensemble.fit(torch.cat([self._x, self._c], -1), self._y, epochs=self.epochs,
                           batch_size=self.batch_size, lr=self.learning_rate, generator=generator)
        return {"generator_loss": gloss, "discriminator_loss": dloss}

    def propose_prepared(self, problem, *, context, generator):
        count = context.candidate_budget
        targets = self._y.max() + torch.linspace(0, self.target_margin, self.target_grid_size, device=self._x.device, dtype=self._x.dtype)
        y = targets.repeat_interleave(count)[:, None]
        c = self.target_context(len(y))
        with torch.no_grad():
            z = torch.randn((len(y), self.latent_dim), device=y.device, dtype=y.dtype, generator=generator)
            x = self.project(self._generator_model(torch.cat([z, y, c], -1)), problem)
            mean, std = self._ensemble(torch.cat([x, c], -1))
            realism = F.logsigmoid(self._discriminator(torch.cat([x, y, c], -1)))
            score = (mean - std + self.realism_weight * realism).reshape(self.target_grid_size, count)
            best = score.argmax(dim=0)
            selected = x.reshape(self.target_grid_size, count, -1)[best, torch.arange(count, device=x.device)]
        self._diagnostics["searched_target_utilities"] = targets.tolist()
        return self.decode(selected, problem)
