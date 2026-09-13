from __future__ import annotations

import torch
from torch.nn import functional as F

from llm_design_bench.optimizers.additional_metadata import additional_metadata
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_components import ContinuousTorchMethod, Diffusion, batches, noise_like, step_loss


def calibration_loss(predicted_mean, observed, generator):
    predicted_mean, observed = predicted_mean.reshape(-1), observed.reshape(-1)
    mse = F.mse_loss(predicted_mean, observed)
    perm = torch.randperm(len(observed), device=observed.device, generator=generator)
    sign = torch.sign(observed - observed[perm])
    valid = sign != 0
    rank = F.softplus(-sign[valid] * (predicted_mean[valid] - predicted_mean[perm][valid])).mean() if valid.any() else mse.new_zeros(())
    return mse + rank


def support_statistics(query, training_features, training_y, k, *, exclude_ids=None):
    distances = torch.cdist(query, training_features)
    if exclude_ids is not None and len(training_features) > 1:
        distances[torch.arange(len(query), device=query.device), exclude_ids] = torch.inf
        k = min(k, len(training_features) - 1)
    else:
        k = min(k, len(training_features))
    values, indices = distances.topk(k, dim=1, largest=False)
    return training_y[indices, 0].mean(-1), values[:, -1].clamp_min(1e-6)


def support_lcb(samples, distance, *, beta, mean_penalty, sigma_base, sigma_distance):
    mean = samples.mean(0)
    std = samples.std(0, unbiased=False)
    log_radius = distance.log()
    adjusted_mean = mean - mean_penalty * log_radius
    floor = (sigma_base + sigma_distance * log_radius).clamp_min(0)
    return adjusted_mean - beta * torch.maximum(std, floor)


@register_method("spade")
class SPADE(ContinuousTorchMethod):
    metadata = additional_metadata("spade", "scalar conditional diffusion with moment/rank and support-proximity training losses",
                                   "cosine schedule, compact MLP, exact torch kNN, and seeded evolutionary LCB search")

    def __init__(self, *, diffusion_steps=100, sampling_steps=20, mc_samples=8,
                 calibration_samples=4, calibration_steps=4, calibration_weight=0.1,
                 support_weight=0.1, support_k=5, mean_penalty=0.1,
                 sigma_base=0.1, sigma_distance=0.1, lcb_beta=1.0, mutation_scale=0.2, **kwargs):
        super().__init__(**kwargs)
        if diffusion_steps < 2 or min(sampling_steps, mc_samples, calibration_samples, calibration_steps, support_k) < 1:
            raise ValueError("diffusion and support sizes must be positive")
        if min(calibration_weight, support_weight, mean_penalty, sigma_base, sigma_distance, lcb_beta, mutation_scale) < 0:
            raise ValueError("SPADE weights must be nonnegative")
        self.diffusion_steps, self.sampling_steps, self.mc_samples = int(diffusion_steps), int(sampling_steps), int(mc_samples)
        self.calibration_samples, self.calibration_steps, self.support_k = int(calibration_samples), int(calibration_steps), int(support_k)
        self.calibration_weight, self.support_weight = float(calibration_weight), float(support_weight)
        self.mean_penalty, self.sigma_base, self.sigma_distance = float(mean_penalty), float(sigma_base), float(sigma_distance)
        self.lcb_beta, self.mutation_scale = float(lcb_beta), float(mutation_scale)

    def samples(self, features, generator, *, count, steps):
        values = self._diffusion.sample(features.repeat(count, 1), generator, steps=steps)
        return values.reshape(count, len(features))

    def fit_prepared(self, problem, *, context, generator):
        self.prepare(problem)
        self._features = torch.cat([self._x, self._c], -1)
        self._diffusion = Diffusion(1, self._features.shape[1], self.hidden_size, self.diffusion_steps).to(self._x)
        optimizer = torch.optim.Adam(self._diffusion.parameters(), lr=self.learning_rate)
        final = calibration = support = 0.0
        for _ in range(self.epochs):
            for ids in batches(len(self._x), self.batch_size, generator, self._x.device):
                features, labels = self._features[ids], self._y[ids]
                loss = self._diffusion.loss(labels, features, generator).mean()
                if self.calibration_weight or self.support_weight:
                    draws = self.samples(features, generator, count=self.calibration_samples, steps=self.calibration_steps)
                    mean, std = draws.mean(0), draws.std(0, unbiased=False)
                    cal = calibration_loss(mean, labels, generator)
                    nearby, radius = support_statistics(features, self._features, self._y, self.support_k, exclude_ids=ids)
                    log_radius = radius.log()
                    shrink = F.relu(mean - nearby - self.mean_penalty * log_radius)
                    floor = F.relu(self.sigma_base + self.sigma_distance * log_radius - std)
                    proximity = (shrink + floor).mean()
                    loss = loss + self.calibration_weight * cal + self.support_weight * proximity
                    calibration, support = float(cal.detach()), float(proximity.detach())
                final = step_loss(loss, optimizer, self._diffusion.parameters())
        self._diffusion.eval()
        return {"diffusion_calibrated_loss": final, "calibration_loss": calibration, "support_proximity_loss": support}

    def propose_prepared(self, problem, *, context, generator):
        count = context.candidate_budget
        population_size = max(32, 2 * count)
        population = self.project(self.initial(population_size) + 0.05 * noise_like(self.initial(population_size), generator), problem)
        elite_count = max(count, population_size // 4)
        with torch.no_grad():
            for generation in range(self.steps + 1):
                features = torch.cat([population, self.target_context(len(population))], -1)
                draws = self.samples(features, generator, count=self.mc_samples, steps=self.sampling_steps)
                _, radius = support_statistics(features, self._features, self._y, self.support_k)
                score = support_lcb(draws, radius, beta=self.lcb_beta, mean_penalty=self.mean_penalty,
                                    sigma_base=self.sigma_base, sigma_distance=self.sigma_distance)
                order = score.argsort(descending=True, stable=True)
                elites = population[order[:elite_count]]
                if generation == self.steps:
                    result = elites[:count]
                    break
                size = population_size - elite_count
                first = elites[torch.randint(len(elites), (size,), device=elites.device, generator=generator)]
                second = elites[torch.randint(len(elites), (size,), device=elites.device, generator=generator)]
                mix = torch.rand((size, 1), device=elites.device, dtype=elites.dtype, generator=generator)
                children = mix * first + (1 - mix) * second
                children = children + self.mutation_scale * (0.98 ** generation) * noise_like(children, generator)
                population = torch.cat([elites, self.project(children, problem)])
        self._diagnostics.update(final_best_lcb=float(score.max()), acquisition="support-adjusted LCB")
        return self.decode(result, problem)
