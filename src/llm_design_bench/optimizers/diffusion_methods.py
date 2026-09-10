from __future__ import annotations

import copy
import math

import torch

from llm_design_bench.optimizers.additional_metadata import additional_metadata
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_components import (
    ContinuousTorchMethod, Diffusion, GaussianEnsemble, batches,
    classifier_free_condition, noise_like, step_loss, train_diffusion,
)


@register_method("ddom")
class DDOM(ContinuousTorchMethod):
    metadata = additional_metadata("ddom", "cosine discrete VP schedule and deterministic DDIM replace the upstream continuous SDE solver",
                                   "exponential utility reweighting instead of histogram-smoothed weights")

    def __init__(self, *, diffusion_steps=100, guidance=2.0, target_margin=0.5,
                 weight_temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        if diffusion_steps < 2 or guidance < 0 or target_margin < 0 or weight_temperature <= 0:
            raise ValueError("invalid diffusion configuration")
        self.diffusion_steps, self.guidance = int(diffusion_steps), float(guidance)
        self.target_margin, self.weight_temperature = float(target_margin), float(weight_temperature)

    def fit_prepared(self, problem, *, context, generator):
        self.prepare(problem)
        self._diffusion = Diffusion(self._dimension, 2 + self._c.shape[1], self.hidden_size, self.diffusion_steps).to(self._x)
        weights = torch.softmax(self._y[:, 0] / self.weight_temperature, 0) * len(self._y)
        final = train_diffusion(self, self._diffusion, self._x, self._y, self._c, generator, weights=weights)
        return {"diffusion_loss": final, "diffusion_steps": self.diffusion_steps}

    def conditions(self, count):
        y, c = self.target_utility(count, self.target_margin), self.target_context(count)
        return classifier_free_condition(y, c), classifier_free_condition(y, c, unconditional=True)

    def propose_prepared(self, problem, *, context, generator):
        cond, uncond = self.conditions(context.candidate_budget)
        with torch.no_grad():
            x = self._diffusion.sample(cond, generator, steps=self.steps, guidance=self.guidance,
                                       unconditional=uncond, project=lambda x: self.project(x, problem))
        return self.decode(x, problem)


def probability_flow_log_density(model, x, condition, generator, *, steps=16):
    """Hutchinson-trace likelihood for the cosine VP probability-flow ODE.

    Midpoint integration stops at t=.98 to avoid the cosine endpoint
    singularity. This is a numerical likelihood estimate, not an exact density.
    """
    if steps < 2:
        raise ValueError("likelihood_steps must be at least two")
    current = x.detach().clone()
    probe = (torch.randint(2, x.shape, device=x.device, generator=generator) * 2 - 1).to(x)
    divergence_integral = x.new_zeros(len(x))
    dt = 0.98 / steps

    def drift_and_divergence(state, time, divergence=True):
        beta = math.pi / 1.008 * math.tan((time + 0.008) / 1.008 * math.pi / 2)
        ab = math.cos((time + 0.008) / 1.008 * math.pi / 2) ** 2 / math.cos(0.008 / 1.008 * math.pi / 2) ** 2
        with torch.enable_grad():
            state = state.detach().requires_grad_(True)
            t = state.new_full((len(state),), time * model.timesteps - 0.5)
            epsilon = model(state, t, condition)
            drift = 0.5 * beta * (epsilon / math.sqrt(max(1 - ab, 1e-6)) - state)
            if divergence:
                derivative = torch.autograd.grad((drift * probe).sum(), state)[0]
                trace = (derivative * probe).sum(-1)
            else:
                trace = None
        return drift.detach(), None if trace is None else trace.detach()

    for index in range(steps):
        time = max(index * dt, 1e-4)
        first, _ = drift_and_divergence(current, time, False)
        middle = current + 0.5 * dt * first
        drift, trace = drift_and_divergence(middle, time + 0.5 * dt)
        current = current + dt * drift
        divergence_integral += dt * trace
    log_prior = -0.5 * (current.square() + math.log(2 * math.pi)).sum(-1)
    result = log_prior + divergence_integral
    if not torch.isfinite(result).all():
        raise FloatingPointError("probability-flow likelihood became non-finite")
    return result


def log_utility_prior(labels, training_labels):
    bandwidth = max(0.1, 1.06 * float(training_labels.std(unbiased=False)) * len(training_labels) ** -0.2)
    delta = (labels.reshape(-1, 1) - training_labels.reshape(1, -1)) / bandwidth
    return torch.logsumexp(-0.5 * delta.square(), -1) - math.log(len(training_labels) * bandwidth * math.sqrt(2 * math.pi))


def reverse_kl_proxy_loss(log_proxy, log_diffusion):
    # Score-function gradient of KL(q_proxy(y|x) || p_diffusion(y|x)).
    weights = (1 + log_proxy.detach() - log_diffusion.detach()).clamp(1 - math.log(100), 1 + math.log(100))
    return (weights * log_proxy).mean()


@register_method("rgd")
class RGD(DDOM):
    metadata = additional_metadata("rgd", "cosine VP diffusion with a compact Gaussian ensemble proxy",
                                   "midpoint probability-flow likelihood with a seeded Hutchinson trace and terminal t=.98",
                                   "clipped reverse-KL proxy refinement; normalized gradients on denoised samples")

    def __init__(self, *, refinement_rounds=2, refinement_candidates=16,
                 likelihood_samples=8, likelihood_steps=16, refinement_weight=0.1,
                 proxy_guidance=0.05, **kwargs):
        super().__init__(**kwargs)
        if min(refinement_rounds, refinement_candidates, likelihood_samples) < 1 or likelihood_steps < 2:
            raise ValueError("refinement sizes must be positive and likelihood_steps >= 2")
        if refinement_weight < 0 or proxy_guidance < 0:
            raise ValueError("refinement and guidance weights must be nonnegative")
        self.refinement_rounds, self.refinement_candidates = int(refinement_rounds), int(refinement_candidates)
        self.likelihood_samples, self.likelihood_steps = int(likelihood_samples), int(likelihood_steps)
        self.refinement_weight, self.proxy_guidance = float(refinement_weight), float(proxy_guidance)

    def fit_prepared(self, problem, *, context, generator):
        summary = dict(super().fit_prepared(problem, context=context, generator=generator))
        self._proxy = GaussianEnsemble(self._dimension + self._c.shape[1], self.hidden_size).to(self._x)
        features = torch.cat([self._x, self._c], -1)
        summary["proxy_nll"] = self._proxy.fit(features, self._y, epochs=self.epochs,
                                              batch_size=self.batch_size, lr=self.learning_rate, generator=generator)
        optimizer = torch.optim.Adam(self._proxy.parameters(), lr=self.learning_rate)
        refinements = []
        for _ in range(self.refinement_rounds):
            candidates = self.initial(self.refinement_candidates)
            c = self.target_context(len(candidates))
            for _ in range(self.steps):
                candidates = candidates.detach().requires_grad_(True)
                mean, _ = self._proxy(torch.cat([candidates, c], -1))
                gradient = torch.autograd.grad(mean.sum(), candidates)[0]
                candidates = self.project(candidates + 0.05 * gradient, problem).detach()
            inputs = candidates.repeat_interleave(self.likelihood_samples, 0)
            contexts = self.target_context(len(inputs))
            with torch.no_grad():
                mean, std = self._proxy(torch.cat([inputs, contexts], -1))
                labels = mean + std * noise_like(mean, generator)
            cond = classifier_free_condition(labels, contexts)
            uncond = classifier_free_condition(labels, contexts, unconditional=True)
            joint = probability_flow_log_density(self._diffusion, inputs, cond, generator, steps=self.likelihood_steps)
            marginal = probability_flow_log_density(self._diffusion, inputs, uncond, generator, steps=self.likelihood_steps)
            log_target = joint + log_utility_prior(labels, self._y) - marginal
            for _ in range(self.steps):
                mean, std = self._proxy(torch.cat([inputs, contexts], -1))
                logq = torch.distributions.Normal(mean[:, 0], std[:, 0]).log_prob(labels[:, 0])
                kl = reverse_kl_proxy_loss(logq, log_target)
                ids = torch.randint(len(features), (min(len(features), self.batch_size),), device=inputs.device, generator=generator)
                real_mean, real_std = self._proxy(features[ids])
                nll = -torch.distributions.Normal(real_mean, real_std).log_prob(self._y[ids]).mean()
                step_loss(nll + self.refinement_weight * kl, optimizer, self._proxy.parameters())
            refinements.append(float(kl.detach()))
        self._diagnostics["reverse_kl_refinement_losses"] = refinements
        return summary

    def propose_prepared(self, problem, *, context, generator):
        cond, uncond = self.conditions(context.candidate_budget)

        def guide(x):
            with torch.enable_grad():
                variable = x.detach().requires_grad_(True)
                mean, std = self._proxy(torch.cat([variable, self.target_context(len(x))], -1))
                gradient = torch.autograd.grad((mean - std).sum(), variable)[0]
            return self.proxy_guidance * gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1.0)

        with torch.no_grad():
            x = self._diffusion.sample(cond, generator, steps=self.steps, guidance=self.guidance,
                                       unconditional=uncond, gradient=guide, project=lambda x: self.project(x, problem))
        return self.decode(x, problem)


@register_method("demo")
class DEMO(DDOM):
    metadata = additional_metadata("demo", "surrogate-gradient pseudo-targets and target-distribution diffusion fine-tuning",
                                   "partial cosine noising and DDIM editing replace the upstream VP-SDE/Heun solver")

    def __init__(self, *, edit_fraction=0.4, editing_epochs=20, pseudo_samples=128,
                 editing_rate=0.05, **kwargs):
        super().__init__(**kwargs)
        if not 0 < edit_fraction <= 1 or min(editing_epochs, pseudo_samples) < 1 or editing_rate <= 0:
            raise ValueError("invalid diffusion editing configuration")
        self.edit_fraction, self.editing_rate = float(edit_fraction), float(editing_rate)
        self.editing_epochs, self.pseudo_samples = int(editing_epochs), int(pseudo_samples)

    def fit_prepared(self, problem, *, context, generator):
        summary = dict(super().fit_prepared(problem, context=context, generator=generator))
        self._proxy = GaussianEnsemble(self._dimension + self._c.shape[1], self.hidden_size).to(self._x)
        self._proxy.fit(torch.cat([self._x, self._c], -1), self._y, epochs=self.epochs,
                        batch_size=self.batch_size, lr=self.learning_rate, generator=generator)
        x = self.initial(max(self.pseudo_samples, context.candidate_budget))
        c = self.target_context(len(x))
        for _ in range(self.steps):
            x = x.detach().requires_grad_(True)
            mean, _ = self._proxy(torch.cat([x, c], -1))
            gradient = torch.autograd.grad(mean.sum(), x)[0]
            x = self.project(x + self.editing_rate * gradient, problem).detach()
        with torch.no_grad():
            pseudo_y, _ = self._proxy(torch.cat([x, c], -1))
        self._pseudo_x = x
        self._target_diffusion = copy.deepcopy(self._diffusion)
        summary["target_diffusion_loss"] = train_diffusion(self, self._target_diffusion, x, pseudo_y, c,
                                                           generator, epochs=self.editing_epochs)
        self._diagnostics["pseudo_target_utility_range"] = [float(pseudo_y.min()), float(pseudo_y.max())]
        return summary

    def propose_prepared(self, problem, *, context, generator):
        cond, uncond = self.conditions(context.candidate_budget)
        index = max(0, min(self.diffusion_steps - 1, round(self.edit_fraction * (self.diffusion_steps - 1))))
        with torch.no_grad():
            base = self._pseudo_x[:context.candidate_budget]
            t = torch.full((len(base),), index, device=base.device, dtype=torch.long)
            noisy = self._target_diffusion.q_sample(base, t, noise_like(base, generator))
            edited = self._target_diffusion.sample(cond, generator, steps=self.steps, guidance=self.guidance,
                                                   unconditional=uncond, initial=noisy, start_index=index,
                                                   project=lambda x: self.project(x, problem))
        self._diagnostics["editing_start_index"] = index
        return self.decode(edited, problem)
