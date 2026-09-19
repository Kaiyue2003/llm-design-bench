from __future__ import annotations

import math

import torch

from llm_design_bench.optimizers.additional_metadata import additional_metadata
from llm_design_bench.optimizers.latent_gp import LatentGP
from llm_design_bench.optimizers.registry import register_method
from llm_design_bench.optimizers.torch_components import (
    ConditionalVAE,
    ContinuousTorchMethod,
    GaussianEnsemble,
    batches,
    step_loss,
)


def expected_improvement(mean, std, best):
    std = std.clamp_min(1e-8)
    z = (mean - best) / std
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2 * math.pi)
    return (mean - best) * cdf + std * pdf


@register_method("gabo")
class GABO(ContinuousTorchMethod):
    metadata = additional_metadata(
        "gabo",
        "conditional VAE latent representation with an adversarial Wasserstein source critic",
        "finite-grid dual coefficient; GPyTorch exact RBF GP with fixed median lengthscale and sequential analytic EI instead of BoTorch qEI",
        "fixed unit kernel variance and ridge noise; no latent input standardization or GP hyperparameter training",
    )

    def __init__(
        self,
        *,
        latent_dim=8,
        initial_points=32,
        acquisition_steps=20,
        acquisition_restarts=16,
        critic_steps=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if (
            min(
                latent_dim,
                initial_points,
                acquisition_steps,
                acquisition_restarts,
                critic_steps,
            )
            < 1
        ):
            raise ValueError("latent optimization sizes must be positive")
        self.latent_dim, self.initial_points = int(latent_dim), int(initial_points)
        self.acquisition_steps, self.acquisition_restarts, self.critic_steps = (
            int(acquisition_steps),
            int(acquisition_restarts),
            int(critic_steps),
        )

    def fit_prepared(self, problem, *, context, generator):
        self.prepare(problem)
        self._vae = ConditionalVAE(
            self._dimension, self._c.shape[1], self.latent_dim, self.hidden_size
        ).to(self._x)
        optimizer = torch.optim.Adam(self._vae.parameters(), lr=self.learning_rate)
        final = 0.0
        for _ in range(self.epochs):
            for ids in batches(
                len(self._x), self.batch_size, generator, self._x.device
            ):
                final = step_loss(
                    self._vae.loss(self._x[ids], self._c[ids], generator).mean(),
                    optimizer,
                    self._vae.parameters(),
                )
        self._vae.eval().requires_grad_(False)
        with torch.no_grad():
            self._reference_z, _ = self._vae.encode(self._x, self._c)
        self._surrogate = GaussianEnsemble(
            self._dimension + self._c.shape[1], self.hidden_size
        ).to(self._x)
        self._surrogate.fit(
            torch.cat([self._x, self._c], -1),
            self._y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            generator=generator,
        )
        self._surrogate.requires_grad_(False)
        self._critic = self.module(self.latent_dim, 1)
        return {"latent_vae_loss": final, "latent_gp_backend": "gpytorch"}

    def propose_prepared(self, problem, *, context, generator):
        count = max(self.initial_points, context.candidate_budget)
        z = torch.randn(
            (count, self.latent_dim),
            device=self._x.device,
            dtype=self._x.dtype,
            generator=generator,
        )
        critic_opt = torch.optim.RMSprop(
            self._critic.parameters(), lr=self.learning_rate
        )
        alphas = []

        def surrogate(latent):
            x, _ = self._vae.distribution(latent, self.target_context(len(latent)))
            x = self.project(x, problem)
            mean, _ = self._surrogate(torch.cat([x, self.target_context(len(x))], -1))
            return mean[:, 0]

        for _ in range(self.steps):
            self._critic.requires_grad_(True)
            for _ in range(self.critic_steps):
                ids = torch.randint(
                    len(self._reference_z),
                    (len(z),),
                    device=z.device,
                    generator=generator,
                )
                loss = (
                    self._critic(z.detach()).mean()
                    - self._critic(self._reference_z[ids]).mean()
                )
                step_loss(loss, critic_opt, self._critic.parameters())
                with torch.no_grad():
                    for parameter in self._critic.parameters():
                        parameter.clamp_(-0.05, 0.05)
            self._critic.requires_grad_(False)
            with torch.no_grad():
                prior = torch.randn(
                    (128, self.latent_dim),
                    device=z.device,
                    dtype=z.dtype,
                    generator=generator,
                )
                grid = torch.linspace(0, 1, 21, device=z.device, dtype=z.dtype)
                reference_critic = self._critic(self._reference_z).mean()
                distance = reference_critic - self._critic(prior)[:, 0]
                dual = (grid[:, None] - 1) * surrogate(prior)[None] + grid[
                    :, None
                ] * distance[None]
                alpha = grid[dual.min(dim=1).values.argmax()]
                values = (1 - alpha) * surrogate(z) - alpha * (
                    reference_critic - self._critic(z)[:, 0]
                )
                alphas.append(float(alpha))
            gp = LatentGP(z, values)
            starts = torch.randn(
                (self.acquisition_restarts, self.latent_dim),
                device=z.device,
                dtype=z.dtype,
                generator=generator,
            )
            starts.requires_grad_(True)
            optimizer = torch.optim.Adam([starts], lr=0.05)
            for _ in range(self.acquisition_steps):
                mean, std = gp.predict(starts)
                loss = -expected_improvement(mean, std, values.max()).mean()
                step_loss(loss, optimizer, [starts])
                with torch.no_grad():
                    starts.clamp_(-3, 3)
            with torch.no_grad():
                mean, std = gp.predict(starts)
                best = expected_improvement(mean, std, values.max()).argmax()
                z = torch.cat([z, starts[best : best + 1].detach()])
        with torch.no_grad():
            score = (1 - alpha) * surrogate(z) - alpha * (
                reference_critic - self._critic(z)[:, 0]
            )
            selected = z[
                score.argsort(descending=True, stable=True)[: context.candidate_budget]
            ]
            x, _ = self._vae.distribution(selected, self.target_context(len(selected)))
        self._diagnostics.update(
            source_critic_alpha=alphas,
            latent_surrogate_evaluations=len(z),
            latent_gp=gp.diagnostics(),
        )
        return self.decode(x, problem)
