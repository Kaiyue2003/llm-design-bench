"""Native, generator-controlled CQL/SAC for certified offline PGS replay."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from llm_design_bench.problem import RunContext


def mlp(input_dim: int, hidden_size: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_dim),
    )


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.network = mlp(state_dim, hidden_size, 2 * action_dim)
        self.log_std_multiplier = nn.Parameter(torch.tensor(1.0))
        self.log_std_offset = nn.Parameter(torch.tensor(-1.0))

    def parameters_at(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = self.network(states).chunk(2, dim=-1)
        return mean, (self.log_std_multiplier * raw + self.log_std_offset).clamp(-20, 2)

    def sample(
        self, states: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.parameters_at(states)
        noise = torch.randn(
            mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
        )
        latent = mean + log_std.exp() * noise
        normal_log_prob = -0.5 * (noise.square() + 2 * log_std + math.log(2 * math.pi))
        log_jacobian = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        return latent.tanh(), (normal_log_prob - log_jacobian).sum(-1)

    def mode(self, states: torch.Tensor) -> torch.Tensor:
        # Deterministic evaluation convention: tanh of the Gaussian location.
        return self.parameters_at(states)[0].tanh()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.network = mlp(state_dim + action_dim, hidden_size, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([states, actions], -1)).squeeze(-1)


@dataclass(frozen=True)
class SACBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminals: torch.Tensor


def bellman_target(
    rewards, terminals, next_q, next_log_prob, *, discount, alpha, backup_entropy
):
    continuation = next_q - alpha * next_log_prob if backup_entropy else next_q
    return (
        rewards
        + discount
        * torch.where(terminals, torch.zeros_like(continuation), continuation)
    ).detach()


def conservative_penalty(q_samples, log_densities, q_logged, temperature):
    """Source-style importance-corrected logsumexp minus logged-action Q.

    Equal counts of uniform, current-policy, and next-policy proposals. All
    proposals are evaluated at CURRENT states. Densities must be detached.
    The unnormalized logsumexp convention is retained (not log-mean-exp).
    """
    return (
        temperature
        * torch.logsumexp(
            (q_samples - log_densities.detach()) / temperature,
            dim=1,
        )
        - q_logged
    ).mean()


def polyak_update(source: nn.Module, target: nn.Module, rate: float) -> None:
    with torch.no_grad():
        for parameter, target_parameter in zip(
            source.parameters(), target.parameters(), strict=True
        ):
            target_parameter.lerp_(parameter, rate)


def checked_step(loss, optimizer, parameters) -> float:
    parameters = tuple(parameters)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite PGS training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError("non-finite PGS training gradient")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise RuntimeError("non-finite PGS weights after update")
    return float(loss.detach())


class ConservativeSAC:
    """Per-run learner; critic, actor, entropy and target updates are isolated."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context: RunContext,
        generator: torch.Generator,
        *,
        hidden_size: int,
        learning_rate: float,
        discount: float,
        target_rate: float,
        cql_weight: float,
        cql_samples: int,
        cql_temperature: float,
        initial_alpha: float,
        automatic_entropy: bool,
        backup_entropy: bool,
    ):
        seed = int(
            torch.randint(
                0, 2**31 - 1, (), device=context.device, generator=generator
            ).item()
        )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.actor = GaussianActor(state_dim, action_dim, hidden_size).to(
                context.device, context.dtype
            )
            self.critics = nn.ModuleList(
                [Critic(state_dim, action_dim, hidden_size) for _ in range(2)]
            ).to(context.device, context.dtype)
        self.targets = copy.deepcopy(self.critics).requires_grad_(False).eval()
        self.log_alpha = nn.Parameter(
            torch.tensor(
                math.log(initial_alpha), device=context.device, dtype=context.dtype
            )
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(), lr=learning_rate
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        self.action_dim, self.generator = action_dim, generator
        self.discount, self.target_rate = discount, target_rate
        self.cql_weight, self.cql_samples = cql_weight, cql_samples
        self.cql_temperature = cql_temperature
        self.automatic_entropy, self.backup_entropy = automatic_entropy, backup_entropy
        self.target_entropy = -float(action_dim)

    @property
    def alpha(self):
        value = self.log_alpha.detach().exp()
        if not torch.isfinite(value):
            raise RuntimeError("non-finite PGS entropy coefficient")
        return value

    def critic_loss(self, batch: SACBatch) -> tuple[torch.Tensor, dict]:
        count, proposals = len(batch.states), self.cql_samples
        repeated = batch.states.repeat_interleave(proposals, dim=0)
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(
                batch.next_states, self.generator
            )
            next_q = torch.minimum(
                *(q(batch.next_states, next_actions) for q in self.targets)
            )
            target = bellman_target(
                batch.rewards,
                batch.terminals,
                next_q,
                next_log_prob,
                discount=self.discount,
                alpha=self.alpha,
                backup_entropy=self.backup_entropy,
            )
            uniform = (
                2
                * torch.rand(
                    (count, proposals, self.action_dim),
                    device=batch.states.device,
                    dtype=batch.states.dtype,
                    generator=self.generator,
                )
                - 1
            )
            current_actions, current_density = self.actor.sample(
                repeated, self.generator
            )
            next_proposals, next_density = self.actor.sample(
                batch.next_states.repeat_interleave(proposals, 0), self.generator
            )
            actions = torch.cat(
                [
                    uniform,
                    current_actions.reshape(count, proposals, -1),
                    next_proposals.reshape(count, proposals, -1),
                ],
                dim=1,
            )
            densities = torch.cat(
                [
                    batch.states.new_full(
                        (count, proposals), -self.action_dim * math.log(2)
                    ),
                    current_density.reshape(count, proposals),
                    next_density.reshape(count, proposals),
                ],
                dim=1,
            )
        td_losses, penalties = [], []
        for critic in self.critics:
            q_logged = critic(batch.states, batch.actions)
            q_samples = critic(
                batch.states.repeat_interleave(3 * proposals, 0),
                actions.reshape(-1, self.action_dim),
            ).reshape(count, 3 * proposals)
            td_losses.append(F.mse_loss(q_logged, target))
            penalties.append(
                conservative_penalty(
                    q_samples, densities, q_logged, self.cql_temperature
                )
            )
        td, penalty = sum(td_losses), sum(penalties)
        return td + self.cql_weight * penalty, {
            "td_loss": float(td.detach()),
            "cql_penalty": float(penalty.detach()),
        }

    def update(self, batch: SACBatch) -> dict:
        if batch.states.ndim != 2 or not len(batch.states):
            raise ValueError("PGS replay states must be a non-empty matrix")
        if (
            batch.rewards.shape != (len(batch.states),)
            or batch.terminals.shape != batch.rewards.shape
            or batch.terminals.dtype != torch.bool
        ):
            raise ValueError(
                "PGS replay requires vector rewards and boolean terminal flags"
            )
        if batch.states.shape != batch.next_states.shape or batch.actions.shape != (
            len(batch.states),
            self.action_dim,
        ):
            raise ValueError("incompatible PGS replay shapes")
        if not all(
            torch.isfinite(x).all()
            for x in (batch.states, batch.next_states, batch.actions, batch.rewards)
        ):
            raise ValueError("PGS replay must be finite")
        if (batch.actions.abs() > 1).any():
            raise ValueError("PGS replay actions must lie in [-1, 1]")
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.alpha_optimizer.zero_grad(set_to_none=True)
        critic_loss, metrics = self.critic_loss(batch)
        metrics["critic_loss"] = checked_step(
            critic_loss, self.critic_optimizer, self.critics.parameters()
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critics.requires_grad_(False)
        try:
            actions, log_prob = self.actor.sample(batch.states, self.generator)
            q_value = torch.minimum(*(q(batch.states, actions) for q in self.critics))
            metrics["actor_loss"] = checked_step(
                (self.alpha * log_prob - q_value).mean(),
                self.actor_optimizer,
                self.actor.parameters(),
            )
        finally:
            self.critics.requires_grad_(True)
        entropy_loss = -(
            self.log_alpha * (log_prob.detach() + self.target_entropy)
        ).mean()
        metrics["alpha_loss"] = (
            checked_step(entropy_loss, self.alpha_optimizer, [self.log_alpha])
            if self.automatic_entropy
            else 0.0
        )
        polyak_update(self.critics, self.targets, self.target_rate)
        metrics["alpha"] = float(self.alpha)
        return metrics
