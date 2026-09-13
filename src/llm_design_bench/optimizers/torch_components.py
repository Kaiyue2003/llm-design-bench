"""Seeded continuous models shared by the additional offline methods.

These are independent PyTorch implementations. Algorithm-specific choices and
departures from the upstream programs are recorded in method metadata.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from llm_design_bench.optimizers.base import MethodCapabilities, PreparedFitThenProposeMethod
from llm_design_bench.transforms import TensorStandardizer


def noise_like(x, generator):
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)


def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


def batches(count, batch_size, generator, device):
    order = torch.randperm(count, generator=generator, device=device)
    yield from order.split(batch_size)


def step_loss(loss, optimizer, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


class ContinuousTorchMethod(PreparedFitThenProposeMethod):
    capabilities = MethodCapabilities(supports_box=True, supports_simplex=True, supports_context=True)

    def __init__(self, *, epochs=100, batch_size=64, hidden_size=128,
                 learning_rate=1e-3, steps=50, validation_fraction=0.2):
        for name, value in (("epochs", epochs), ("batch_size", batch_size),
                            ("hidden_size", hidden_size), ("steps", steps)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        self.epochs, self.batch_size, self.hidden_size = int(epochs), int(batch_size), int(hidden_size)
        self.learning_rate, self.steps = float(learning_rate), int(steps)
        self.validation_fraction = float(validation_fraction)
        self._diagnostics = {}

    def optimize(self, problem, *, context, generator):
        # Module initializers also use the method seed, even outside a runner.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(context.method_seed)
            self._diagnostics = {}
            return super().optimize(problem, context=context, generator=generator)

    def prepare(self, problem):
        encoded = problem.transforms.encode_designs(problem.split.train.designs)
        self._design_scaler = TensorStandardizer.fit(encoded, minimum_scale=0.05)
        self._x = self._design_scaler.transform(encoded)
        self._c = problem.transforms.transform_context(problem.split.train.context)
        self._y = problem.train_utility.reshape(-1, 1)
        self._target_c = problem.transforms.transform_context(problem.problem.target_context[None])
        self._dimension = self._x.shape[1]
        self._diagnostics["design_transform"] = {
            "mean": self._design_scaler.mean.tolist(), "scale": self._design_scaler.scale.tolist(),
        }

    def module(self, inputs, outputs):
        return mlp(inputs, outputs, self.hidden_size).to(self._x)

    def decode(self, x, problem):
        return problem.transforms.decode_designs(self._design_scaler.inverse(x)).detach()

    def project(self, x, problem):
        physical = problem.transforms.decode_designs(self._design_scaler.inverse(x))
        return self._design_scaler.transform(problem.transforms.encode_designs(physical))

    def target_context(self, count):
        return self._target_c.expand(count, -1)

    def target_utility(self, count, margin=0.5):
        return (self._y.max() + margin).expand(count, 1)

    def initial(self, count):
        indices = self._y[:, 0].argsort(descending=True, stable=True)
        return self._x[indices.repeat(math.ceil(count / len(indices)))[:count]].clone()

    def diagnostics(self):
        return self._diagnostics


class GaussianEnsemble(nn.Module):
    def __init__(self, inputs, hidden=128, members=3):
        super().__init__()
        self.models = nn.ModuleList([mlp(inputs, 2, hidden) for _ in range(members)])

    def forward(self, features):
        outputs = torch.stack([model(features) for model in self.models])
        means = outputs[..., 0]
        variances = outputs[..., 1].clamp(-8, 6).exp()
        mean = means.mean(0)
        variance = (variances + means.square()).mean(0) - mean.square()
        return mean[:, None], variance.clamp_min(1e-6).sqrt()[:, None]

    def fit(self, features, labels, *, epochs, batch_size, lr, generator):
        final = 0.0
        # A fixed bootstrap per member, sampled only from training observations.
        for model in self.models:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            bootstrap = torch.randint(len(features), (len(features),),
                                      device=features.device, generator=generator)
            for _ in range(epochs):
                for batch in batches(len(features), batch_size, generator, features.device):
                    ids = bootstrap[batch]
                    output = model(features[ids])
                    logvar = output[:, 1:2].clamp(-8, 6)
                    loss = 0.5 * (logvar + (labels[ids] - output[:, :1]).square() * (-logvar).exp())
                    final = step_loss(loss.mean(), optimizer, model.parameters())
        self.eval()
        return final


class ConditionalVAE(nn.Module):
    def __init__(self, dimension, context_dim, latent_dim, hidden):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = mlp(dimension + context_dim, 2 * latent_dim, hidden)
        self.decoder = mlp(latent_dim + context_dim, 2 * dimension, hidden)

    def encode(self, x, c):
        mean, logvar = self.encoder(torch.cat([x, c], -1)).chunk(2, -1)
        return mean, logvar.clamp(-8, 6)

    def distribution(self, z, c):
        mean, logvar = self.decoder(torch.cat([z, c], -1)).chunk(2, -1)
        return mean, logvar.clamp(-8, 4)

    def loss(self, x, c, generator):
        mean, logvar = self.encode(x, c)
        z = mean + (0.5 * logvar).exp() * noise_like(mean, generator)
        pred, pred_logvar = self.distribution(z, c)
        reconstruction = -gaussian_log_prob(x, pred, pred_logvar)
        kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).sum(-1)
        return reconstruction + kl

    def sample(self, c, generator):
        z = torch.randn((len(c), self.latent_dim), device=c.device, dtype=c.dtype, generator=generator)
        mean, logvar = self.distribution(z, c)
        x = mean + (0.5 * logvar).exp() * noise_like(mean, generator)
        return x, z


def gaussian_log_prob(value, mean, logvar):
    return -0.5 * (math.log(2 * math.pi) + logvar + (value - mean).square() * (-logvar).exp()).sum(-1)


class Diffusion(nn.Module):
    """Cosine VP diffusion with epsilon prediction and a complete DDIM solver."""
    def __init__(self, dimension, condition_dim, hidden, timesteps=100):
        super().__init__()
        if timesteps < 2:
            raise ValueError("diffusion_steps must be at least two")
        self.dimension, self.timesteps = dimension, timesteps
        self.net = mlp(dimension + condition_dim + 16, dimension, hidden)
        t = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-5, 0.999)
        self.register_buffer("alpha_bar", (1 - beta).cumprod(0).float())

    def forward(self, x, t, condition):
        frequencies = torch.exp(torch.linspace(0, math.log(1000), 8, device=x.device, dtype=x.dtype))
        phase = (t.to(x.dtype)[:, None] + 0.5) / self.timesteps * frequencies
        return self.net(torch.cat([x, condition, phase.sin(), phase.cos()], -1))

    def q_sample(self, x, t, noise):
        a = self.alpha_bar[t, None]
        return a.sqrt() * x + (1 - a).sqrt() * noise

    def loss(self, x, condition, generator, *, return_prediction=False):
        t = torch.randint(self.timesteps, (len(x),), device=x.device, generator=generator)
        noise = noise_like(x, generator)
        noisy = self.q_sample(x, t, noise)
        pred = self(noisy, t, condition)
        losses = (pred - noise).square().mean(-1)
        if return_prediction:
            a = self.alpha_bar[t, None]
            return losses, (noisy - (1 - a).sqrt() * pred) / a.sqrt()
        return losses

    def sample(self, condition, generator, *, steps, guidance=1.0, unconditional=None,
               initial=None, start_index=None, gradient=None, project=None, clamp=8.0):
        top = self.timesteps - 1 if start_index is None else int(start_index)
        indices = torch.linspace(top, 0, min(steps, top + 1), device=condition.device).long().unique_consecutive()
        if initial is None:
            x = torch.randn((len(condition), self.dimension), device=condition.device,
                            dtype=condition.dtype, generator=generator)
        else:
            x = initial
        for i, value in enumerate(indices):
            t = value.expand(len(x))
            epsilon = self(x, t, condition)
            if unconditional is not None:
                base = self(x, t, unconditional)
                epsilon = base + guidance * (epsilon - base)
            a = self.alpha_bar[value]
            x0 = ((x - (1 - a).sqrt() * epsilon) / a.sqrt()).clamp(-clamp, clamp)
            if gradient is not None:
                x0 = x0 + gradient(x0)
            if project is not None:
                x0 = project(x0)
            # Final step reaches clean data (alpha_bar = 1), not t=0 noise.
            previous = self.alpha_bar[indices[i + 1]] if i + 1 < len(indices) else x.new_tensor(1.0)
            x = previous.sqrt() * x0 + (1 - previous).sqrt() * epsilon
        return x


def classifier_free_condition(y, c, *, generator=None, dropout=0.0, unconditional=False):
    # Explicit presence bit disambiguates a dropped label from a real zero label.
    mask = torch.ones_like(y)
    if unconditional:
        mask.zero_()
    elif dropout:
        mask = (torch.rand(y.shape, device=y.device, generator=generator) >= dropout).to(y)
    return torch.cat([y * mask, mask, c], -1)


def train_diffusion(method, model, x, y, c, generator, *, weights=None, epochs=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=method.learning_rate)
    final = 0.0
    model.train()
    for _ in range(method.epochs if epochs is None else epochs):
        for ids in batches(len(x), method.batch_size, generator, x.device):
            condition = classifier_free_condition(y[ids], c[ids], generator=generator, dropout=0.1)
            losses = model.loss(x[ids], condition, generator)
            loss = losses.mean() if weights is None else (losses * weights[ids]).mean()
            final = step_loss(loss, optimizer, model.parameters())
    model.eval()
    return final
