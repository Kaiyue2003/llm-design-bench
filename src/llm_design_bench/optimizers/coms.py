import torch
from torch import nn

from llm_design_bench.optimizers.offline_utils import (
    MLPSurrogate,
    design_parameters,
    finalize_offline_trace,
    offline_data,
    parameters_to_designs,
    set_seed,
    target_features,
    unique_top_mixtures,
)


class ConservativeObjectiveModelOptimizer:
    def __init__(
        self,
        recommendations: int = 128,
        seed: int = 0,
        hidden_size: int = 128,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        alpha: float = 0.1,
        alpha_learning_rate: float = 1e-2,
        overestimation_limit: float = 0.5,
        adversarial_steps: int = 20,
        particle_steps: int = 100,
        particle_learning_rate: float = 5e-2,
        device: str = "cpu",
    ) -> None:
        self.recommendations = recommendations
        self.seed = seed
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.alpha_learning_rate = alpha_learning_rate
        self.overestimation_limit = overestimation_limit
        self.adversarial_steps = adversarial_steps
        self.particle_steps = particle_steps
        self.particle_learning_rate = particle_learning_rate
        self.device = device

    def optimize(self, task):
        set_seed(self.seed)
        data = offline_data(task, device=self.device)
        model = MLPSurrogate(data.features.shape[1], hidden_size=self.hidden_size).to(self.device)
        model_optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        log_alpha = nn.Parameter(
            torch.tensor(self.alpha, dtype=torch.float32, device=self.device).log()
        )
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=self.alpha_learning_rate)

        for _ in range(self.epochs):
            order = torch.randperm(len(data.features), device=self.device)
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                positive = data.features[indices]
                labels = data.utility[indices]
                negative = self._adversarial_features(model, positive, task)
                positive_score = model(positive)
                negative_score = model(negative)
                overestimation = negative_score - positive_score
                alpha = log_alpha.exp().clamp(max=1e6)

                alpha_loss = alpha * (self.overestimation_limit - overestimation.detach().mean())
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

                model_loss = nn.functional.mse_loss(positive_score, labels)
                model_loss = model_loss + alpha.detach() * overestimation.mean()
                model_optimizer.zero_grad()
                model_loss.backward()
                model_optimizer.step()

        parameters = nn.Parameter(
            design_parameters(
                unique_top_mixtures(task, self.recommendations),
                task,
                device=self.device,
            )
        )
        particle_optimizer = torch.optim.Adam([parameters], lr=self.particle_learning_rate)
        for _ in range(self.particle_steps):
            mixtures = parameters_to_designs(parameters, task)
            loss = -model(target_features(mixtures, task)).mean()
            particle_optimizer.zero_grad()
            loss.backward()
            particle_optimizer.step()

        return finalize_offline_trace("coms", task, parameters_to_designs(parameters, task))

    def _adversarial_features(self, model, features: torch.Tensor, task) -> torch.Tensor:
        parameters = design_parameters(features[:, : task.mixture_dim], task, device=self.device)
        parameters.requires_grad_(True)
        fidelity = features[:, task.mixture_dim :].detach()
        for _ in range(self.adversarial_steps):
            mixtures = parameters_to_designs(parameters, task)
            score = model(torch.cat([mixtures, fidelity], dim=1)).sum()
            (gradient,) = torch.autograd.grad(score, parameters)
            parameters = (parameters + self.particle_learning_rate * gradient).detach()
            parameters.requires_grad_(True)
        return torch.cat([parameters_to_designs(parameters, task), fidelity], dim=1).detach()
