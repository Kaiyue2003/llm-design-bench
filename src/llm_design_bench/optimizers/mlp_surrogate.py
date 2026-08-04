import torch

from llm_design_bench.optimizers.offline_utils import (
    MLPSurrogate,
    design_parameters,
    finalize_offline_trace,
    fit_surrogate,
    offline_data,
    parameters_to_designs,
    set_seed,
    target_features,
    unique_top_mixtures,
)


class OfflineMLPOptimizer:
    def __init__(
        self,
        recommendations: int = 128,
        seed: int = 0,
        hidden_size: int = 128,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
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
        self.particle_steps = particle_steps
        self.particle_learning_rate = particle_learning_rate
        self.device = device

    def optimize(self, task):
        set_seed(self.seed)
        data = offline_data(task, device=self.device)
        model = MLPSurrogate(data.features.shape[1], hidden_size=self.hidden_size).to(self.device)
        fit_surrogate(
            model,
            data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )

        parameters = torch.nn.Parameter(
            design_parameters(
                unique_top_mixtures(task, self.recommendations),
                task,
                device=self.device,
            )
        )
        optimizer = torch.optim.Adam([parameters], lr=self.particle_learning_rate)
        for _ in range(self.particle_steps):
            mixtures = parameters_to_designs(parameters, task)
            loss = -model(target_features(mixtures, task)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return finalize_offline_trace("offline_mlp", task, parameters_to_designs(parameters, task))
