"""Fixed-hyperparameter GPyTorch GP for GABO's latent analytic-EI search."""

from __future__ import annotations

import math

import gpytorch
import linear_operator
import torch

from llm_design_bench.optimizers.exact_gp import _exact_settings, _ExactRBFModel


class LatentGP:
    """Preserve GABO's scalar RBF posterior on its unstandardized latent inputs.

    This shares the exact GPyTorch model/settings with the forward GP methods,
    but not their learned hyperparameters or input preprocessing. All model
    parameters are fixed: unit output variance, one median-distance lengthscale
    copied to every coordinate, and observation variance equal to ``ridge``.
    ``predict`` returns latent-function uncertainty, without observation noise.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> None:
        _validate_training_tensors(x, y)
        if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be finite and positive")
        self.x = x.detach().clone()
        targets = y.detach().clone()
        self.mean = targets.mean()
        self.scale = targets.std(unbiased=False).clamp_min(0.1)
        distances = torch.pdist(self.x)
        positive = distances[distances > 0]
        self.lengthscale = (
            positive.median().clamp_min(0.1)
            if len(positive)
            else self.x.new_tensor(1.0)
        )
        self.ridge = float(ridge)
        standardized_targets = (targets - self.mean) / self.scale

        # The diagonal ridge is the entire training noise term; no learned
        # noise, extra stabilization, or candidate observation noise is added.
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Positive(
                transform=torch.exp, inv_transform=torch.log
            )
        ).to(device=x.device, dtype=x.dtype)
        self.model = _ExactRBFModel(self.x, standardized_targets, self.likelihood).to(
            device=x.device, dtype=x.dtype
        )
        with torch.no_grad():
            self.model.covar_module.base_kernel.raw_lengthscale.copy_(
                self.lengthscale.log().expand_as(
                    self.model.covar_module.base_kernel.raw_lengthscale
                )
            )
            self.model.covar_module.raw_outputscale.zero_()
            self.likelihood.noise_covar.raw_noise.fill_(math.log(self.ridge))
        self.model.eval().requires_grad_(False)
        self.likelihood.eval().requires_grad_(False)

    def kernel(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Retain the previous kernel helper using the shared GPyTorch kernel."""
        with _exact_settings():
            return self.model.covar_module(first, second).to_dense()

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != self.x.shape[1]:
            raise ValueError("candidate inputs must match the latent input dimension")
        if x.dtype != self.x.dtype or x.device != self.x.device:
            raise ValueError(
                "candidate inputs must use the GP training dtype and device"
            )
        if not torch.isfinite(x).all():
            raise ValueError("candidate inputs must be finite")
        with _exact_settings():
            posterior = self.model(x)
            # MultivariateNormal.variance applies GPyTorch's dtype-dependent
            # floor (1e-6 for float32). Preserve the previous 1e-8 floor instead.
            variance = posterior.lazy_covariance_matrix.diagonal(dim1=-2, dim2=-1)
            mean = posterior.mean * self.scale + self.mean
            std = variance.clamp_min(1e-8).sqrt() * self.scale
        return mean, std

    def diagnostics(self) -> dict[str, str | int | float]:
        return {
            "backend": "gpytorch",
            "gpytorch_version": gpytorch.__version__,
            "linear_operator_version": linear_operator.__version__,
            "inference": "dense_exact_cholesky",
            "posterior": "latent_function",
            "kernel": "fixed_isotropic_rbf",
            "hyperparameter_training_steps": 0,
            "input_transform": "none",
            "lengthscale": float(self.lengthscale),
            "output_scale": 1.0,
            "training_noise_variance": self.ridge,
            "additional_jitter": 0.0,
            "utility_mean": float(self.mean),
            "utility_std": float(self.scale),
            "posterior_variance_floor": 1e-8,
            "dtype": str(self.x.dtype),
        }


def _validate_training_tensors(x: torch.Tensor, y: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("GP inputs and targets must be torch tensors")
    if x.ndim != 2 or not len(x) or x.shape[1] < 1:
        raise ValueError("GP inputs must be a non-empty matrix")
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("GP targets must have one value per training input")
    if x.dtype not in (torch.float32, torch.float64) or y.dtype != x.dtype:
        raise TypeError("GP inputs and targets must share float32 or float64 dtype")
    if y.device != x.device:
        raise ValueError("GP inputs and targets must share a device")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("GP training inputs and targets must be finite")
