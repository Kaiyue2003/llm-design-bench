# Adding an Offline Method

## Minimal implementation

Use `PreparedFitThenProposeMethod` when the algorithm has a training
phase and a candidate-generation phase:

```python
import torch

from llm_design_bench.optimizers import (
    ImplementationKind,
    MethodCapabilities,
    MethodFamily,
    MethodMetadata,
    PreparedFitThenProposeMethod,
    register_method,
)


@register_method("my_method")
class MyMethod(PreparedFitThenProposeMethod):
    metadata = MethodMetadata(
        method_id="my_method",
        display_name="My Method",
        family=MethodFamily.INVERSE_GENERATIVE,
        implementation_kind=ImplementationKind.FAITHFUL_PYTORCH_PORT,
        paper_url="https://...",
        source_url="https://...",
        source_commit="<exact upstream commit>",
        original_framework="TensorFlow",
        adaptations=("multi-fidelity context conditioning",),
    )
    capabilities = MethodCapabilities(
        supports_simplex=True,
        supports_box=True,
        supports_context=True,
    )

    def fit_prepared(self, problem, *, context, generator):
        x = problem.train_features
        y = problem.train_utility
        # Train only PyTorch modules with the supplied tensors and seed.
        return {"train_rows": len(x)}

    def propose_prepared(self, problem, *, context, generator):
        encoded = torch.zeros(
            context.candidate_budget,
            problem.problem.design_space.model_dimension,
            device=context.device,
            dtype=context.dtype,
        )
        return problem.transforms.decode_designs(encoded)
```

The evaluator constructs the method with
`make_method("my_method", **config)`. Do not import or call a task
oracle from a method module.

## Porting TensorFlow or JAX code

Port the algorithm, not merely the class names:

1. Pin the official repository commit and retain its license notice.
2. Record architecture, initialization, optimizer, schedules, loss reductions,
   clipping, exponential moving averages, and sampling equations.
3. Replace arrays, modules, optimizers, and random state with PyTorch
   equivalents.
4. Route every random draw through the run seed or supplied
   `torch.Generator`.
5. Preserve checkpoint tensor shapes and parameterization where possible.
6. Add component parity tests on fixed tiny tensors.
7. Add an end-to-end deterministic smoke test on simplex and box tasks.
8. Compare distributions across several seeds, not one lucky run.
9. Label material substitutions as adaptations in metadata.
10. Promote the catalog entry to the runnable registry only after these checks.

Framework conversion alone does not establish numerical equivalence. CbAS
density ratios, diffusion noise schedules, classifier-free guidance, and
probabilistic-bridge solvers each need targeted parity tests.

## Connections required by family

| Method | Shared components to implement |
| --- | --- |
| CbAS | conditional VAE, probabilistic surrogate ensemble, adaptive threshold, density-ratio weights |
| MINs | conditional inverse GAN, discriminator, target-utility search |
| DDOM | design diffusion, utility conditioning, classifier-free guidance |
| GABO | VAE latent representation, adversarial source critic, latent GP, acquisition optimizer |
| GTG | trajectory builder, conditional trajectory diffusion, return guidance |
| RGD | design diffusion, proxy guidance, diffusion-based proxy refinement |
| BONET | trajectory builder, autoregressive transformer, regret budget |
| DEMO | surrogate ascent, design diffusion prior, noise-and-denoise editor |
| ROOT | score strata, synthetic task generator, probabilistic bridge |
| SPADE | conditional score diffusion for `p(y|x)`, calibration, kNN support penalty, LCB, evolutionary search |

For data mixtures, every conditional model also receives transformed model
scale and training steps. For synthetic tasks the same context path is used
with constant context, avoiding a second method implementation.

Discrete spaces are deliberately not advertised by current capabilities.
Adding them requires a `DiscreteSpace` with an explicit encoding,
decoding, and validity policy before TFBind-style tasks can be claimed.
