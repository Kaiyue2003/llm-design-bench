# Standard Baseline Source Audit

This audit covers the `standard_ga`, `cma_es`, and `reinforce` method IDs. It
separates source-backed behavior from changes required by this benchmark.

## Sources and reproducibility boundary

- Paper: [SPADE, arXiv:2605.11246](https://arxiv.org/abs/2605.11246).
- Baseline code: [Design-Baselines](https://github.com/brandontrabucco/design-baselines),
  audited at commit `785dbcfa58107bfcc426257a1c2e69d7f71c3c27` under the MIT license.
- SPADE code: [HarryYoung2018/spade](https://github.com/HarryYoung2018/spade).

The SPADE paper identifies the baseline mechanisms and evaluates `K=128`
candidates over eight seeds. The public SPADE repository states that final
task-specific configurations, preprocessing, and full evaluation scripts are
not included. In particular, there is no public final LLM-DM configuration
that can establish numeric parity with the paper table. These implementations
must therefore be reported as adaptations, not exact reproductions.

## Retained source behavior

| Method ID | Retained mechanism | Audited common configuration |
| --- | --- | --- |
| `standard_ga` | Probabilistic neural forward model; use its mean as a deterministic objective; start from high-utility logs; plain gradient ascent | width 2048, two hidden layers, 100 surrogate epochs, 200 search steps, base search rate 0.01 |
| `cma_es` | Five bootstrapped probabilistic neural surrogates; optimize ensemble mean from high-utility logs with independent CMA-ES runs | width 256, one hidden layer, 100 epochs, 100 CMA generations, initial sigma 0.5 |
| `reinforce` | Same probabilistic ensemble; initialize one Gaussian policy from high-utility logs; fixed exploration variance; standardized surrogate reward | width 256, one hidden layer, 100 epochs, 200 iterations, batch size 2048, exploration std 0.1 |

These common continuous-task values are exposed as constructor defaults where
the source is consistent. Source configurations still vary by task, and no
final LLM-DM values are public. Smoke tests use explicitly smaller values and
do not define publication settings.

## Benchmark-specific adaptations

All three methods receive only `OfflineProblem`; none can access the oracle.
Their surrogates consume design coordinates together with logged model scale
and training steps, then score candidates at the target fidelity. The
benchmark maps unconstrained search variables through softmax for simplex
tasks and scaled sigmoid for box tasks.

Further implementation changes are explicit:

- TensorFlow models were replaced with deterministic, seed-isolated PyTorch
  modules.
- Standard GA uses one audited probabilistic MLP and optimizes its mean. It is
  intentionally separate from the benchmark-native MSE/Adam `offline_mlp`
  control.
- CMA-ES is implemented directly in PyTorch with full covariance rather than
  calling `pycma`, allowing shared device, dtype, and seed handling.
- REINFORCE samples in unconstrained coordinates and maps samples to a valid
  design. This is required for LLM data mixtures.
- The source's held-out validation split is not reproduced because the
  benchmark contract supplies one offline training set and candidates are
  never selected using oracle values.
- Shared defaults are used across benchmark tasks because the final LLM-DM
  task-specific hyperparameters are not public.

## Promotion criteria

A method may lose the “adaptation” label only after the missing task
configuration is published or otherwise pinned, preprocessing and stopping
rules are matched, and candidate-level or aggregate parity is demonstrated.
