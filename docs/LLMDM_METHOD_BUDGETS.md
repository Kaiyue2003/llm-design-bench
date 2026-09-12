# LLM-DM first-batch method budgets: forward v1

`configs/llmdm_methods.forward_v1.json` contains the agreed budgets for **our 19
methods**, including controls, forward surrogates and standard optimization
baselines. It does not assign budgets to the other member's nine inverse
generative methods. It is an input to the protocol's `freeze` command, not by
itself a frozen experiment plan or evidence of training.

The common data, utility, split, target fidelity, precision, candidate budget,
seeds and reporting rules are in [LLMDM_PROTOCOL.md](LLMDM_PROTOCOL.md). Both the
multiscale main experiment and fixed-1B ablation use the budgets below. Each
method returns K=128; the full seed-0 pilot uses the same settings as formal
seeds 38-45. No mixed precision is enabled.

## Explicit method overrides

Ordinary neural widths are 128 and row batch sizes are 64. Exceptions are
RoMA's width 64, LTR's batch of 32 lists, and MATCH-OPT's preserved architecture
parameterized by embedding dimension 8 (hidden widths 128 and 32). Existing
method-specific depths, losses and learning rates are preserved unless listed
below. In particular, the CMA-ES and REINFORCE surrogates keep one hidden layer;
"width 128" does not impose one shared architecture on all methods.

| Method ID | Training budget | Search / special budget |
| --- | --- | --- |
| `best_logged` | None | Select visible logged designs, target reevaluation by evaluator |
| `random_search` | None | 128 random designs |
| `sobol` | None | 128 Sobol designs |
| `offline_mlp` | 200 epochs | 200 particle steps |
| `standard_ga` | 200 surrogate epochs | 200 solver steps; GA means **gradient ascent**, not genetic algorithm |
| `coms` | 200 epochs, 20 adversarial steps | 200 particle steps |
| `bdi` | RBF kernel-ridge fit | 200 optimization steps; **BDI adaptation**, not original NTK/JAX reproduction |
| `ga_on_gp` | 200 GP training steps | 200 gradient-ascent steps |
| `bo_qei` | 200 GP training steps | 200 acquisition steps, 128 MC samples |
| `cma_es` | Ensemble of 5, 200 epochs per model | Population 16 **per independent trajectory**, 100 generations |
| `reinforce` | Ensemble of 5, 200 epochs per model | 200 policy iterations, 512 policy samples per iteration |
| `mc_dropout` | 200 epochs, dropout probability 0.1 | 32 MC samples, 200 particle steps |
| `roma` | Width 64, 100 epochs, 5 weight-perturbation steps | 10 local adaptation steps per solver step, 100 solver steps |
| `ict` | 3 proxies, 200 epochs each, `surrogate_learning_rate=0.001` | 100 adaptation steps, 100 solver steps |
| `tri_mentoring` | 3 proxies, 200 epochs each, `surrogate_learning_rate=0.001` | 100 solver steps, 10 neighbors |
| `ltr` | 100 epochs, 256 lists per epoch, list length 32, batch 32 lists | 200 solver steps |
| `match_opt` | Embedding 8, 200 epochs, 32 requested buckets, 5 quadrature nodes | 200 solver steps |
| `pgs` | 200 surrogate epochs, 10,000 CQL/SAC updates | 50 requested solver steps; actual certified replay/horizon recorded |
| `spade` | Width 128, time embedding 32, 100 epochs, diffusion schedule 100 steps; calibration MC 4 / DDIM 10 | Acquisition MC 64 / DDIM 50, population 128, elite 64, 100 generations |

ICT and Tri-Mentoring explicitly override the source-derived constructor
learning rate of 0.1 with 0.001. Their other mentoring/label/weight learning
rates are separate parameters and are not changed. PGS's batch size 64 applies
to its surrogate minibatches and RL replay minibatches. Its 10,000 updates reuse
offline replay; they are not 10,000 simulator queries.

The baseline constructor defaults used when choosing these overrides were the
integrated method implementations at commit `78eae4e`. `freeze` expands **all**
remaining constructor defaults into the output plan and fingerprints the actual
installed source. The output plan, not moving source defaults, is the executable
experiment configuration. For example, SPADE retains `acq_beta=0.1` and
`support_transform=false`; PGS retains `backup_entropy=false`. These are declared
adaptation settings, not claims of matching the paper's configurations.

## Freeze without training

After preparing and verifying the real shared bundle, and committing the final
code revision, run:

```bash
python -m llm_design_bench.llmdm_cli freeze \
  --data-recipes-root ../data-recipes \
  --data-bundle results/llmdm_shared_v1 \
  --methods-file configs/llmdm_methods.forward_v1.json \
  --experiment-id llmdm_forward_v1 \
  --output results/llmdm_forward_v1_plan.json
```

This constructs configurations only: no surrogate training, candidate search or
oracle evaluation. Use the saved plan on Colab, with the same source and shared
data bundle. Do not refreeze it merely because the machine changes. Keep the
plan and results in persistent storage, and run selected method IDs sequentially
to avoid overlapping writes to one results directory.

When the inverse methods are ready for a combined comparison, collect everyone's
explicit settings in a common plan and source snapshot **before that combined
run**. The current report merger rejects different plans; independently frozen
reports must not be spliced into one ranking. This first-batch plan covers our
methods only, and does not freeze another member's future implementation.

## Pilot and interpretation

These are **resource-controlled adaptation budgets**, not equal training compute,
an exact paper reproduction, or oracle-tuned optimal hyperparameters. The shared
128 candidates bound final oracle evaluations, not internal surrogate work. For
example:

- CMA-ES has 128 independent trajectories with population 16, not 16 candidates
  total. Its five-model ensemble adds surrogate-scoring cost.
- RoMA adapts locally for each candidate; 128 candidates x 100 solver steps x 10
  adaptation steps can require 128,000 local updates, beyond initial training.
  Tri-Mentoring also maintains per-candidate adapted ensembles.
- SPADE repeatedly evaluates an MC diffusion acquisition. Its 128-member
  population with 64 draws and 50 DDIM steps is substantially more expensive than
  one MLP forward pass. With K=population=128, the final selection returns the
  whole freshly scored final population, including retained elites.
- LTR's lists reuse visible rows; their number is not an independent sample
  count. MATCH-OPT caps its bucket count within each available fidelity group.
  PGS can have a shorter certified rollout than the requested 50 steps.

Run each selected method/setting's **full** seed-0 pilot before its eight formal
seeds. Review elapsed time, memory, numerical stability, replay diagnostics and
complete candidate/result files. Colab suitability is not yet demonstrated by
constructor tests. Do not reduce a difficult method to a smaller smoke-test
budget and label it as this pilot.

Do not choose budgets, checkpoints or retries using pilot oracle scores. If
resource or numerical evidence requires a budget/source change, record the
reason, create a new plan, and rerun the relevant pilots; do not silently edit
or merge existing successful results. Keep numerical failures visible rather
than relabeling them as infrastructure interruptions.
