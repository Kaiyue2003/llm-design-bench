# Agreed LLM-DM protocol: scale-stratified v1

This is the shared protocol for the new LLM data-mixture experiments. It does
not change the frozen `reference_results/publication` table or prescribe another
member's method-specific training budgets. Synthetic tasks are out of scope.

## Shared settings

| Item | Frozen rule |
| --- | --- |
| Main experiment | All available model scales' visible logged rows -> recommend a target-1B mixture |
| Objective | `eval/RedPajamaStackExchange/CrossEntropyLoss`, metric index 4 |
| Direction | `utility = -loss`; maximize utility |
| Split | Utility percentiles 0-40 **within each model scale**, then merge in original source order |
| Percentile convention | NumPy linear interpolation, inclusive endpoints and all boundary ties; not necessarily exactly 40% of rows |
| Logged observation | Last history row, matching the adapter's existing convention; training steps remain context |
| Domain order | Wikipedia, StackExchange, GitHub, ArXiv, Book; verified against upstream feature names |
| Shared data | One content-hashed visible row list and float64 arrays; identical across methods and method seeds |
| Target | Model scale 1000 (1B), 19,500 training steps |
| fixed-1B ablation | Only the 1B rows of the main visible set; no new split |
| Candidates | Exactly 128 per method/seed; duplicates allowed, unique count recorded after rounding components to 12 decimals |
| Precision | BO-qEI, GA on GP and BDI: float64; other methods: float32; no mixed precision |
| Pilot | Seed 0, **same full method configuration** and K=128; assess cost, numerical stability and output completeness |
| Formal seeds | 38-45 inclusive; eight runs; no dataset resampling |

The domain-order source is the official
[data-recipes adapter input definition](https://github.com/namkoong-lab/data-recipes/blob/main/opt_algos/benchmarks.py).
The frozen manifest verifies the actual source being used, not just this link.
Rows with tied utilities can increase the selected fraction; exact group counts
can be recovered from the frozen arrays. This is a hidden-*observation*/utility split, not a guarantee that every
hidden target-fidelity mixture is absent at other fidelities.

## Evaluation and provenance

Methods receive only `OfflineProblem` and `RunContext`. They cannot receive a
task oracle, hidden outcomes or the reference range through this interface.
Training normalization uses visible data only. The evaluator saves the final
candidate batch **before** calling the oracle at target fidelity. Do not use
oracle scores, including pilot scores, to choose training budgets or checkpoints.

The main score is each seed's maximum reference-normalized utility:

```text
(candidate utility - full logged minimum utility)
/ (full logged maximum utility - full logged minimum utility)
```

Both main and fixed-1B use the same full **multiscale** reference, not a separate
1B reference. Scores above 1 are allowed. A degenerate reference range follows
the evaluator's explicit zero-score convention. Also save median utility/score,
raw loss, unique candidates and runtime. `D(best)` is observed at the original
logged fidelity; use target-reevaluated Best Logged for a same-target comparison.

Reports retain per-seed outcomes, mean, sample standard deviation (`ddof=1`),
standard error, and success/failure/missing counts. Incomplete formal methods
remain visible but are not eligible for formal ranking. The pilot is never a
formal seed or ranking entry. Deterministic baselines may have zero variation;
the eight executions do not create independent training randomness for them.

Freeze the logged pickle checksum, ordered row IDs, source fingerprint, oracle
checkpoint and sidecar checksums, and method plan before running. The method
plan expands all constructor defaults and fingerprints installed Python source.
Each attempt also records its environment/hardware and exact dtype/device.
Different methods' update counts are not equal-compute budgets; interpret timing
with hardware information and report adaptations under their registered labels.

## Workflow

The formal entry point is `llm-design-bench-llmdm`, also available as
`llm-design-bench`, or `python -m llm_design_bench.llmdm_cli` after installation.
The generic `llm-design-bench-suite` command is for current-method development
and synthetic tasks; it is not a substitute for this frozen protocol. Historical
execution commands are archived in Git, not installed by the current package.

### 1. Prepare the shared bundle once

Only use a trusted data-recipes checkout: its logged `.pkl` and oracle `.pt`
files are trusted inputs, not safe formats for untrusted downloads. Preparation
reads the logged pickle but does **not** load the checkpoint or run the oracle.

```bash
python -m llm_design_bench.llmdm_cli prepare \
  --data-recipes-root ../data-recipes \
  --oracle-checkpoint ../data-recipes/opt_algos/data_models/20250119_174726_j8mad2i5/checkpoints/checkpoint_latest.pt \
  --output results/llmdm_shared_v1
```

Use the actual checkpoint of the agreed upstream version. The adapter verifies
the checkpoint path loaded at evaluation and rejects undeclared files. Known
configuration/feature-mask sidecars are hashed too. Changed data, source or
oracle files invalidate the bundle.

The bundle separates `visible.npz` (optimizer-visible rows and arrays) from
`reference.npz` and `manifest.json` (evaluator-owned reference and provenance).
Give another member's optimizer the visible data/`OfflineProblem`, **not** the
reference file. This is an API/data-discipline boundary, not an operating-system
security sandbox against a malicious method.

### 2. Freeze the selected methods' own budgets

Start from `configs/llmdm_methods.example.json`. It intentionally contains only
the three non-training controls. Add the methods and explicit parameter
overrides owned by each member. For example, an empty `kwargs` object explicitly
chooses that method's current defaults; it is not a claim that these are the
paper's budgets. All defaults are expanded into the saved plan.

```bash
python -m llm_design_bench.llmdm_cli freeze \
  --data-recipes-root ../data-recipes \
  --data-bundle results/llmdm_shared_v1 \
  --methods-file configs/llmdm_methods.example.json \
  --experiment-id llmdm_v1 \
  --output results/llmdm_plan_v1.json
```

Keep the shared bundle and plan with the archived results. Distribute the same
source revision to all runners. Changing source or method settings requires a
new plan and pilot, never editing an old successful run's configuration.

Each member chooses their own method budgets, but collect the selected settings
into **one shared frozen plan** for this run. Execute subsets using `--run-id`.
The current report merger intentionally rejects independently frozen plans or
different source snapshots; it does not assume they are comparable automatically.

### 3. Run one complete seed-0 pilot

**The following commands train/evaluate; preparing the bundle and plan does not.**
Select methods with repeated `--run-id` options, or omit them to run all methods
in the plan. `--setting both` runs main and ablation; omit it for main only.

```bash
python -m llm_design_bench.llmdm_cli run \
  --data-recipes-root ../data-recipes \
  --data-bundle results/llmdm_shared_v1 \
  --plan results/llmdm_plan_v1.json \
  --phase pilot --setting both --device cpu \
  --results-dir results/llmdm_v1_pilot
```

On a GPU machine, choose `--device cuda`; the oracle remains on CPU unless
`--oracle-device` is explicitly changed. Check cost, memory availability,
numerical stability and artifact completeness before proceeding. Pilot success
is necessary, but does not replace this human resource review.

### 4. Run eight formal seeds

```bash
python -m llm_design_bench.llmdm_cli run \
  --data-recipes-root ../data-recipes \
  --data-bundle results/llmdm_shared_v1 \
  --plan results/llmdm_plan_v1.json \
  --phase formal --setting both --device cpu \
  --pilot-results results/llmdm_v1_pilot \
  --results-dir results/llmdm_v1_formal
```

The runner requires a successful same-plan seed-0 pilot for each selected
method/setting. Formal seeds can be split across executions using repeated
`--seed` options, but expected coverage stays 38-45. Do not combine pilot and
formal output directories. Partial results are incomplete, not an eight-seed
estimate. Run shards sequentially against a given results directory.

## Saved results and retries

Each method/task/seed has an immutable attempt directory containing configuration
and provenance, `candidates.npz` saved before evaluation, `evaluation.npz` with
per-candidate utilities/scores/loss where applicable, and `result.json` including
status and error details. Aggregate CSV/Markdown/LaTeX reports are derived views;
attempt records are retained when the report is updated.

An interrupted/failed attempt is never silently overwritten. Use `--resume` to
reuse an already successful identical run without new training or oracle calls.
An infrastructure interruption may be retried with an explicit
`--infrastructure-retry-reason`, keeping the same seed/config and earlier attempt
records. Do not relabel numerical/algorithmic failure as infrastructure failure
or replace failed seeds with easier ones. Changing budgets creates a new plan;
do not mix its results into the old plan.

## Current implementation versus execution

The protocol can be tested with fixtures without genuine LLM data. Those tests
are not simulator benchmark results. A real shared manifest, real pilots and the
eight-seed experiment require the group's actual data-recipes data/checkpoint.
The historical publication-v1 table used a different split/budget and must not
be spliced into the new comparison; rerun controls and existing methods here.
