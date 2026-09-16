# Data-mixture API quickstart

This example demonstrates the current `OfflineBBOMethod` API. It is **not** a
frozen pilot or formal benchmark run, and its output must not be added to a
formal result table.

## Prepare and run

From the `llm-design-bench` repository root, install the package:

```bash
python -m pip install -e .
```

Use a trusted data-recipes checkout containing `results/data_mixing_runs.pkl`
and the simulator checkpoint expected by that checkout. The pickle, checkpoint
and upstream Python code are trusted inputs; do not run untrusted downloads.
This demo does not download data or install upstream dependencies for you.

```bash
python examples/data_recipes_quickstart.py --data-recipes-root ../data-recipes
```

The default is CPU execution, seed 0 and 128 candidates. To make a smaller
demonstration, explicitly override the budget:

```bash
python examples/data_recipes_quickstart.py --data-recipes-root ../data-recipes --candidate-budget 8 --seed 7
```

If `--data-recipes-root` is omitted, the adapter uses `DATA_RECIPES_ROOT` or its
normal checkout discovery. From another working directory, pass the script's
absolute path and an explicit checkout path (or set the environment variable).
Importing the example or running it with `--help` does not read the dataset or
load the simulator.

## What the example does

1. `make_data_recipes_task_spec` reads the logs and selects the low-utility
   0–40 percentile within each model scale. The metric is StackExchange cross
   entropy, exposed as `utility = -loss`.
2. `make_method("random_search").run(OfflineProblem, RunContext)` samples the
   final candidate batch without oracle access. Only the visible problem and
   run context enter the method; no surrogate or LLM is trained by this control.
3. The evaluator converts the returned mixtures to the target fidelity,
   1B / 19,500 training steps, through `at_target_fidelity`, then calls `predict`
   once on the full batch. It does not query extra candidates or select a subset
   using oracle scores.
4. The example prints the candidate count, maximum utility and corresponding
   minimum loss. It does not write result CSVs, freeze a plan or create backups.

## Formal experiments use a different entry point

This demo constructs its data view afresh and has no frozen source/data identity,
pilot approval, eight-seed coverage or durable per-attempt artifact persistence.
Seed 0 here is simply a demonstration seed, not a validated full-budget pilot.

For comparable results, use the shared manifest and frozen plan described in
[LLMDM_PROTOCOL.md](LLMDM_PROTOCOL.md). The automated Colab workflow is documented
in [COLAB_GPYTORCH.md](COLAB_GPYTORCH.md). Existing historical plans and notebook
pins must not be changed just to run this example.
