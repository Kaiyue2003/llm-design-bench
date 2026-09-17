# Reproducing the Benchmark and Reading Historical Archives

The formal LLM-DM workflow is the only recommended experiment workflow for
the agreed data-mixture comparison. Historical outputs remain available for
inspection; they are not another experiment that contributors must run and
are not automatically part of the final paper.

## Current formal LLM-DM experiment

Start with [LLMDM_PROTOCOL.md](LLMDM_PROTOCOL.md) for the shared data, utility
direction, scale-stratified split, target fidelity, method budgets, pilot gate
and eight formal seeds. The installed commands `llm-design-bench` and
`llm-design-bench-llmdm` are aliases of the same formal CLI. Unlike the retired
online command, the unqualified name now takes prepare/freeze/run subcommands.

The completed [GPyTorch v2 archive](../reference_results/llmdm_forward_gpytorch_v2/README.md)
contains 19 methods x 8 seeds for the multi-scale main experiment. It excludes
the fixed-1B ablation. Its [release metadata](../experiments/llmdm_forward_gpytorch_v2/release.json)
records:

- benchmark code: `9d70e1458239142353e34cc596ba2b8b8764871d`;
- upstream data-recipes: `37269969a0957448d51622e0c083977bc5d260e8`;
- plan ID: `1e097b0f5f554982ac5029a28b66337dfa22cfd9a309da200523202eea786b8c`.

Use the pinned [GPyTorch Colab notebook](../notebooks/LLMDM_GPyTorch_Colab.ipynb)
and [its guide](COLAB_GPYTORCH.md) to resume or reproduce that release. Keep
its code/data/plan identity and runtime policy together. Current-source
refactoring changes the package fingerprint; it must not silently replace a
release's pinned source or be used to bypass compatibility checks. A genuinely
new experiment on current code needs a new frozen plan and pilots. Existing
archived runs do not need retraining merely because source organization changed.

Results should be written to a separate directory under ignored `results/`.
Do not overwrite `reference_results/` or reuse a result directory for a
different plan. Preserve per-seed failures and the exact requested seeds.
The archive's recorded environment is the reproduction reference; matching
seeds alone does not guarantee bitwise equality across hardware and libraries.

## Development verification is not a formal experiment

Install a development checkout with `python -m pip install -e ".[dev]"` and
run the tests/build:

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```

After updating an editable installation, reinstall it to refresh entry points;
stale legacy launchers from another installation do not implement this protocol.
For an installed-wheel smoke check, install the generated wheel into a separate
environment and run `llm-design-bench --help`, `llm-design-bench-suite --help`
and `llm-design-bench-report --help`.

The synthetic tasks and generic method API remain available. For example:

```bash
llm-design-bench-suite --function branin --method random_search \
  --seed 0 --logged-samples 64 --candidate-budget 8 \
  --results-dir results/synthetic_smoke
```

This tests the current method/task integration, not the frozen LLM-DM protocol.
Use [METHODS.md](METHODS.md) for method behavior and the Python `MethodSpec`
interface for explicit reduced training budgets. Do not label reduced-budget
smoke output as full-budget pilot or formal benchmark results.

## Archived publication-v1 results (historical only)

The directory name `reference_results/publication/` is historical, not a promise
that its table will appear in the final paper. Its three-method table uses a
different split/budget and a synthetic subset selected from prior COM/BDI
category winners. It cannot be spliced into the current 19-method comparison
or treated as an unbiased all-47-task synthetic evaluation.

The retained [metadata](../reference_results/publication/run_metadata.json)
records code commit `8cc1f2466140648f7edad810e2ce8b5d75f5111a`, upstream commit
`37269969a0957448d51622e0c083977bc5d260e8`, and clean code/upstream working trees.
It also records the dependencies, seeds 38-45, K=128 and resolved historical
settings. Its result table reports mean +/- sample SD (`ddof=1`).

The old online, offline, publication and synthetic execution entry points have
been removed from the current package. Their code and usage instructions can
be inspected in Git history. **Any old execution command is applicable only
to the corresponding historical code revision and environment, not to this
checkout.** If a historical rerun is specifically needed, use a separate
checkout of the recorded commit and follow that version's documentation after
checking its recorded inputs/dependencies. Do not change the current working
tree or reactivate legacy commands to run the modern benchmark.

The legacy offline split computed utility percentiles globally; its 1B option
filtered to 1B before re-splitting and used a 1B normalization reference. The
current formal protocol splits within each scale, shares the visible manifest,
takes its 1B subset without a new split and retains the multi-scale reference.
These differences are experimental, not cosmetic command-line changes.

## Earlier exploratory snapshots (historical only)

The older single-seed synthetic, online search and offline result directories
remain under `reference_results/`. Their shared
[metadata](../reference_results/run_metadata.json) records an upstream commit
and environment/settings, but **does not record the benchmark code commit**.
Do not invent a matching revision or claim those files support exact
code-pinned reproduction. They remain useful as historical development
records, not as substitutes for the current frozen experiment.

The first frozen forward-method release, `llmdm_forward_v1`, also remains in
the archive. Its original pinned notebook/release files describe that distinct
experiment. Do not mix v1 outputs with GPyTorch v2 outputs.

## Read-only report rebuilding and historical conversion

The current reporting command can read historical publication-v1 inputs and
write converted tables to a new directory without rerunning any method or
modifying the archive:

```bash
llm-design-bench-report from-legacy \
  --publication-dir reference_results/publication \
  --results-dir results/publication_v1_unified
```

Rebuild a report from current unified per-seed rows with:

```bash
llm-design-bench-report from-unified \
  --input-csv results/experiment/method_seed_results.csv \
  --results-dir results/experiment_report
```

Conversion preserves historical status/provenance; it does not upgrade old
experiments to the current protocol. Generated summaries include sample SD
and SE; the generic Markdown/LaTeX reports display SE, so label uncertainty
explicitly when making a paper table. The GPyTorch v2 archived main table
reports mean +/- SD. See [RESULT_SCHEMA.md](RESULT_SCHEMA.md) for the exact
fields and conversion limitations.
