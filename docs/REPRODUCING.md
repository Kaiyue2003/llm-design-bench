# Reproducing the Reference Results

The committed files under `reference_results/` are comparison artifacts, not
inputs to the benchmark. Fresh runs are always written to the ignored
`results/` directory.

## 1. Create an environment

```bash
python -m venv .venv
```

Activate the environment, then install the checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The reference environment uses Python 3.11. The CI suite also checks Python
3.12.

## 2. Synthetic benchmark

Run:

```bash
llm-design-bench-synthetic \
  --logged-samples 256 \
  --recommendations 64 \
  --epochs 100 \
  --particle-steps 100 \
  --bdi-steps 100 \
  --seed 38 \
  --results-dir results/synthetic
```

The task at index `i` receives logged-data seed `38 + i`. COM and BDI both use
optimizer seed 38. Compare `results/synthetic/synthetic_bo_results.csv` with
`reference_results/synthetic/synthetic_bo_results.csv`.

Exact floating-point values can vary slightly across PyTorch, BLAS, and
operating-system versions. Task names, row counts, columns, ranking behavior,
and scores within normal numerical tolerance should agree.

## 3. Data-mixture benchmark

Clone the exact upstream dependency and make its path available:

```bash
git clone https://github.com/namkoong-lab/data-recipes.git
```

The committed data-mixture snapshots use upstream commit
`37269969a0957448d51622e0c083977bc5d260e8`. For strict historical replication,
check out that commit and record `git -C ../data-recipes status --short` alongside
your experiment.

Run the search baselines:

```bash
llm-design-bench \
  --data-recipes-root ../data-recipes \
  --queries 256 \
  --reference-queries 2048 \
  --recommendations 128 \
  --seed 38 \
  --results-dir results/data_recipes_baselines
```

Run the offline methods:

```bash
llm-design-bench-offline \
  --data-recipes-root ../data-recipes \
  --reference-queries 2048 \
  --recommendations 128 \
  --epochs 100 \
  --particle-steps 100 \
  --bdi-steps 100 \
  --train-min-percentile 0 \
  --train-max-percentile 40 \
  --seed 38 \
  --results-dir results/data_recipes_offline
```

To reproduce the fixed-1B ablation, add `--logged-model-scale 1000` and write
to a separate output directory:

```bash
llm-design-bench-offline \
  --data-recipes-root ../data-recipes \
  --logged-model-scale 1000 \
  --reference-queries 2048 \
  --recommendations 128 \
  --epochs 100 \
  --particle-steps 100 \
  --bdi-steps 100 \
  --train-min-percentile 0 \
  --train-max-percentile 40 \
  --seed 38 \
  --results-dir results/data_recipes_1b_offline
```

The `data-recipes` task loads logged runs from
`results/data_mixing_runs.pkl` in the upstream checkout and evaluates
candidates through its simulator checkpoint. Because that checkpoint is
loaded as a trusted local artifact, use only an upstream checkout you trust.

## 4. Seeded publication benchmark

The compact publication table uses the following predeclared trial seeds:

```text
38, 39, 40, 41, 42, 43, 44, 45
```

For synthetic tasks, the logged-dataset seed and optimizer seed both equal the
trial seed. The Data Recipes logged dataset is fixed, while COM and BDI use the
trial seed. All methods return `K=128` recommendations. Results are aggregated
as mean +/- sample standard deviation (`ddof=1`), not standard error.

Run:

```bash
llm-design-bench-publication \
  --data-recipes-root ../data-recipes \
  --seed 38 --seed 39 --seed 40 --seed 41 \
  --seed 42 --seed 43 --seed 44 --seed 45 \
  --logged-samples 256 \
  --recommendations 128 \
  --epochs 100 \
  --particle-steps 100 \
  --bdi-steps 100 \
  --train-min-percentile 0 \
  --train-max-percentile 40 \
  --results-dir results/publication
```

The runner is resumable. It rejects an existing `raw_runs.csv` when its
configuration fingerprint differs, preventing accidental aggregation across
incompatible runs. Compare all files against
`reference_results/publication/`, especially `raw_runs.csv`,
`seed_manifest.csv`, and `run_metadata.json`.

The default synthetic list is fixed in
`PUBLICATION_SYNTHETIC_FUNCTIONS`. It is the union of the prior exploratory
COM and BDI winner in each category, so this compact result is descriptive and
performance-selected. Use Section 2 for an unbiased all-task sweep.

## 5. Convert or rebuild unified reports

The publication-v1 directory is immutable. Convert it into the unified schema
in a new directory with:

```bash
llm-design-bench-report from-legacy \
  --publication-dir reference_results/publication \
  --results-dir results/publication_v1_unified
```

Rebuild summaries and human-readable tables from a unified raw file with:

```bash
llm-design-bench-report from-unified \
  --input-csv results/experiment/method_seed_results.csv \
  --results-dir results/experiment_report
```

The generated summary includes both sample standard deviation and standard
error. The Markdown and LaTeX tables display standard error.

## 6. Verify the package

The unified suite now defaults to thirteen registered methods, including
`tri_mentoring`. To run that method alone at its resolved constructor defaults:

```bash
llm-design-bench-suite --function branin --method tri_mentoring \
  --candidate-budget 128 --results-dir results/tri_mentoring
```

Omitting `--seed` uses seeds 38--45. This trains small surrogate networks,
not a language model. Per-candidate mentoring with width 2048 and 200 search
steps can still be expensive. Development smoke runs should provide a smaller
configuration through `MethodSpec` (for example, width 8, two surrogate epochs,
two solver steps, and four neighbors), and must be reported separately from
full experiments. See [method notes](METHODS.md) for defaults and semantics.

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```

To test the wheel itself, create another environment and install the generated
`.whl` from `dist/`, then run `llm-design-bench-synthetic --help`.
