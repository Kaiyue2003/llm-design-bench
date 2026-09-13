# Reference Results

These files are snapshots from the documented single-seed and eight-seed runs.
Each experiment directory records its own configuration and interpretation.

- `data_recipes/` contains the online-style baseline and offline MLP/COM/BDI
  CSV summaries and plots.
- `synthetic/` contains the full 47-task CSV, the best task per method/category
  table, and the vertical overview plot.
- `publication/` contains the eight-seed Best Logged/COM/BDI raw runs,
  aggregate summaries, explicit seed manifest, GitHub table, and Overleaf-ready
  LaTeX table for the data-mixture and selected synthetic tasks.
- [`additional_methods_integration/`](additional_methods_integration/README.md)
  contains all fourteen methods' 1,120 short-budget simulation results, score
  tables, source/configuration records, and the exact native and Docker replay
  artifacts. These two-epoch runs validate integration and reproducibility.

The source of truth for commands and caveats is
[`docs/REPRODUCING.md`](../docs/REPRODUCING.md). Small floating-point differences
across machines are expected. `run_metadata.json` records the package
configuration, library versions, and upstream `data-recipes` commit used for
these snapshots.
