# Reference Results

These files are compact snapshots from the documented seed-38 runs. They are
included so that users can compare a fresh checkout with a known result without
committing every generated plot.

- `data_recipes/` contains the online-style baseline and offline MLP/COM/BDI
  CSV summaries and plots.
- `synthetic/` contains the full 47-task CSV, the best task per method/category
  table, and the vertical overview plot.

The source of truth for commands and caveats is
[`docs/REPRODUCING.md`](../docs/REPRODUCING.md). Small floating-point differences
across machines are expected. `run_metadata.json` records the package
configuration, library versions, and upstream `data-recipes` commit used for
these snapshots.
