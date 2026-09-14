# Reference Results

These directories contain separately versioned result snapshots. They are
included so that users can inspect known results without committing every
generated plot. Check each snapshot's protocol and provenance before comparing
results; the experiments below do not share one interchangeable ranking.

- `data_recipes/` contains the online-style baseline and offline MLP/COM/BDI
  CSV summaries and plots.
- `synthetic/` contains the full 47-task CSV, the best task per method/category
  table, and the vertical overview plot.
- `publication/` contains the eight-seed Best Logged/COM/BDI raw runs,
  aggregate summaries, explicit seed manifest, GitHub table, and Overleaf-ready
  LaTeX table for the data-mixture and selected synthetic tasks.
- [`llmdm_forward_gpytorch_v2/`](llmdm_forward_gpytorch_v2/README.md) contains
  the formal multi-scale data-mixture results from the frozen GPyTorch v2 plan:
  19 methods x 8 seeds (38-45), original per-seed and summary CSVs, a mean +/- SD
  table and snapshot-specific metadata. Only the main setting is included;
  fixed-1B and full raw artifact archives are not included. CSV consistency
  verification is documented separately from full artifact verification.

For the earlier snapshots, commands and caveats are in
[`docs/REPRODUCING.md`](../docs/REPRODUCING.md). The root `run_metadata.json`
describes the seed-38 `data_recipes/` and `synthetic/` snapshots;
`publication/` retains its own metadata. GPyTorch v2 likewise has its own
README and `run_metadata.json` in the linked directory, with links to the
frozen inputs and Colab guide. Small floating-point differences across
machines are expected.
