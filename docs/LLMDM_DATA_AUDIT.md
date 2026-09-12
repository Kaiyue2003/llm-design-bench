# LLM-DM real-data audit: scale-stratified v1

This audit records the real data and simulator assets prepared for the agreed
[LLM-DM protocol](LLMDM_PROTOCOL.md). It is a data/provenance audit, not a new
optimization result. The historical files in `reference_results/publication/`
are preserved unchanged.

## Pinned source and bundle identity

- Upstream repository: [namkoong-lab/data-recipes](https://github.com/namkoong-lab/data-recipes).
- Verified upstream revision: `37269969a0957448d51622e0c083977bc5d260e8`.
- Frozen data manifest ID: `a27d497d18b2a476cbcc5d09651ba6e98a6618c78e1d22569c02b0de8858f8bc`.
- Prepared local bundle: `results/llmdm_shared_v1_pinned/`.
- Experiment bundle location: `experiments/llmdm_forward_v1/data/`; copying the
  three bundle files does not change their identity.
- `manifest.json` SHA256: `8ace01695e662db3ef42a080d6cc6b54435af9ba417f01baef43347f770c30d7`.

The complete identity includes the data file, upstream Python source inventory,
checkpoint and sidecar hashes, split rules, ordered row IDs, and array contents.
The manifest records the actual pinned revision, not the upstream `main` branch.
Only this manifest ID is intended for the first-batch experiment.

A superseded local preparation under `results/llmdm_shared_v1/` recorded a null
upstream revision because Windows Git rejected the sandbox user's ownership of
the new checkout. It is marked `DO_NOT_USE.md`. Preparation was repeated using
an exact, process-only Git safe-directory setting; no global Git setting or
upstream source file was changed. The superseded bundle must not be used to
freeze plans or run experiments.

## Observation extraction and split

The trusted upstream `results/data_mixing_runs.pkl` contains **472 source rows**.
Exactly **18 rows have empty histories** and are skipped. No nonempty history
is missing the selected StackExchange metric column. This leaves **454 usable
logged observations**, explaining the earlier 472-versus-454 count difference
for this pinned data file and adapter.

Each usable observation takes the **last row of its history** (`history.iloc[-1]`),
not every checkpoint and not the best checkpoint. Its `_step` remains an input
condition. The original objective is
`eval/RedPajamaStackExchange/CrossEntropyLoss` (metric index 4), and the optimizer
receives `utility = -loss`, so higher utility is better.

The five design dimensions are fixed in this order: Wikipedia, StackExchange,
GitHub, ArXiv, Book. They are nonnegative mixture proportions summing to one.

Within each model scale, the visible split selects utility percentiles 0-40
using linear interpolation, inclusive endpoints, and all boundary ties. Rows
are then merged in original source-position order. Counts are therefore not
required to equal exactly 40% of every finite group.

| Model scale | Usable logged | Visible | Hidden |
| --- | ---: | ---: | ---: |
| 20M | 112 | 45 | 67 |
| 60M | 71 | 29 | 42 |
| 150M | 51 | 21 | 30 |
| 300M | 64 | 26 | 38 |
| 500M | 39 | 16 | 23 |
| 700M | 52 | 21 | 31 |
| 1B | 65 | 26 | 39 |
| Total | **454** | **184** | **270** |

The target fidelity remains 1B parameters and 19,500 training steps. Of the
logged observations, **42** have exactly that context; **9** of those are in
the visible set. The main visible feature matrix has shape `(184, 7)`:
five mixture dimensions plus model scale and training steps.

The fixed-1B ablation is the **26-row 1B subset of the main visible set** and
has feature shape `(26, 7)`. It is not restricted to 19,500-step observations,
and it is not independently re-split. Exact equality of its designs, context,
utility and ordered row IDs to the corresponding main subset was verified.
Its original zero-based source row IDs are:

```text
54, 61, 82, 83, 87, 88, 117, 122, 192, 210, 211, 212, 349,
362, 363, 365, 380, 400, 402, 403, 424, 425, 426, 429, 437, 438
```

This is a hidden-**observation** split, not a promise that every hidden mixture
is unseen at all other model scales or training steps. A mixture can occur in
both visible and hidden observations at different fidelities.

## Real asset checksums

The pinned upstream loader selects
`opt_algos/data_models/20250119_174726_j8mad2i5/checkpoints/checkpoint_latest.pt`.
It does **not** select the sibling `checkpoint_epoch_24.pt`. The selected
checkpoint and its configuration/feature-mask sidecars are explicitly declared
in the frozen manifest.

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| `results/data_mixing_runs.pkl` | 8,561,549 | `ef90f34fb544b9d0e15aaac41bae8867fc52467e5fb42330788e6705147fafac` |
| Selected `checkpoint_latest.pt` | 135,518 | `022d0327d3872e2ff70e41877d3bb26a03f7e73ab7381830a39b1db050f14e66` |
| Selected model `config.json` | 474 | `027a3c1f01a2bb6a01648c8f17e110bfe1f9896783b32f5520b95e1109985216` |
| Selected model `feature_mask.json` | 202 | `2c9d102fedc4c3a24efc9fa47d3d062f81f4dd3900c8450e4b17148c94c9dd74` |
| Frozen `visible.npz` | 9,295 | `021b0fba9bcc5f2dbfb5f51cdec647f9ab37d3cad7ff0da8241dbf1685a913d7` |
| Frozen `reference.npz` | 21,229 | `031f8001f709ad7aef85962c035aba541d0123d83ba999a3275041bf0c758599` |

Pickle and checkpoint files are trusted executable inputs, not safe formats for
arbitrary uploads. The downstream frozen NumPy files are loaded with
`allow_pickle=False`. Optimizers receive only the visible data/`OfflineProblem`;
the reference file and evaluation metadata remain evaluator-owned.

## Checks performed, and what has not run

Data preparation read the trusted logged pickle and hashed the declared assets.
It did **not** deserialize the simulator checkpoint, construct the simulator,
or query the oracle. The exact fixed-1B subset checks also passed with
`DataRecipesTask._get_benchmark` and `torch.load` patched to raise if called.

After the real assets became available, a separate full regression test run
reported **512 passed, 5 skipped**. This included the existing
`test_data_recipes_predict_smoke` health check: it queries one fixed uniform
mixture `[0.2, 0.2, 0.2, 0.2, 0.2]` at 1B/19,500 steps and checks shape,
finiteness and the negative-loss direction. Thus the overall verification did
include one real simulator health query, even though preparation itself did
not. That fixed query was not generated by an optimization method, used for
budget/checkpoint tuning, or reported as a scientific benchmark result.

No method-training campaign, full seed-0 pilot, or eight-seed formal experiment
is represented by this audit. Those runs remain separate actions using the
frozen method plan and result-saving protocol.

## Relationship to the historical publication table

The historical publication-v1 LLM-DM experiment used **global** utility
percentiles 0-40 and exposed **182** rows. The new agreed per-scale split exposes
**184** rows with different membership. The difference is a protocol change,
not two extra training results or a performance improvement.

Do not place old publication-v1 COM/BDI/control scores alongside new method
scores as if they shared this experiment setting. Rerun the comparison methods
under the new manifest and frozen budgets. The original
`reference_results/publication/` data and [historical audit](RESULTS_AUDIT.md)
remain intact; this document supplies the later real-file count verification.
