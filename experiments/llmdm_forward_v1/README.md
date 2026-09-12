# Frozen first-batch LLM-DM inputs

These are **experiment inputs, not training results**. They implement the agreed
per-scale 0-40 utility-percentile protocol. The historical publication table used
a different split and must not be merged with new results.

## Contents

- `data/visible.npz`: optimizer-visible row IDs, mixtures, context and utility;
  184 observations. The 26-row fixed-1B ablation is its exact 1B subset.
- `data/reference.npz`: 454 logged observations for evaluator-only reference
  scores. Never pass this file or its hidden labels to a method.
- `data/manifest.json`: source, checkpoint, sidecar and array hashes, dimension
  order and split definition.
- `plan.json`: all constructor parameters (including expanded defaults) for
  our 19 methods, numerical precision, K=128, pilot seed 0 and formal seeds 38-45.
- `release.json`: release identities, counts and exact file SHA256 checksums.

The numeric data is derived from the published
[namkoong-lab/data-recipes source](https://github.com/namkoong-lab/data-recipes/tree/37269969a0957448d51622e0c083977bc5d260e8)
at commit `37269969a0957448d51622e0c083977bc5d260e8`. No private training records
are included. The original pickle and oracle checkpoint remain in the separately
pinned upstream checkout. See the [real-data audit](../../docs/LLMDM_DATA_AUDIT.md)
for extraction rules, exclusions, attribution and asset checksums.

The plan was frozen from clean code commit
`3d09ac372b171c58b8375942694c39bc78f40948`. The later release/notebook commits
add files without changing the package source fingerprint. Loading the plan
checks that fingerprint; a checkout of a newer, modified package is not a valid
substitute. Git attributes preserve the original bytes of hashed files on both
Windows and Linux, including JSON line endings.

## Running

Follow the [Colab guide](../../docs/COLAB.md). Use the launcher's pinned checkout,
not an arbitrary `main` version, and do not re-split data or re-freeze the plan.
The launcher starts with training disabled. After verifying the environment,
select one method/setting and explicitly enable its full-budget seed-0 pilot.
Review cost, numerical stability and saved files before formal seeds 38-45.

Main experiments use multi-scale history to recommend a mixture at 1B / 19,500
steps. The fixed-1B subset is a separate ablation. Normalization used for method
training sees only visible data; `utility = -loss` is maximized throughout.

These budgets are a resource-controlled benchmark adaptation, not a claim of
paper-exact replication or equal compute across methods. BDI remains an RBF
kernel adaptation. Another member's inverse-generative methods are not assigned
budgets here; independently frozen plans cannot simply be concatenated into the
same ranking. See [method budgets](../../docs/LLMDM_METHOD_BUDGETS.md).

Changing data, package sources or method parameters requires a new version and
new pilots; never overwrite these frozen inputs. No full-budget real-data pilot
or formal run was performed while preparing this release.
