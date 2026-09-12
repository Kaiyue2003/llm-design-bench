# Fresh GPyTorch LLM-DM experiment

These are immutable **inputs, not training results**. This release starts every
method afresh and never reuses or overwrites `llmdm_forward_v1` results.

- Clean code commit: `9d70e1458239142353e34cc596ba2b8b8764871d`.
- Plan ID: `1e097b0f5f554982ac5029a28b66337dfa22cfd9a309da200523202eea786b8c`.
- Data manifest: `a27d497d18b2a476cbcc5d09651ba6e98a6618c78e1d22569c02b0de8858f8bc`.
- GP backend: GPyTorch 1.15.2 / linear_operator 0.6.1, exact zero-mean ARD RBF.

The data bundle is copied byte-for-byte from the verified original release.
There is no new split: 454 usable logged observations, 184 visible observations
from per-scale utility percentiles 0–40, and the exact 26-row 1B visible subset.
The five-domain order and upstream oracle/checkpoint provenance are unchanged.
Source attribution and exclusions remain in the
[data audit](../../docs/LLMDM_DATA_AUDIT.md).

`methods.json` preserves all 19 original method budgets. `plan.json` expands
their constructor defaults and freezes the new package-source fingerprint.
Only `ga_on_gp` and `bo_qei` switch GP backend; BDI remains an independent
kernel-ridge adaptation. All method adaptation labels remain appropriate.

The agreed target is 1B / 19500 steps, utility is negative StackExchange cross
entropy, K=128, pilot seed=0, and formal seeds=38–45. GP methods and BDI use
float64; neural methods use float32 without mixed precision. Main and fixed-1B
are separate settings with separately verified pilots.

Use [the automated notebook](../../notebooks/LLMDM_GPyTorch_Colab.ipynb) and
[its guide](../../docs/COLAB_GPYTORCH.md) in a fresh Colab GPU runtime.
Run all prepares the environment and runs all selected pilots; after one
explicit review it queues all eight formal seeds per method. Drive authorization
is still interactive. Completed results are verified before reuse **within this
new plan only**. Failures stop the queue, without automatic retries or tuning.

`runtime-constraints.txt` pins the GP libraries. Colab's installed CUDA Torch is
preserved, while the first runtime's Python/GPU/numerical-library contract is
recorded and checked on resume. This does not claim a universal fixed Colab
image. Every attempt also records its complete installed-package inventory.

`release.json` lists the byte hashes of all experiment inputs. Code and asset
commits are pinned separately by the notebook. Never edit these inputs or replace
their hashes to admit another version. Later release/notebook commits do not
change the frozen code. The historical publication table and v1 results must
not be mixed into this new ranking.

Preparation passed numerical unit/regression tests and small real-data checks.
No full-budget pilot, formal experiment, or real Colab/Drive run was executed
while freezing these files. Those checks run in the notebook on the actual
allocated runtime.
