# GPyTorch v2: one Colab workflow, one pilot review

Open [LLMDM_GPyTorch_Colab.ipynb](../notebooks/LLMDM_GPyTorch_Colab.ipynb) in a
**fresh GPU runtime**, then choose **Run all**. The notebook is a new release;
it neither reuses nor overwrites results from `llmdm_forward_v1`.

The default main experiment runs all 19 frozen methods. Setup, exact checkouts,
dependency installation, dataset verification, verified restore and the full
seed-0 pilot queue run automatically. Google Drive authorization is interactive.
After reviewing the combined pilot cost, memory and training diagnostics, type
the displayed `RUN FORMAL <scope hash>` once. The notebook then runs all eight
formal seeds (38–45) for every selected method, without method/seed edits.
Any other answer leaves formal training unstarted. Importing the helper script
does not start installation, authentication or training.

## Scope and fixed settings

- Main: multi-scale logged data → target 1B / 19500 steps.
- StackExchange cross entropy is converted to maximization utility `-loss`.
- Shared, per-scale 0–40 percentile visible rows: 454 logged / 184 visible.
- Fixed-1B uses the exact 26-row subset of the main visible data, not a new split.
- 128 candidates; the original frozen per-method training budgets and precision
  remain in `experiments/llmdm_forward_gpytorch_v2/plan.json`.
- GP methods `ga_on_gp` and `bo_qei` use GPyTorch; BDI remains the independently
  labelled kernel-ridge adaptation.

The default queue contains **19 pilots + 152 formal jobs**. To include the
ablation, check `INCLUDE_FIXED_1B` before starting. Both settings then receive
their own 19 pilots and 152 formal jobs; confirmation covers exactly the
displayed methods/settings and frozen plan. A main-only pilot does not approve
the ablation. A successful exit or finite final diagnostic is not by itself a
human review or proof of convergence. Pilot reports do not expose oracle scores
for choosing budgets, methods or seeds.

## Version and environment integrity

Code and frozen assets have separate full commit pins; the upstream data-recipes
checkout is also pinned. Release metadata lists SHA256 hashes for the frozen plan,
visible/reference arrays, data manifest and exact runtime dependency constraints.
Every required artifact is checked before setup proceeds. Loaded modules from an
old checkout block execution instead of being reloaded in place.

The installer applies the frozen GP dependency constraints while pinning the
already installed Colab CUDA Torch version. It does not install the original
data-recipes environment. Editable-import refresh is handled explicitly after a
successful install. If an already loaded library was changed, restart Python and
run all again; continuing with stale native modules is not supported. `pip check`
failures block training. Only the exact known Colab failure “IPython requires Jedi,
which is not installed” receives a targeted Jedi installation and another check.

The first prepared v2 state records Python, GPU, CUDA/cuDNN and method-related
package versions, including GPyTorch and linear_operator. Each job checks this
contract as well as the existing per-method environment contract. A different
GPU or software environment stops recovery for explicit investigation; this
notebook has no ignore-environment switch. Preserving Colab's Torch means this
release is not a universal fixed Colab image: the first run records the actual
image and subsequent runs must match it.

## Backups, resume and failures

Results run on the VM's local disk at
`/content/llmdm_gpytorch_v2_state/<plan_id>/`. Drive backups are isolated at
`MyDrive/llm_design_bench/llmdm_forward_gpytorch_v2/<plan_id>/`. Drive must actually
be mounted; merely creating a `MyDrive` directory is not sufficient. The old v1
paths are not consulted. Only one notebook may own a state/backup pair.

The existing verified single-job runner is reused: one subprocess per seed,
append-only dispatch records, periodic checksum-verified snapshots, full artifact
validation before skipping, and no automatic retries. Re-running after normal
completion skips matching verified jobs. VM loss requires a verified restore;
an interrupted job remains blocked for inspection. Snapshots restore completed
results, **not in-memory training progress**.

An algorithmic failure, bad artifact, environment mismatch or backup failure
stops the queue and is not relabelled as success. Inspect the failed seed before
any deliberately authorized infrastructure retry. Never delete failure records,
change seed/budget, or mix plans just to make the coverage table green.

Drive archives accumulate; they are not automatically pruned. Check available
space. The notebook does not keep Colab alive, schedule background jobs or promise
that the full queue fits within a single runtime. Formal coverage is complete
only when each selected method/setting has eight verified successful seeds,
zero pending and zero blocked jobs.

See [snapshot size safety](COLAB.md#snapshot-size-safety) for the 4 GiB
uncompressed-content limit and the development helper's write-time check.
This frozen notebook retains its original helper pins; that development fix
is not automatically included in its checkout.

## Implementation and tests

`scripts/colab_v2.py` handles setup checks, the v2-wide environment contract and
the pilot/review/formal workflow. It wraps the unchanged `BatchRunner` and
`run_job` implementation rather than duplicating training or backup machinery.
Tests use mocked jobs and temporary artifacts: they check pinned releases,
path traversal, installation guards, one-review scope, formal gating and resume
coverage without launching real training or mounting a real Google Drive.
