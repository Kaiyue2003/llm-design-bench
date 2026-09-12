# First-batch LLM-DM experiments on Colab

Use `notebooks/LLMDM_Colab.ipynb`. This notebook is a launcher for the frozen
first-batch release, not a second implementation of data splitting or methods.
Its repository checkout is pinned to a full commit, and its data-recipes checkout
to `37269969a0957448d51622e0c083977bc5d260e8`. The saved plan also checks installed
Python source fingerprints and the data/oracle file hashes.

## Before running

1. Open the notebook in Colab and choose a GPU runtime for neural methods.
2. Run checkout, installation and data/plan verification cells. They do not train
   or initialize the oracle. Do not run the upstream `setup_env.sh` or install its
   old Python/PyTorch environment. Installation constrains the existing Colab
   PyTorch version instead of replacing its CUDA build.
3. Mount Drive yourself in the notebook. Review the notebook before granting
   access. The only backup location it uses is
   `MyDrive/llm_design_bench/llmdm_forward_v1/<plan_id>/`.
4. Choose one method, one setting and one seed. The default is the complete
   `offline_mlp` pilot, but **RUN_JOB=False prevents accidental execution**.
   Check the displayed parameters before explicitly enabling it.

The real data bundle has 454 usable observations, 184 visible observations and
26 fixed-1B visible observations. It is loaded from the committed shared files,
not re-split in Colab. The formal seeds are 38-45 and K=128 throughout.

The oracle checkpoint is a small simulator network, not the weights of a 1B
language model. Actual work is surrogate/policy/diffusion training by the selected
optimization method. A small oracle does not make every method cheap: RoMA,
Tri-Mentoring, PGS and SPADE should be piloted individually.

## Method/device policy

The three controls and BO-qEI, GA on GP and BDI default to CPU in the launcher.
The three kernel/GP methods still use **float64**. Neural methods default to CUDA
and float32, without mixed precision. Select a method's CPU/GPU device type before
its pilot and keep it for that method's formal runs.

The notebook records and compares critical package versions and GPU model against
the first run of a method. An environment change requires explicit review; do not
silently pool timings from different GPUs. Full installed packages, hardware,
dtype, source, candidates and scores are also saved by the evaluator. Exact
cross-platform numeric equality is not promised by PyTorch.

The notebook preserves the preinstalled torch version but does not claim a
permanently fixed Colab environment. Its displayed environment and pilot costs
must be checked on the actual runtime. If dependencies fail `pip check`, resolve
that environment problem before training.

## Pilot, then formal runs

Start with `PHASE="pilot"` (seed 0). Use the full frozen budget, not a reduced
smoke-test budget. Review time, peak memory, numerical stability, replay/rollout
diagnostics and output completeness. Do not tune budgets using the oracle score.

After that method/setting passes, set `PHASE="formal"`, manually confirm
`CONFIRM_PILOT_REVIEWED=True`, and run each of seeds 38-45. The evaluator verifies
the same-plan pilot's candidates, hashes, configuration and evaluations. It does
not rank incomplete formal seed sets. Run the main `multi_scale` setting first;
`fixed_1b` uses the same bundle and budgets as an ablation, with its own pilot.

Change methods by selecting another frozen run ID, not by editing `plan.json`.
The first-batch plan covers our 19 methods. The other member's inverse methods
are not assigned budgets here; independently frozen plans are not automatically
mergeable into one ranking.

## Persistence and interruption

Training writes under `/content/llmdm_run_state`, not directly into Drive's
mounted filesystem. The launcher:

- records an immutable dispatch intent in Drive **before** starting a child job;
- takes a verified archive before the job, periodically during execution and
  after either success or failure;
- stores each archive under a new filename plus SHA256, checks the copied bytes,
  and preserves older valid snapshots;
- records live console output and, on Linux with GNU time, peak resident memory
  in a per-job log directory;
- restores only safe, hash-valid archives into a fresh local state directory.

Do not run two Colab sessions against the same backup directory. File locks are
local to `/content`; the notebook does not claim distributed locking on Drive.
The helper is tested on local filesystems, not a guarantee of Drive/FUSE behavior.

If the runtime disappears, the latest unsaved work can still be lost. A dispatch
without a completion record requires inspection and an explicit
`INFRASTRUCTURE_RETRY_REASON`. Keep the same frozen seed and method settings.
Never use that field to hide a numerical/algorithmic failure or choose easier
seeds. A corrupted newest archive can fall back to an older valid one with a
warning, but a known completed success must not be silently retrained if that
older snapshot lacks its completion record.

**Resume is at job level**, not mid-training checkpoint continuation: verified
successful jobs are skipped; interrupted jobs restart under the same seed/config
after explicit review. All previous dispatches/attempts remain recorded. No
keep-alive or quota-avoidance scripts are included.

## Reference and validation scope

The [Colab FAQ](https://research.google.com/colaboratory/faq.html) documents variable
GPU/runtime availability, VM deletion and Drive I/O limitations. Plan around
those limits; a particular GPU or uninterrupted runtime is not guaranteed.
[PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
explain why recording seeds does not guarantee identical outputs across platforms.

Local validation covers the notebook's Python syntax, frozen release integrity,
CLI contracts, fake child jobs, archive corruption, safe restoration and retries.
An actual Colab GPU session and Google Drive mount still need to be checked during
the first pilot. No full-budget real-data method training is implied by these tests.
