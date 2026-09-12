# Batch execution in Colab without changing the frozen experiment

The batch launcher removes manual method/seed editing. It does not change the
methods, data split, candidate budget, numerical precision or frozen plan.
Existing verified results, including a completed Offline MLP pilot and its eight
formal seeds, are reused rather than retrained.

## Two finite queues, one review between them

1. Run the **pilot queue** once. By default it covers all 19 frozen run IDs at
   `multi_scale`, seed 0, with full budgets. Completed valid pilots are skipped.
2. Review the combined pilot report once: completeness, training diagnostics,
   time and memory. Passing a process exit-code check is not an automatic human
   review, and final finite diagnostics are not a complete training trace.
3. Explicitly confirm the reviewed method/setting pairs, then run the **formal
   queue**. It executes seeds 38-45 for those methods without manual seed edits.
   Existing valid formal results are skipped. A pilot for `multi_scale` does not
   approve `fixed_1b`; run and review that ablation separately.

Every queue is sequential: one method, one setting and one seed per subprocess.
There are no parallel GPU workers, automatic retries, automatic budget changes,
background schedules or keepalive scripts. This does not extend Colab runtime
limits or guarantee that a long queue fits in a single session.

## Continue an already prepared notebook

Use the [existing-session cell](../examples/colab_batch_cell.py) after the original
notebook's setup, data verification and Drive mounting cells have succeeded.
The cell needs `REPO`, `ASSETS`, `UPSTREAM`, `STATE` and `BACKUPS` from that notebook.
Do not delete/reclone the experiment checkout or allocate a new runtime just to
use batching. Stop any currently running single-job cell before starting a queue.

The cell fetches only the additional, commit-pinned launcher script and verifies
its SHA256 before loading it from a separate `/content/llmdm_batch_tools/` folder.
It does not replace the installed package or alter `/content/llmdm_repo`.
The frozen package/data checks still run before dispatch. The backup folder and
job identities are identical to the original notebook's, so prior artifacts and
Drive interruption records remain effective.

The existing-session cell only loads a `batch` object and previews the queue;
it does not launch any job. Then run the pilot queue explicitly:

```python
pilot_runs = batch.run(phase="pilot")
display(pd.DataFrame(batch.pilot_report()))
```

After reviewing every selected pilot, explicitly approve those pairs and run all
their formal seeds:

```python
reviewed = [("multi_scale", entry["run_id"]) for entry in batch.plan["methods"]]
formal_runs = batch.run(phase="formal", reviewed_pilots=reviewed)
```

Do not run the second snippet until reviewing the first report. You can select a
subset via `methods=["coms", "bdi"]` in both calls and approve only those pairs.
No per-seed edits are needed. Re-running a finished queue verifies and skips it.

For a fresh session use [LLMDM_Colab_Batch.ipynb](../notebooks/LLMDM_Colab_Batch.ipynb),
which includes setup, verified restore, queue preview and explicit execution.
That notebook defaults to `RUN_BATCH=False`; formal execution additionally needs
`CONFIRM_BATCH_PILOTS_REVIEWED=True` for exactly the selected methods/settings.
The original [single-job notebook](../notebooks/LLMDM_Colab.ipynb) remains available
for a deliberately reviewed infrastructure retry or targeted diagnosis.

## Queue settings

```python
METHODS = None                 # All run IDs in the existing frozen plan
SETTINGS = ("multi_scale",)    # Main experiment only; fixed_1b is separate
PHASE = "pilot"               # Change to formal after reviewing these pilots
NEURAL_DEVICE = "cuda"
RUN_BATCH = False              # Explicitly enable execution
CONFIRM_BATCH_PILOTS_REVIEWED = False
ALLOW_ENVIRONMENT_CHANGE = False
```

To run only a subset, use e.g. `METHODS = ["coms", "bdi"]`. No per-seed setting is
needed: pilot means seed 0; formal means all eight agreed seeds 38-45. CPU
controls/kernel methods and CUDA neural methods retain the original device
policy. A method's existing device type cannot silently change.

The underlying API is `BatchRunner(...).preview(...)`, `.run(...)` and
`.pilot_report(...)`. Preview/report are read-only and never train or query the
oracle. Direct API callers must pass reviewed `(setting, run_id)` pairs to
`.run(phase="formal", reviewed_pilots=...)`; the notebook confirmation constructs
that explicit list only for its selected queue.

## Failure, interruption and persistence

An invalid saved artifact, environment mismatch, unsuccessful child process or
backup failure stops the queue. It is not counted as a successful experiment and
does not silently trigger another attempt. Previously completed results remain.
The batch launcher does not accept a blanket infrastructure-retry reason.

Inspect the stopped job. If it was a verified infrastructure interruption, use
the original single-job workflow with the same seed/configuration and an explicit
reason, then resume the batch. Do not relabel an algorithmic failure, omit a seed
from the requested coverage, or modify a budget to make the old plan succeed.

Re-running the batch after a normal stop verifies and skips completed jobs. After
a VM loss, first restore the original state from Drive. A known completed job
whose record is missing from a restored older snapshot is blocked, not retrained.

Each actual child job still uses the original append-only journal and periodic,
verified full-state snapshots. Skipped jobs do not start new subprocesses or
create redundant archives. Old snapshots are not automatically deleted; monitor
Drive space. Never run two notebooks against the same backup directory.

See [the Colab guide](COLAB.md) for environment/authentication, runtime limits and
the difference between recovering completed jobs and mid-training checkpoints.
Tests of the queue use fake jobs/artifacts, not a new full-budget training run.
