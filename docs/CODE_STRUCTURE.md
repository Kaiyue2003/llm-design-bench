# Code Structure and Data Contracts

The current experiment has one method-facing boundary: `OfflineProblem` plus
`RunContext` in, `MethodResult` out. The evaluator retains the oracle and hidden
reference data. Module organization does not change that boundary.

## Execution and records

```text
src/llm_design_bench/
  _integer_parameters.py                    strict execution-only integer checks
  problem.py, spaces.py, optimizers/base.py   method-facing contracts
  evaluation/
    plan_types.py          frozen plan field types and structural JSON checks
    llmdm_protocol.py      plan identity, protocol, pilot gate and formal entry
    seed_types.py          run configuration, evaluator protocol, per-seed records
    seed_contracts.py      shared phase, required-seed and ranking eligibility checks
    seed_runner.py         method construction, execution and final evaluation
    seed_statistics.py     candidate scores, diagnostics and cross-seed summary
    report_scores.py       read-only raw/refnorm consistency checks for report rows
    seed_persistence.py    row construction, identities, shard merge and progress
    unified_report.py      report preparation and directory-locked publication
    artifact_types.py      manifest and environment record types
    run_artifacts.py       locks, atomic artifacts and successful-attempt validation
scripts/
  colab_types.py           job, journal, preview and diagnostic report contracts
  colab_batch.py           finite queue, pilot approval and sequential dispatch
  colab_verification.py    read-only CSV, configuration and artifact validation
  colab_support.py         single-job execution, journaling, snapshots and restore
  colab_v2.py              pinned checkout/environment setup and source checks
```

`seed_runner` retains its existing public configuration, runner, normalization,
summary and merge imports as re-exports. It does not implement a second copy of
the extracted code. New internal code should import the module that owns a
responsibility. In particular, statistics never launches methods and Colab
verification never dispatches jobs or grants approval.

Artifact locking and attempt publication remain together in `run_artifacts`:
their ordering is part of the recovery contract. The stdlib-only
`colab_support` remains independently usable; its type-only imports do not add
runtime dependencies to snapshot or single-job recovery.

## Function-level lifecycles

The seed runner's `_run_method_seed_benchmark_unlocked` separates method/seed
iteration and final summary construction from `_run_seed_attempt`, which owns
a single seed's attempt lifecycle. Private stage helpers handle method
construction, attempt preparation, timed method execution, candidate persistence,
oracle evaluation and diagnostics. The single-attempt owner retains the
`try/except/finally` boundary: failures and keyboard interruptions still record
their stage, finalize available artifacts and save progress before any required
exception propagation.
Preparing or restoring an attempt is not a second training entry point.

The ordering remains part of the contract:

- Seed and construct the method before checking its saved attempt. A verified
  resume may reconstruct configuration but does not run the method or oracle.
- Persist the exact candidate batch before even converting it to target fidelity.
- Keep per-method dtype overrides separate from the suite's shared configuration.
- Finish the attempt before publishing the updated seed-result CSV. Keep the
  original timing boundaries; total time is recorded before finalization and
  CSV writes, while method and evaluation timers cover only their own stages.

`run_job` delegates to `_prepare_dispatch`, `_supervise_process` and
`_publish_completion`. Process supervision uses `_capture_output` for its reader
and `_wait_for_process` for polling and periodic backups. Process cleanup and
reader shutdown remain one supervised lifecycle. The dispatch order is remote intent, local
intent, initial snapshot, child execution and log cleanup, local completion,
final snapshot, then remote completion. A failed final backup therefore cannot
advertise a completed job in the remote journal. Exceptions and nonzero child
exit codes are propagated after the existing failure-publication steps; none of
these helpers grants approval, retries a job or changes its identity.

These are private function boundaries within the existing modules, not a new
execution framework. The public runner APIs, JSON records and frozen-release
rules remain unchanged. Statistics and frozen-protocol gate functions do not
need further splitting merely to reduce their line counts.

## Type checking is not trust validation

- Named internal records use `TypedDict` and status/phase literals so misspelled
  fields and incorrect value types are caught during development. Existing
  dictionary and JSON shapes, field order and serialization are retained.
- `MethodSpec`, `SeedBenchmarkConfig` and `SeedBenchmarkResult` remain dataclasses.
  The evaluator protocol describes only the two operations the runner needs;
  methods themselves still never receive the evaluator.
- Method-specific kwargs and provenance remain extensible. Dynamic summary
  columns use mappings rather than a misleading closed schema.
- Failed or unfinished measurements may be null in saved JSON. Result types
  preserve that possibility instead of asserting every metric is a float.
- External JSON and CSV remain untrusted inputs to the existing validators.
  Frozen plan reads now also check required field types without coercing values,
  injecting defaults, deleting extra metadata, or rewriting any hashes.
  `read_plan_shape` alone does **not** establish protocol/source compatibility.
  Use `load_method_plan` / `validate_method_plan` before execution.
- Restored result rows are typed at the internal producer/consumer boundary only
  after the existing artifact checks. A `cast` does not perform runtime checks
  and must never replace checksum, identity, numeric or retry validation.

Strict mypy checks cover the selected modules listed in `pyproject.toml`, not
the whole repository. Pandas stubs are a development-only dependency. Narrow
casts at OS, NumPy archive-writing and PyTorch backend boundaries describe APIs
whose upstream/platform stubs cannot express the existing calls; runtime checks
and behavior are retained.

## Frozen releases and Colab loading

No files under `experiments/`, `notebooks/`, `reference_results/` or `configs/`
are rewritten by this refactor. Archived results do not need retraining.
Source organization changes the current package fingerprint, so current code
must not be substituted into an old frozen plan or have its hash checks bypassed.

Current Colab queue code is loaded from a complete checkout with its `scripts/`
directory on the import path; keep `colab_verification.py` with `colab_batch.py`.
The v2 launcher also checks preloaded companion modules for a different checkout.
Historical notebooks and the pinned single-file batch-cell downloader continue
to use their recorded old revisions; this refactor does not migrate an active
Colab runtime. A new release must freeze and test its complete source set.

Regression tests cover plan identity, per-seed JSON ordering and fingerprints,
summaries, queue identity, read-only previews, artifact checks, interruption and
retry rules. These tests do not emulate the actual Google Drive backend or save
in-memory training progress.
