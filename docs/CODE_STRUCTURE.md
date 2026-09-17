# Code Structure and Data Contracts

The current experiment has one method-facing boundary: `OfflineProblem` plus
`RunContext` in, `MethodResult` out. The evaluator retains the oracle and hidden
reference data. Module organization does not change that boundary.

## Execution and records

```text
src/llm_design_bench/
  problem.py, spaces.py, optimizers/base.py   method-facing contracts
  evaluation/
    plan_types.py          frozen plan field types and structural JSON checks
    llmdm_protocol.py      plan identity, protocol, pilot gate and formal entry
    seed_types.py          run configuration, evaluator protocol, per-seed records
    seed_runner.py         method construction, execution and final evaluation
    seed_statistics.py     candidate scores, diagnostics and cross-seed summary
    seed_persistence.py    row construction, identities, shard merge and progress
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
their ordering is part of the recovery contract. The small stdlib-only
`colab_support` remains independently usable; its type-only imports do not add
runtime dependencies to snapshot or single-job recovery.

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
