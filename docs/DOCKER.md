# Docker Reproduction

The container pins Python 3.11.15, uv 0.11.8, and every Python package through
`uv.lock`. It is CPU-first and fixes common BLAS thread counts to one.

## One command

The all-methods services include CbAS, MINs, DDOM, GABO, GTG, RGD, BONET,
DEMO, ROOT, SPADE, and the four existing baselines. Build the current checkout
and run an independent replay check:

```bash
docker compose run --build --rm all-methods-smoke
```

This trains all fourteen methods on Ackley and Booth twice, with seed 38 and
small settings from `configs/all_methods_smoke.json`. Both executions start
fresh. The verifier checks input hashes and compares every candidate and oracle
score exactly. Results persist in
`results/docker/all-methods-smoke/{first,replay}/`; the first directory contains
`replay_verification.json`.

For the complete configured eight-seed suite (the nine publication synthetic
tasks plus real Data Recipes logged data and simulator):

```bash
docker compose run --build --rm all-methods
```

This uses normal training budgets and can take a long time on CPU. Interrupted
runs retain completed rows and resume only when code, dependency versions,
data-asset hashes, method settings, and saved artifact hashes match. The smoke
configuration is for integration checks, not a claim of optimization quality.

The original three-method reproduction remains available:

From a clone of the repository:

```bash
docker compose run --rm publication
```

Windows users can run:

```powershell
.\scripts\reproduce_docker.ps1
```

Linux and macOS users can run:

```bash
./scripts/reproduce_docker.sh
```

The first publication run clones Data Recipes at commit
`37269969a0957448d51622e0c083977bc5d260e8` into the named
`data-recipes` volume. The upstream dataset and simulator are not
copied into this repository or image. Later runs reuse the verified checkout.

Results are written to `results/docker/publication/` on the host, not
left inside the disposable container. This directory includes raw per-seed
rows, summary CSVs, Markdown tables, Overleaf tables, run metadata, and
`container_environment.json`.

## Quick container test

```bash
docker compose run --rm smoke
```

The smoke service runs Ackley and Booth with one seed and small training
budgets. Its outputs appear in `results/docker/smoke/`.

## Pulling a published image

The container workflow publishes branch, commit-SHA, and release tags to:

```text
ghcr.io/kaiyue2003/llm-design-bench
```

Set an immutable image tag before running Compose:

```bash
LLM_DESIGN_BENCH_IMAGE=ghcr.io/kaiyue2003/llm-design-bench:sha-<commit> \
docker compose run --rm publication
```

On PowerShell:

```powershell
$env:LLM_DESIGN_BENCH_IMAGE = "ghcr.io/kaiyue2003/llm-design-bench:sha-<commit>"
docker compose run --rm publication
```

Commit-SHA tags are preferred for publication. The `main` tag is
convenient but mutable.

## What reproducibility means

New runs save:

- `datasets/*.npz`: exact optimizer-visible designs, context, utility, target
  fidelity and original bounds where applicable;
- `candidates/*.npz`: each method/seed candidate batch and evaluator scores;
- `raw_runs.csv`: per-run settings, source revisions, diagnostics and artifact
  hashes;
- `run_metadata.json` and `METHOD_PROVENANCE.md`: source and data provenance,
  source-content fingerprint, package versions, and algorithmic adaptations;
- `container_environment.json`: image revision, Python/Torch versions,
  dependency lock hash and the full method registry.

Verify integrity or compare two independent runs from the same configuration:

```bash
python scripts/verify_reproduction.py results/first
python scripts/verify_reproduction.py results/first --compare results/replay
```

Default comparison is exact. Use explicit `--atol` and `--rtol` only when
cross-platform numerical tolerance is intended. The verifier also checks
complete task/seed/method coverage and refuses mismatched configurations.
It never executes code from NPZ files.

The Data Recipes volume is fetched at its pinned revision on first use. Once
that commit is cached, reproduction does not require another Git fetch.
Original dataset and simulator assets are hashed separately from the Git
revision, so modified assets cannot silently enter a resumed experiment.

The current Docker services use CPU. Native PyTorch methods also accept
`--device cuda`; GPU replay requires a compatible CUDA runtime and explicitly
matched device environment. CPU validation does not establish GPU parity.

The image fixes code, Python dependencies, CPU execution, random seeds,
threading, and the external dataset commit. Raw results and manifests make a
run auditable. Bitwise equality can still be affected by host CPU instruction
sets and low-level numerical libraries; publication checks should compare
within declared numerical tolerances as well as inspect seeds, row counts,
failures, and configuration fingerprints.
