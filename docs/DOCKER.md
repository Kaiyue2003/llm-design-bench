# Docker Reproduction

The container pins Python 3.11.15, uv 0.11.8, and every Python package through
`uv.lock`. It is CPU-first and fixes common BLAS thread counts to one.

## One command

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

The image fixes code, Python dependencies, CPU execution, random seeds,
threading, and the external dataset commit. Raw results and manifests make a
run auditable. Bitwise equality can still be affected by host CPU instruction
sets and low-level numerical libraries; publication checks should compare
within declared numerical tolerances as well as inspect seeds, row counts,
failures, and configuration fingerprints.
