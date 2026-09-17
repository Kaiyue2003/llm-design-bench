# Contributing

Thank you for helping improve LLM Design Bench.

## Development setup

```bash
git clone https://github.com/Kaiyue2003/llm-design-bench.git
cd llm-design-bench
python -m pip install -e ".[dev]"
python -m pytest -q
```

The `data-recipes` integration needs a separate upstream checkout. Synthetic
task and metric tests do not require those external assets.

## Code quality

Quality rules live in `pyproject.toml`. The development dependencies pin Ruff
to `0.16.6` and mypy to `1.19.1`, with pinned pandas type stubs, so contributors
and CI use the same tools.
Refresh an existing development environment with the installation command
above before running these checks:

```bash
python -m ruff check src scripts examples tests setup.py
python -m ruff format --check src scripts examples tests setup.py
python -m mypy
```

To apply formatting locally, run:

```bash
python -m ruff format src scripts examples tests setup.py
```

Review the diff and rerun the checks and tests after applying fixes. Lint and
format checks cover the maintained Python package, scripts, examples, and
tests. Type checking is deliberately incremental: it covers the public
task/problem/method boundaries, frozen plans, seed execution/statistics/
persistence, artifact metadata, and the current Colab queue and its helpers.
The exact file list is in `tool.mypy.files` in `pyproject.toml`. Passing this
gate does not mean every optimizer and reporting module has strict type
coverage. Expand the checked scope with focused tests as interfaces improve.
See [code structure](docs/CODE_STRUCTURE.md) for module ownership and the
distinction between typed internal records and external JSON validation.

CI runs these quality checks in a separate Python 3.11 job alongside the
Python 3.11/3.12 tests and distribution checks. All jobs must pass for a
change to satisfy the documented contribution checks.

Do not reformat or regenerate frozen artifacts under `notebooks/`,
`experiments/`, `reference_results/`, or `configs/` as part of code-quality
cleanup. Source edits, including formatting, change source fingerprints.
Existing frozen Colab experiments must keep their recorded code and
environment pins rather than having their manifests rewritten to match
the current checkout. Adding quality gates alone does not require rerunning
completed experiments.

## Changes

- Keep optimizer inputs restricted to the logged dataset in offline mode.
- Report objective minimization tasks as maximization utilities using
  `utility = -objective`.
- Add focused tests for behavior changes.
- Do not commit local environments, caches, or generated files under
  `results/`.
- Update `docs/REPRODUCING.md` when defaults or result semantics change.

Before opening a pull request, run:

```bash
python -m ruff check src scripts examples tests setup.py
python -m ruff format --check src scripts examples tests setup.py
python -m mypy
python -m pytest -q
python -m build
python -m twine check dist/*
```
