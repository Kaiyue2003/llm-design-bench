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
python -m pytest -q
python -m build
python -m twine check dist/*
```
