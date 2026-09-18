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
- Keep `llm-design-bench` as the single formal LLM-DM workflow;
  `llm-design-bench-llmdm` is its alias. Do not reintroduce retired experiment
  entry points or parallel budget presets.
- Preserve generic Python task/method APIs, source attribution, and immutable
  historical archives; do not rewrite old results to match a new implementation.
- Update `docs/REPRODUCING.md` when defaults or result semantics change.

Code integration is reviewed with regression, packaging and small runtime checks.
New experimental budgets, frozen release plans and full-budget pilots are agreed
and run **after merge**. Full benchmark reproduction and deferred SPADE selection
are not prerequisites for this code merge. Record unavailable container/CUDA
checks honestly rather than treating CPU tests as equivalent validation.

Before opening a pull request, run:

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
```
