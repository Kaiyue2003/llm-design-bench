from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

from llm_design_bench.metrics.diversity import pairwise_diversity
from llm_design_bench.metrics.novelty import candidate_novelty
from llm_design_bench.optimizers.registry import (
    get_method_capabilities,
    get_method_metadata,
    make_method,
)
from llm_design_bench.problem import OfflineProblem, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace


DEFAULT_METHOD_SEEDS = tuple(range(38, 46))

_SCORE_COLUMNS = (
    "raw_max_utility",
    "raw_median_utility",
    "raw_mean_utility",
    "refnorm_max_score",
    "refnorm_median_score",
    "refnorm_mean_score",
)

_SUMMARY_METRICS = _SCORE_COLUMNS + (
    "method_seconds",
    "evaluation_seconds",
    "total_seconds",
    "unique_candidate_count",
    "unique_candidate_fraction",
    "candidate_diversity",
    "candidate_novelty",
    "mean_mixture_entropy",
    "mean_active_domain_count",
)


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.method_id:
            raise ValueError("method_id must not be empty")
        resolved_run_id = self.run_id or self.method_id
        if not resolved_run_id:
            raise ValueError("run_id must not be empty")
        object.__setattr__(self, "run_id", resolved_run_id)
        object.__setattr__(self, "kwargs", dict(self.kwargs))


@dataclass(frozen=True)
class SeedBenchmarkConfig:
    experiment_id: str = "default"
    normalization_reference_id: str = "provided_reference"
    seeds: tuple[int, ...] = DEFAULT_METHOD_SEEDS
    candidate_budget: int = 128
    device: torch.device | str = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    dataset_seed: int | None = None
    split_seed: int | None = None
    results_dir: Path = Path("results")
    fail_fast: bool = False
    package_commit: str | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.normalization_reference_id:
            raise ValueError("normalization_reference_id must not be empty")
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds:
            raise ValueError("seeds must not be empty")
        if any(seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if self.candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError("dtype must be floating point")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(self, "results_dir", Path(self.results_dir))


@dataclass(frozen=True)
class SeedBenchmarkResult:
    per_seed: pd.DataFrame
    summary: pd.DataFrame


def run_method_seed_benchmark(
    evaluator_task,
    problem: OfflineProblem,
    methods: Sequence[MethodSpec | str],
    *,
    reference_utility: np.ndarray,
    config: SeedBenchmarkConfig = SeedBenchmarkConfig(),
) -> SeedBenchmarkResult:
    """Run registered methods across paired seeds, then evaluate candidates.

    ``problem`` is the only object passed to a method. ``evaluator_task`` and
    ``reference_utility`` remain evaluator-only, so final oracle information is
    unavailable during method training and candidate search.
    """

    specs = _normalize_method_specs(methods)
    _validate_specs(specs)
    reference = _validate_reference(reference_utility)
    reference_low = float(reference.min())
    reference_high = float(reference.max())

    rows: list[dict[str, Any]] = []
    for spec in specs:
        metadata = get_method_metadata(spec.method_id)
        capabilities = get_method_capabilities(spec.method_id)
        _validate_method_compatibility(
            spec.method_id,
            capabilities,
            problem,
            config,
        )
        for seed in config.seeds:
            row = _base_row(
                spec=spec,
                metadata=metadata,
                problem=problem,
                config=config,
                seed=seed,
                reference_low=reference_low,
                reference_high=reference_high,
            )
            started = time.perf_counter()
            try:
                _seed_everything(seed)
                method = make_method(spec.method_id, **spec.kwargs)
                row["resolved_method_config_json"] = _json_dumps(
                    method.configuration()
                )
                context = RunContext(
                    method_seed=seed,
                    candidate_budget=config.candidate_budget,
                    device=config.device,
                    dtype=config.dtype,
                    dataset_seed=config.dataset_seed,
                    split_seed=config.split_seed,
                )

                method_started = time.perf_counter()
                method_result = method.run(problem, context)
                row["method_seconds"] = time.perf_counter() - method_started

                evaluation_started = time.perf_counter()
                candidates = method_result.candidates.detach().cpu().numpy()
                candidate_batch = evaluator_task.at_target_fidelity(candidates)
                utilities = np.asarray(
                    evaluator_task.predict(candidate_batch),
                    dtype=float,
                )
                _validate_oracle_output(
                    utilities,
                    expected=config.candidate_budget,
                )
                row["evaluation_seconds"] = time.perf_counter() - evaluation_started

                row.update(
                    _candidate_scores(
                        utilities,
                        reference_low=reference_low,
                        reference_high=reference_high,
                    )
                )
                row.update(_candidate_diagnostics(candidates, problem))
                row["training_summary_json"] = _json_dumps(
                    method_result.training_summary
                )
                row["diagnostics_json"] = _json_dumps(method_result.diagnostics)
                row["status"] = "success"
            except Exception as exc:
                row["status"] = "failed"
                row["error_type"] = type(exc).__name__
                row["error_message"] = str(exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if config.fail_fast:
                    raise
            finally:
                row["total_seconds"] = time.perf_counter() - started
                rows.append(row)

    per_seed = pd.DataFrame(rows)
    summary = summarize_seed_results(per_seed)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(config.results_dir / "method_seed_results.csv", index=False)
    summary.to_csv(config.results_dir / "method_seed_summary.csv", index=False)
    (config.results_dir / "method_seed_table.md").write_text(
        render_seed_summary_markdown(summary),
        encoding="utf-8",
    )
    (config.results_dir / "method_seed_table.tex").write_text(
        render_seed_summary_latex(summary),
        encoding="utf-8",
    )
    return SeedBenchmarkResult(per_seed=per_seed, summary=summary)


def summarize_seed_results(per_seed: pd.DataFrame) -> pd.DataFrame:
    required = {
        "experiment_id",
        "run_id",
        "method_id",
        "status",
        "method_seed",
    }
    missing = sorted(required.difference(per_seed.columns))
    if missing:
        raise KeyError(f"missing seed-result columns: {missing}")

    rows: list[dict[str, Any]] = []
    groups = per_seed.groupby(["experiment_id", "run_id"], sort=False)
    for (experiment_id, run_id), group in groups:
        first = group.iloc[0]
        successful = group[group["status"] == "success"]
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "method_id": first["method_id"],
            "display_name": first.get("display_name", first["method_id"]),
            "family": first.get("family"),
            "implementation_kind": first.get("implementation_kind"),
            "requested_runs": int(len(group)),
            "successful_runs": int(len(successful)),
            "failed_runs": int(len(group) - len(successful)),
            "candidate_budget": int(first.get("candidate_budget", 0)),
        }
        for metric in _SUMMARY_METRICS:
            values = pd.to_numeric(
                successful[metric] if metric in successful else pd.Series(dtype=float),
                errors="coerce",
            ).dropna()
            count = int(len(values))
            mean = float(values.mean()) if count else float("nan")
            sample_std = float(values.std(ddof=1)) if count > 1 else float("nan")
            standard_error = sample_std / math.sqrt(count) if count > 1 else float("nan")
            minimum = float(values.min()) if count else float("nan")
            maximum = float(values.max()) if count else float("nan")
            critical = (
                float(student_t.ppf(0.975, df=count - 1))
                if count > 1
                else float("nan")
            )
            row[f"{metric}_n"] = count
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = sample_std
            row[f"{metric}_se"] = standard_error
            row[f"{metric}_ci95_low"] = mean - critical * standard_error
            row[f"{metric}_ci95_high"] = mean + critical * standard_error
            row[f"{metric}_min"] = minimum
            row[f"{metric}_max"] = maximum
            row[f"{metric}_range"] = maximum - minimum
        rows.append(row)
    return pd.DataFrame(rows)


def render_seed_summary_markdown(
    summary: pd.DataFrame,
    *,
    metric: str = "refnorm_max_score",
) -> str:
    """Render an auditable seed table with both uncertainty and observed range."""

    _validate_summary_metric(summary, metric)
    lines = [
        "# Seeded Method Results",
        "",
        (
            "Primary estimate is mean +/- standard error across successful seeds. "
            "Sample SD, the two-sided 95% Student-t confidence interval, and the "
            "observed seed range are shown separately."
        ),
        "",
        "| Experiment | Method | Mean +/- SE | Sample SD | 95% CI | Observed range | Runs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["experiment_id"]),
                    str(row["display_name"]),
                    _mean_error_cell(row, metric, "se"),
                    _number(row[f"{metric}_std"]),
                    _interval_cell(row, metric, "ci95"),
                    _interval_cell(row, metric, "observed"),
                    f"{int(row['successful_runs'])}/{int(row['requested_runs'])}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Metric: `{metric}`. Confidence intervals describe uncertainty in the seed mean; ",
            "the observed range reports the minimum and maximum realized seed scores.",
            "",
        ]
    )
    return "\n".join(lines)


def render_seed_summary_latex(
    summary: pd.DataFrame,
    *,
    metric: str = "refnorm_max_score",
) -> str:
    """Render a compact booktabs table suitable for direct Overleaf inclusion."""

    _validate_summary_metric(summary, metric)
    lines = [
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Seeded offline optimization results. The primary estimate is mean "
        "$\\pm$ standard error; SD is the sample standard deviation, CI is a two-sided "
        "95\\% Student-$t$ interval, and range is the observed minimum and maximum.}",
        "\\label{tab:seeded-method-results}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Experiment & Method & Mean $\\pm$ SE & SD & 95\\% CI & Range & Runs \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            " & ".join(
                (
                    _latex_escape(str(row["experiment_id"])),
                    _latex_escape(str(row["display_name"])),
                    _mean_error_cell(row, metric, "se", latex=True),
                    _number(row[f"{metric}_std"]),
                    _interval_cell(row, metric, "ci95", latex=True),
                    _interval_cell(row, metric, "observed", latex=True),
                    f"{int(row['successful_runs'])}/{int(row['requested_runs'])}",
                )
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _normalize_method_specs(
    methods: Sequence[MethodSpec | str],
) -> tuple[MethodSpec, ...]:
    return tuple(
        method if isinstance(method, MethodSpec) else MethodSpec(str(method))
        for method in methods
    )


def _validate_specs(specs: tuple[MethodSpec, ...]) -> None:
    if not specs:
        raise ValueError("methods must not be empty")
    run_ids = [spec.run_id for spec in specs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("method run_id values must be unique")
    for spec in specs:
        get_method_metadata(spec.method_id)


def _validate_method_compatibility(
    method_id: str,
    capabilities,
    problem: OfflineProblem,
    config: SeedBenchmarkConfig,
) -> None:
    if isinstance(problem.design_space, SimplexSpace) and not capabilities.supports_simplex:
        raise ValueError(f"method {method_id!r} does not support simplex designs")
    if isinstance(problem.design_space, BoxSpace) and not capabilities.supports_box:
        raise ValueError(f"method {method_id!r} does not support box designs")
    if capabilities.requires_gpu and config.device.type != "cuda":
        raise ValueError(f"method {method_id!r} requires a CUDA device")
    if problem.context_dim and not capabilities.supports_context:
        context = problem.train_context.detach().cpu()
        context_varies = bool(
            torch.any(context != context[0]).item()
            or torch.any(problem.target_context.detach().cpu() != context[0]).item()
        )
        if context_varies:
            raise ValueError(
                f"method {method_id!r} does not support varying fidelity context"
            )


def _validate_reference(values: np.ndarray) -> np.ndarray:
    reference = np.asarray(values, dtype=float).reshape(-1)
    if reference.size == 0:
        raise ValueError("reference_utility must not be empty")
    if not np.isfinite(reference).all():
        raise ValueError("reference_utility must be finite")
    return reference


def _base_row(
    *,
    spec: MethodSpec,
    metadata,
    problem: OfflineProblem,
    config: SeedBenchmarkConfig,
    seed: int,
    reference_low: float,
    reference_high: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": spec.run_id,
        "experiment_id": config.experiment_id,
        "method_id": spec.method_id,
        "display_name": metadata.display_name,
        "family": metadata.family.value,
        "implementation_kind": metadata.implementation_kind.value,
        "adaptations_json": _json_dumps(metadata.adaptations),
        "source_url": metadata.source_url,
        "source_commit": metadata.source_commit,
        "paper_url": metadata.paper_url,
        "original_framework": metadata.original_framework,
        "implementation_framework": metadata.implementation_framework,
        "optional_dependencies_json": _json_dumps(metadata.optional_dependencies),
        "config_schema_version": metadata.config_schema_version,
        "method_config_json": _json_dumps(spec.kwargs),
        "resolved_method_config_json": None,
        "package_commit": config.package_commit,
        "task_name": problem.metadata.task_name,
        "objective_name": problem.metadata.objective_name,
        "problem_metadata_json": _json_dumps(problem.metadata.extra),
        "target_context_json": _json_dumps(
            problem.target_context.detach().cpu().tolist()
        ),
        "method_seed": seed,
        "dataset_seed": config.dataset_seed,
        "split_seed": config.split_seed,
        "candidate_budget": config.candidate_budget,
        "train_size": problem.sample_count,
        "reference_min_utility": reference_low,
        "reference_max_utility": reference_high,
        "normalization_reference_id": config.normalization_reference_id,
        "device": str(config.device),
        "dtype": str(config.dtype),
        "status": "failed",
        "error_type": None,
        "error_message": None,
        "method_seconds": float("nan"),
        "evaluation_seconds": float("nan"),
        "total_seconds": float("nan"),
        "training_summary_json": None,
        "diagnostics_json": None,
        "unique_candidate_count": float("nan"),
        "unique_candidate_fraction": float("nan"),
        "candidate_diversity": float("nan"),
        "candidate_novelty": float("nan"),
    }
    row.update({column: float("nan") for column in _SCORE_COLUMNS})
    return row


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_oracle_output(utilities: np.ndarray, expected: int) -> None:
    if utilities.shape != (expected,):
        raise ValueError(
            f"oracle returned utility shape {utilities.shape}; expected {(expected,)}"
        )
    if not np.isfinite(utilities).all():
        raise ValueError("oracle returned non-finite utilities")


def _candidate_scores(
    utilities: np.ndarray,
    *,
    reference_low: float,
    reference_high: float,
) -> dict[str, float]:
    width = reference_high - reference_low
    minimum_width = 1e-12 * max(
        1.0,
        abs(reference_low),
        abs(reference_high),
    )

    def normalize(value: float) -> float:
        return 0.0 if width <= minimum_width else (value - reference_low) / width

    maximum = float(np.max(utilities))
    median = float(np.median(utilities))
    mean = float(np.mean(utilities))
    return {
        "raw_max_utility": maximum,
        "raw_median_utility": median,
        "raw_mean_utility": mean,
        "refnorm_max_score": normalize(maximum),
        "refnorm_median_score": normalize(median),
        "refnorm_mean_score": normalize(mean),
    }


def _candidate_diagnostics(
    candidates: np.ndarray,
    problem: OfflineProblem,
) -> dict[str, float | int]:
    normalized_candidates = _normalized_designs(candidates, problem)
    logged = problem.train_designs.detach().cpu().numpy()
    normalized_logged = _normalized_designs(logged, problem)
    rounded = np.round(candidates, decimals=12)
    unique_count = int(len(np.unique(rounded, axis=0)))
    diagnostics: dict[str, float | int] = {
        "unique_candidate_count": unique_count,
        "unique_candidate_fraction": unique_count / len(candidates),
        "candidate_diversity": pairwise_diversity(normalized_candidates),
        "candidate_novelty": candidate_novelty(
            normalized_candidates,
            normalized_logged,
        ),
    }
    if isinstance(problem.design_space, SimplexSpace):
        safe = np.where(candidates > 0, candidates, 1.0)
        entropy = -np.sum(
            np.where(candidates > 0, candidates * np.log(safe), 0.0),
            axis=1,
        )
        diagnostics["mean_mixture_entropy"] = float(entropy.mean())
        diagnostics["mean_active_domain_count"] = float(
            np.sum(candidates > 0.01, axis=1).mean()
        )
    return diagnostics


def _normalized_designs(
    designs: np.ndarray,
    problem: OfflineProblem,
) -> np.ndarray:
    designs = np.asarray(designs, dtype=float)
    if isinstance(problem.design_space, BoxSpace):
        bounds = problem.design_space.bounds.detach().cpu().numpy()
        return (designs - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return designs


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _validate_summary_metric(summary: pd.DataFrame, metric: str) -> None:
    required = {
        "experiment_id",
        "display_name",
        "successful_runs",
        "requested_runs",
        f"{metric}_mean",
        f"{metric}_std",
        f"{metric}_se",
        f"{metric}_ci95_low",
        f"{metric}_ci95_high",
        f"{metric}_min",
        f"{metric}_max",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise KeyError(f"missing seed-summary columns: {missing}")


def _mean_error_cell(
    row: pd.Series,
    metric: str,
    error: str,
    *,
    latex: bool = False,
) -> str:
    separator = " $\\pm$ " if latex else " +/- "
    return _number(row[f"{metric}_mean"]) + separator + _number(
        row[f"{metric}_{error}"]
    )


def _interval_cell(
    row: pd.Series,
    metric: str,
    interval: str,
    *,
    latex: bool = False,
) -> str:
    if interval == "ci95":
        low = row[f"{metric}_ci95_low"]
        high = row[f"{metric}_ci95_high"]
    elif interval == "observed":
        low = row[f"{metric}_min"]
        high = row[f"{metric}_max"]
    else:
        raise ValueError(f"unknown interval {interval!r}")
    left, right = ("$[", "]$") if latex else ("[", "]")
    return f"{left}{_number(low)}, {_number(high)}{right}"


def _number(value: Any) -> str:
    numeric = float(value)
    return "--" if not math.isfinite(numeric) else f"{numeric:.3f}"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(character, character) for character in value)
