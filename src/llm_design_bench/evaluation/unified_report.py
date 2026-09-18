from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_design_bench.evaluation.report_scores import validate_report_scores
from llm_design_bench.evaluation.run_artifacts import (
    atomic_bytes,
    atomic_json,
    result_directory_lock,
)
from llm_design_bench.evaluation.seed_contracts import (
    LEGACY_PUBLICATION_SOURCE as LEGACY_PUBLICATION_SOURCE,
    validate_seed_contracts,
)
from llm_design_bench.evaluation.seed_runner import (
    DEFAULT_SEED_BENCHMARK_CONFIG,
    RESULT_SCHEMA_VERSION,
    MethodSpec,
    SeedBenchmarkConfig,
    merge_result_rows,
    reference_normalize,
    run_method_seed_benchmark,
    summarize_seed_results,
    validate_existing_result_config,
)
from llm_design_bench.problem import OfflineProblem

UNIFIED_RESULTS_FILENAME = "method_seed_results.csv"
UNIFIED_SUMMARY_FILENAME = "method_seed_summary.csv"
RANK_SUMMARY_FILENAME = "rank_summary.csv"
D_BEST_SUMMARY_FILENAME = "d_best_summary.csv"
REPORT_METADATA_FILENAME = "run_metadata.json"
REPORT_MARKDOWN_FILENAME = "README.md"
REPORT_LATEX_FILENAME = "benchmark_table.tex"
UNIFIED_REPORT_FILENAMES = (
    UNIFIED_RESULTS_FILENAME,
    UNIFIED_SUMMARY_FILENAME,
    RANK_SUMMARY_FILENAME,
    D_BEST_SUMMARY_FILENAME,
    REPORT_METADATA_FILENAME,
    REPORT_MARKDOWN_FILENAME,
    REPORT_LATEX_FILENAME,
)


@dataclass(frozen=True)
class BenchmarkTrial:
    """Evaluator-owned objects and references for one paired trial seed."""

    evaluator_task: Any
    problem: OfflineProblem
    reference_utility: np.ndarray
    dataset_seed: int | None = None
    split_seed: int | None = None
    d_best_utility: float | None = None

    def __post_init__(self) -> None:
        reference = np.asarray(self.reference_utility, dtype=float).reshape(-1)
        if reference.size == 0 or not np.isfinite(reference).all():
            raise ValueError("reference_utility must be non-empty and finite")
        d_best = self.d_best_utility
        if d_best is None:
            d_best = float(self.problem.train_utility.max().detach().cpu())
        if not math.isfinite(d_best):
            raise ValueError("d_best_utility must be finite")
        if not hasattr(self.evaluator_task, "at_target_fidelity"):
            raise TypeError("evaluator_task must define at_target_fidelity")
        if not hasattr(self.evaluator_task, "predict"):
            raise TypeError("evaluator_task must define predict")
        object.__setattr__(self, "reference_utility", reference.copy())
        object.__setattr__(self, "d_best_utility", float(d_best))


TrialFactory = Callable[[int], BenchmarkTrial]


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    """A task whose factory builds the data boundary for each paired seed."""

    task_id: str
    display_name: str
    suite: str
    category: str
    category_display_name: str
    trial_factory: TrialFactory
    normalization_reference_id: str = "full_logged_reference"

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "display_name",
            "suite",
            "category",
            "category_display_name",
            "normalization_reference_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if not callable(self.trial_factory):
            raise TypeError("trial_factory must be callable")


@dataclass(frozen=True)
class UnifiedBenchmarkResult:
    per_seed: pd.DataFrame
    summary: pd.DataFrame
    ranks: pd.DataFrame
    d_best: pd.DataFrame


def run_benchmark_suite(
    tasks: Sequence[BenchmarkTaskSpec],
    methods: Sequence[MethodSpec | str],
    *,
    config: SeedBenchmarkConfig = DEFAULT_SEED_BENCHMARK_CONFIG,
    metadata: Mapping[str, Any] | None = None,
) -> UnifiedBenchmarkResult:
    """Reserve the result directory before running any report-producing suite."""
    with result_directory_lock(config.results_dir):
        validate_existing_result_config(config)
        return _run_benchmark_suite_locked(
            tasks, methods, config=config, metadata=metadata
        )


def _run_benchmark_suite_locked(
    tasks: Sequence[BenchmarkTaskSpec],
    methods: Sequence[MethodSpec | str],
    *,
    config: SeedBenchmarkConfig,
    metadata: Mapping[str, Any] | None,
) -> UnifiedBenchmarkResult:
    """Run registered methods over tasks and paired seeds, then write one report.

    A trial factory receives the method seed so generated logged datasets can
    use the same paired seed. It is called once per task/seed, and all methods
    in that trial receive the same immutable offline problem. Only this
    evaluator layer retains the task oracle and normalization reference.
    """

    task_specs = tuple(tasks)
    method_specs = tuple(methods)
    if not task_specs:
        raise ValueError("tasks must not be empty")
    if len({task.task_id for task in task_specs}) != len(task_specs):
        raise ValueError("task_id values must be unique")
    if not method_specs:
        raise ValueError("methods must not be empty")

    frames: list[pd.DataFrame] = []
    for task_spec in task_specs:
        for method_seed in config.seeds:
            trial = task_spec.trial_factory(method_seed)
            if not isinstance(trial, BenchmarkTrial):
                raise TypeError("trial_factory must return BenchmarkTrial")
            trial_config = replace(
                config,
                seeds=(method_seed,),
                dataset_seed=trial.dataset_seed,
                split_seed=trial.split_seed,
                normalization_reference_id=task_spec.normalization_reference_id,
                task_id=task_spec.task_id,
            )
            result = run_method_seed_benchmark(
                trial.evaluator_task,
                trial.problem,
                method_specs,
                reference_utility=trial.reference_utility,
                config=trial_config,
                write_results=False,
            )
            frame = result.per_seed.copy()
            frame["suite"] = task_spec.suite
            frame["task_id"] = task_spec.task_id
            frame["task_display_name"] = task_spec.display_name
            frame["category"] = task_spec.category
            frame["category_display_name"] = task_spec.category_display_name
            frame["d_best_utility"] = trial.d_best_utility
            reference_low = float(trial.reference_utility.min())
            reference_high = float(trial.reference_utility.max())
            frame["refnorm_d_best_score"] = reference_normalize(
                trial.d_best_utility,
                reference_low=reference_low,
                reference_high=reference_high,
            )
            frames.append(frame)
            if config.save_artifacts:
                _write_unified_report_locked(
                    pd.concat(frames, ignore_index=True),
                    config.results_dir,
                    metadata=metadata,
                    config=config,
                )

    per_seed = pd.concat(frames, ignore_index=True)
    return _write_unified_report_locked(
        per_seed,
        config.results_dir,
        metadata=metadata,
        config=config if config.save_artifacts else None,
    )


def write_unified_report(
    per_seed: pd.DataFrame,
    results_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    config: SeedBenchmarkConfig | None = None,
) -> UnifiedBenchmarkResult:
    """Validate and publish a report under the shared result-directory lock.

    Standalone reports can be prepared before touching the output directory.
    For shard updates, read and merge the existing rows only while holding the
    lock. Suite-owned writes use the private already-locked entry below.
    """

    output = Path(results_dir)
    if config is None:
        report, resolved_metadata = _prepare_unified_report(per_seed, metadata)
        with result_directory_lock(output):
            return _publish_unified_report(report, resolved_metadata, output)

    # Reject malformed new rows before creating a directory or lock file. The
    # metadata of a merged report can only be validated after reading old rows.
    _validate_unified_rows(per_seed)
    with result_directory_lock(output):
        return _write_unified_report_locked(
            per_seed, output, metadata=metadata, config=config
        )


def _write_unified_report_locked(
    per_seed: pd.DataFrame,
    results_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    config: SeedBenchmarkConfig | None = None,
) -> UnifiedBenchmarkResult:
    """Merge and write a report; the caller must already own the directory lock."""

    _validate_unified_rows(per_seed)
    output = Path(results_dir)
    existing = output / UNIFIED_RESULTS_FILENAME
    if config is not None and existing.exists():
        per_seed = merge_result_rows(
            pd.read_csv(existing),
            per_seed,
            # Every current row has already passed the attempt-level policy;
            # repeated report generation merely resumes that immutable result.
            resume=True,
            infrastructure_retry_reason=config.infrastructure_retry_reason,
        )
    report, resolved_metadata = _prepare_unified_report(per_seed, metadata)
    return _publish_unified_report(report, resolved_metadata, output)


def _prepare_unified_report(
    per_seed: pd.DataFrame, metadata: Mapping[str, Any] | None
) -> tuple[UnifiedBenchmarkResult, dict[str, Any]]:
    """Compute and validate a report without changing any filesystem state."""
    _validate_unified_rows(per_seed)
    ordered = _order_per_seed(per_seed)
    summary = summarize_seed_results(ordered)
    summary = _add_task_ranks(summary)
    ranks = summarize_method_ranks(summary)
    d_best = aggregate_d_best(ordered)
    resolved_metadata = _report_metadata(ordered, metadata)
    return (
        UnifiedBenchmarkResult(
            per_seed=ordered, summary=summary, ranks=ranks, d_best=d_best
        ),
        resolved_metadata,
    )


def _publish_unified_report(
    report: UnifiedBenchmarkResult,
    metadata: Mapping[str, Any],
    output: Path,
) -> UnifiedBenchmarkResult:
    """Publish a validated report while the caller holds the directory lock."""
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        (UNIFIED_RESULTS_FILENAME, report.per_seed),
        (UNIFIED_SUMMARY_FILENAME, report.summary),
        (RANK_SUMMARY_FILENAME, report.ranks),
        (D_BEST_SUMMARY_FILENAME, report.d_best),
    ):
        atomic_bytes(
            output / name, frame.to_csv(index=False).encode("utf-8"), replace=True
        )
    atomic_json(output / REPORT_METADATA_FILENAME, metadata, replace=True)
    atomic_bytes(
        output / REPORT_MARKDOWN_FILENAME,
        render_unified_markdown(
            report.summary, report.ranks, report.d_best, metadata
        ).encode("utf-8"),
        replace=True,
    )
    atomic_bytes(
        output / REPORT_LATEX_FILENAME,
        render_unified_latex(
            report.summary, report.ranks, report.d_best, metadata
        ).encode("utf-8"),
        replace=True,
    )
    return report


def load_legacy_publication_results(
    results_dir: str | Path,
    *,
    experiment_id: str = "publication_v1",
) -> pd.DataFrame:
    """Convert the frozen publication-v1 CSV into the unified row schema.

    The returned frame can be passed to :func:`write_unified_report` using a
    different output directory. Missing v1 diagnostics remain NaN; they are
    never reconstructed or invented.
    """

    source = Path(results_dir)
    raw_path = source / "raw_runs.csv"
    metadata_path = source / "run_metadata.json"
    if not raw_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "legacy results require raw_runs.csv and run_metadata.json"
        )
    raw = pd.read_csv(raw_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "suite",
        "task",
        "display_name",
        "category",
        "category_display_name",
        "seed",
        "task_seed",
        "optimizer_seed",
        "optimizer",
        "method_display_name",
        "recommendation_count",
        "reference_min_utility",
        "reference_max_utility",
        "d_best_utility",
        "refnorm_d_best_score",
        "raw_max_utility",
        "raw_median_utility",
        "raw_mean_utility",
        "refnorm_max_score",
        "refnorm_median_score",
        "refnorm_mean_score",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise KeyError(f"legacy raw results are missing columns: {missing}")

    converted = raw.copy()
    converted["schema_version"] = RESULT_SCHEMA_VERSION
    converted["result_source"] = LEGACY_PUBLICATION_SOURCE
    converted["experiment_id"] = experiment_id
    converted["run_id"] = converted["optimizer"]
    converted["method_id"] = converted["optimizer"]
    converted["method_seed"] = converted["optimizer_seed"].fillna(converted["seed"])
    converted["dataset_seed"] = converted["task_seed"]
    converted["split_seed"] = np.nan
    converted["task_id"] = converted["task"]
    converted["task_display_name"] = converted["display_name"]
    converted["display_name"] = converted["method_display_name"]
    converted["task_name"] = converted["task"]
    converted["objective_name"] = np.where(
        converted["suite"] == "data_mixture",
        metadata.get("data_mixture_metric", "utility"),
        converted["task"],
    )
    converted["candidate_budget"] = converted["recommendation_count"]
    converted["normalization_reference_id"] = "publication_v1_full_logged"
    converted["package_commit"] = metadata.get("code_git_commit")
    converted["device"] = metadata.get("platform")
    converted["dtype"] = None
    converted["status"] = "success"
    converted["error_type"] = None
    converted["error_message"] = None
    converted["method_seconds"] = np.nan
    converted["evaluation_seconds"] = np.nan
    converted["total_seconds"] = np.nan
    converted["training_summary_json"] = None
    converted["diagnostics_json"] = None
    converted["problem_metadata_json"] = "{}"
    converted["target_context_json"] = None

    provenance = metadata.get("method_provenance", {})
    implementation_kinds = {
        "best_logged": "native_baseline",
        "coms": "lightweight_adaptation",
        "bdi": "lightweight_adaptation",
    }
    families = {
        "best_logged": "reference",
        "coms": "forward_surrogate",
        "bdi": "forward_surrogate",
    }
    adaptations = {
        "best_logged": [],
        "coms": ["legacy_api", "multi_fidelity"],
        "bdi": ["legacy_api", "rbf_kernel", "multi_fidelity"],
    }
    converted["family"] = converted["method_id"].map(families)
    converted["implementation_kind"] = converted["method_id"].map(implementation_kinds)
    converted["adaptations_json"] = converted["method_id"].map(
        lambda method_id: _json_dumps(adaptations.get(method_id, ["legacy_api"]))
    )
    converted["source_url"] = converted["method_id"].map(provenance)
    converted["source_commit"] = None
    legacy_configs = {
        "best_logged": {},
        "coms": {
            "epochs": metadata.get("epochs"),
            "particle_steps": metadata.get("particle_steps"),
        },
        "bdi": {"steps": metadata.get("bdi_steps")},
    }
    converted["method_config_json"] = converted["method_id"].map(
        lambda method_id: _json_dumps(
            {
                "config_fingerprint": metadata.get("config_fingerprint"),
                **legacy_configs.get(method_id, {}),
            }
        )
    )
    converted["requested_method_config_json"] = converted["method_config_json"]
    for column in (
        "unique_candidate_count",
        "unique_candidate_fraction",
        "candidate_diversity",
        "candidate_novelty",
        "mean_mixture_entropy",
        "mean_active_domain_count",
    ):
        converted[column] = np.nan
    return converted


def aggregate_d_best(per_seed: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "experiment_id",
        "suite",
        "task_id",
        "task_display_name",
        "category",
        "category_display_name",
    ]
    trial_identity = [*identity, "method_seed"]
    reference_columns = [
        "d_best_utility",
        "refnorm_d_best_score",
        "reference_min_utility",
        "reference_max_utility",
        "normalization_reference_id",
    ]
    conflicts = per_seed.groupby(trial_identity, dropna=False)[
        reference_columns
    ].nunique(dropna=False)
    if (conflicts > 1).any().any():
        raise ValueError("D(best) or reference values differ across methods in a trial")
    trials = per_seed.drop_duplicates(trial_identity)
    rows: list[dict[str, Any]] = []
    for _, group in trials.groupby(identity, sort=False, dropna=False):
        first = group.iloc[0]
        row = {column: first[column] for column in identity}
        row["normalization_reference_id"] = first["normalization_reference_id"]
        row["trials"] = len(group)
        for source, prefix in (
            ("d_best_utility", "raw_utility"),
            ("refnorm_d_best_score", "refnorm_score"),
        ):
            values = pd.to_numeric(group[source], errors="coerce").dropna()
            count = len(values)
            std = float(values.std(ddof=1)) if count > 1 else float("nan")
            row[f"{prefix}_mean"] = float(values.mean()) if count else float("nan")
            row[f"{prefix}_std"] = std
            row[f"{prefix}_se"] = std / math.sqrt(count) if count > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_method_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    eligible = summary.get("rank_eligible", summary["refnorm_max_score_n"] > 0)
    successful = summary[eligible].copy()
    group_columns = [
        "experiment_id",
        "run_id",
        "method_id",
        "method_display_name",
    ]
    run_counts = summary.groupby(group_columns, sort=False, as_index=False).agg(
        requested_runs=("requested_runs", "sum"),
        successful_runs=("successful_runs", "sum"),
        failed_runs=("failed_runs", "sum"),
        missing_runs=("missing_runs", "sum"),
    )
    rank_statistics = successful.groupby(
        group_columns,
        sort=False,
        as_index=False,
    ).agg(
        mean_rank=("task_rank", "mean"),
        median_rank=("task_rank", "median"),
        tasks_ranked=("task_id", "nunique"),
    )
    ranked = run_counts.merge(
        rank_statistics,
        on=group_columns,
        how="left",
        sort=False,
    )
    ranked["tasks_ranked"] = ranked["tasks_ranked"].fillna(0).astype(int)
    return ranked[
        [
            *group_columns,
            "mean_rank",
            "median_rank",
            "tasks_ranked",
            "requested_runs",
            "successful_runs",
            "failed_runs",
            "missing_runs",
        ]
    ]


def render_unified_markdown(
    summary: pd.DataFrame,
    ranks: pd.DataFrame,
    d_best: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    tasks = _task_records(summary)
    methods = _method_records(summary)
    header = [
        "Method",
        *[task["task_display_name"] for task in tasks],
        "Mean rank",
    ]
    lines = [
        "# Unified Offline BBO Benchmark",
        "",
        (
            "Reference-normalized maximum utility, reported as mean +/- standard "
            "error (SE) across successful seeds. Higher is better. Sample SD and "
            "all other metrics remain available in `method_seed_summary.csv`."
        ),
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---", *["---:" for _ in tasks], "---:"]) + " |",
    ]
    if metadata.get("phases") == ["pilot"]:
        lines.insert(
            2, "Pilot validation only; these are not formal eight-seed results.\n"
        )
    d_best_cells = [
        _format_score_row(
            d_best[d_best["task_id"] == task["task_id"]],
            mean_column="refnorm_score_mean",
            uncertainty_column="refnorm_score_se",
        )
        for task in tasks
    ]
    lines.append("| D(best) | " + " | ".join(d_best_cells) + " | -- |")
    for method in methods:
        cells = []
        for task in tasks:
            row = summary[
                (summary["task_id"] == task["task_id"])
                & (summary["run_id"] == method["run_id"])
            ]
            cells.append(
                _format_score_row(
                    row,
                    mean_column="refnorm_max_score_mean",
                    uncertainty_column="refnorm_max_score_se",
                    include_runs=True,
                )
            )
        rank = ranks[ranks["run_id"] == method["run_id"]]
        rank_value = float("nan") if rank.empty else float(rank.iloc[0]["mean_rank"])
        rank_text = "--" if not math.isfinite(rank_value) else f"{rank_value:.2f}"
        lines.append(
            f"| {method['method_display_name']} | "
            + " | ".join(cells)
            + f" | {rank_text} |"
        )

    total_requested = int(summary["requested_runs"].sum())
    total_failed = int(summary["failed_runs"].sum())
    total_missing = int(summary["missing_runs"].sum())
    lines.extend(
        [
            "",
            "## Run status",
            "",
            f"- Requested method runs: {total_requested}",
            f"- Failed method runs: {total_failed}",
            f"- Missing method runs: {total_missing}",
            f"- Candidate budget per run: K={metadata.get('candidate_budget', 'mixed')}",
            "- `D(best)` is an evaluation reference and is excluded from method ranks.",
            "- A cell suffix `[successful/requested]` marks failed or missing required seeds.",
            "- Incomplete required seed sets and pilot runs are not eligible for method ranks.",
            "",
            "## Method provenance",
            "",
            "| Run ID | Method | Family | Implementation | Adaptations | Source |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for method in methods:
        source = method.get("source_url")
        source_text = "--" if pd.isna(source) or not source else f"[source]({source})"
        adaptations = method.get("adaptations_json")
        adaptations_text = "--" if pd.isna(adaptations) else str(adaptations)
        lines.append(
            f"| `{method['run_id']}` | {method['method_display_name']} | "
            f"{method.get('family', '--')} | {method.get('implementation_kind', '--')} | "
            f"`{adaptations_text}` | {source_text} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- [Per-seed results]({UNIFIED_RESULTS_FILENAME})",
            f"- [Mean, sample SD, and SE summary]({UNIFIED_SUMMARY_FILENAME})",
            f"- [D(best) reference summary]({D_BEST_SUMMARY_FILENAME})",
            f"- [Method ranks and failure counts]({RANK_SUMMARY_FILENAME})",
            f"- [Resolved run metadata]({REPORT_METADATA_FILENAME})",
            f"- [LaTeX table]({REPORT_LATEX_FILENAME})",
            "",
        ]
    )
    return "\n".join(lines)


def render_unified_latex(
    summary: pd.DataFrame,
    ranks: pd.DataFrame,
    d_best: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    tasks = _task_records(summary)
    methods = _method_records(summary)
    columns = "l" + "c" * len(tasks) + "c"
    header = [
        "Method",
        *[_latex_escape(task["task_display_name"]) for task in tasks],
        "Mean rank",
    ]
    lines = [
        "% Requires: \\usepackage{booktabs,graphicx,rotating}",
        "\\begin{sidewaystable*}[t]",
        "\\centering",
        (
            "\\caption{Reference-normalized maximum utility. Values are mean "
            "$\\pm$ standard error across successful seeds; $\\mathcal{D}$(best) "
            "is excluded from method ranks.}"
        ),
        "\\label{tab:unified-offline-bbo}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    if metadata.get("phases") == ["pilot"]:
        lines[3] = (
            "\\caption{Pilot validation only, not formal eight-seed results. "
            "Reference-normalized maximum utility; pilot runs are not ranked.}"
        )
    d_best_cells = [
        _format_latex_score(
            d_best[d_best["task_id"] == task["task_id"]],
            mean_column="refnorm_score_mean",
            uncertainty_column="refnorm_score_se",
        )
        for task in tasks
    ]
    lines.append("$\\mathcal{D}$(best) & " + " & ".join(d_best_cells) + " & -- \\\\")
    lines.append("\\midrule")
    for method in methods:
        cells = []
        for task in tasks:
            row = summary[
                (summary["task_id"] == task["task_id"])
                & (summary["run_id"] == method["run_id"])
            ]
            cells.append(
                _format_latex_score(
                    row,
                    mean_column="refnorm_max_score_mean",
                    uncertainty_column="refnorm_max_score_se",
                )
            )
        rank = ranks[ranks["run_id"] == method["run_id"]]
        rank_value = float("nan") if rank.empty else float(rank.iloc[0]["mean_rank"])
        rank_text = "--" if not math.isfinite(rank_value) else f"{rank_value:.2f}"
        lines.append(
            _latex_escape(method["method_display_name"])
            + " & "
            + " & ".join(cells)
            + f" & {rank_text} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\par\\vspace{0.5em}",
            (
                "\\footnotesize Full sample SD, SE, failure, runtime, diversity, "
                "novelty, configuration, and provenance fields are stored in the "
                "machine-readable result artifacts."
            ),
            "\\end{sidewaystable*}",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_unified_rows(per_seed: pd.DataFrame) -> None:
    required = {
        "schema_version",
        "result_source",
        "experiment_id",
        "suite",
        "task_id",
        "task_display_name",
        "category",
        "category_display_name",
        "run_id",
        "method_id",
        "method_display_name",
        "family",
        "implementation_kind",
        "adaptations_json",
        "source_url",
        "source_commit",
        "requested_method_config_json",
        "method_config_json",
        "method_seed",
        "dataset_seed",
        "split_seed",
        "status",
        "candidate_budget",
        "train_size",
        "d_best_utility",
        "refnorm_d_best_score",
        "reference_min_utility",
        "reference_max_utility",
        "normalization_reference_id",
        "raw_max_utility",
        "raw_median_utility",
        "raw_mean_utility",
        "refnorm_max_score",
        "refnorm_median_score",
        "refnorm_mean_score",
    }
    missing = sorted(required.difference(per_seed.columns))
    if missing:
        raise KeyError(f"unified results are missing columns: {missing}")
    if per_seed.empty:
        raise ValueError("per_seed results must not be empty")
    if set(per_seed["schema_version"]) != {RESULT_SCHEMA_VERSION}:
        raise ValueError(
            f"schema_version must be {RESULT_SCHEMA_VERSION} for every row"
        )
    if (
        per_seed["result_source"].isna().any()
        or (per_seed["result_source"].astype(str).str.len() == 0).any()
    ):
        raise ValueError("result_source must not be empty")
    if not set(per_seed["status"]).issubset({"success", "failed"}):
        raise ValueError("status must be success or failed")
    if per_seed["experiment_id"].nunique(dropna=False) != 1:
        raise ValueError("one report may contain exactly one experiment_id")
    task_fields = [
        "suite",
        "task_display_name",
        "category",
        "category_display_name",
        "normalization_reference_id",
    ]
    task_conflicts = per_seed.groupby("task_id", dropna=False)[task_fields].nunique(
        dropna=False
    )
    if (task_conflicts != 1).any().any():
        raise ValueError("each task_id must have consistent task metadata")
    method_fields = [
        "method_id",
        "method_display_name",
        "family",
        "implementation_kind",
        "adaptations_json",
        "source_url",
        "source_commit",
        "requested_method_config_json",
        "method_config_json",
    ]
    method_conflicts = per_seed.groupby("run_id", dropna=False)[method_fields].nunique(
        dropna=False
    )
    if (method_conflicts != 1).any().any():
        raise ValueError("each run_id must have consistent method metadata")
    budget_conflicts = per_seed.groupby(
        ["suite", "task_id", "run_id"],
        dropna=False,
    )["candidate_budget"].nunique(dropna=False)
    if (budget_conflicts != 1).any():
        raise ValueError("candidate_budget must be constant within a task/method run")
    keys = ["experiment_id", "suite", "task_id", "run_id", "method_seed"]
    if per_seed.duplicated(keys).any():
        raise ValueError("duplicate method/task/seed rows are not allowed")
    validate_seed_contracts(per_seed)
    validate_report_scores(per_seed)


def _order_per_seed(per_seed: pd.DataFrame) -> pd.DataFrame:
    ordered = per_seed.copy()
    task_order = {
        task_id: index
        for index, task_id in enumerate(dict.fromkeys(ordered["task_id"]))
    }
    run_order = {
        run_id: index for index, run_id in enumerate(dict.fromkeys(ordered["run_id"]))
    }
    ordered["_task_order"] = ordered["task_id"].map(task_order)
    ordered["_run_order"] = ordered["run_id"].map(run_order)
    ordered = ordered.sort_values(
        ["experiment_id", "_task_order", "method_seed", "_run_order"],
        kind="stable",
    )
    return ordered.drop(columns=["_task_order", "_run_order"]).reset_index(drop=True)


def _add_task_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["task_rank"] = np.nan
    successful = ranked.get("rank_eligible", ranked["refnorm_max_score_n"] > 0)
    ranked.loc[successful, "task_rank"] = (
        ranked[successful]
        .groupby(
            ["experiment_id", "suite", "task_id"],
            dropna=False,
        )["refnorm_max_score_mean"]
        .rank(method="average", ascending=False)
    )
    return ranked


def _report_metadata(
    per_seed: pd.DataFrame,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate_budgets = sorted(
        int(value) for value in per_seed["candidate_budget"].dropna().unique()
    )
    method_seeds = sorted(
        int(value) for value in per_seed["method_seed"].dropna().unique()
    )
    resolved: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_ids": sorted(
            str(value) for value in per_seed["experiment_id"].unique()
        ),
        "methods": list(dict.fromkeys(per_seed["run_id"].astype(str))),
        "tasks": list(dict.fromkeys(per_seed["task_id"].astype(str))),
        "method_seeds": method_seeds,
        "candidate_budget": (
            candidate_budgets[0] if len(candidate_budgets) == 1 else candidate_budgets
        ),
        "uncertainty_columns": {
            "std": "sample standard deviation (ddof=1)",
            "se": "sample standard error over successful runs",
        },
        "table_uncertainty": "standard_error",
        "d_best_ranked": False,
        "failed_runs_retained": True,
        "incomplete_seed_sets_ranked": False,
    }
    if "phase" in per_seed:
        resolved["phases"] = sorted(str(value) for value in per_seed["phase"].unique())
    if "required_seeds_json" in per_seed:
        resolved["required_seed_sets"] = [
            json.loads(value)
            for value in per_seed["required_seeds_json"].dropna().unique()
        ]
    for key, value in dict(metadata or {}).items():
        if key in resolved and resolved[key] != value:
            raise ValueError(f"report metadata may not override reserved field {key!r}")
        resolved[key] = value
    return resolved


def _task_records(summary: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "suite",
        "task_id",
        "task_display_name",
        "category",
        "category_display_name",
    ]
    return summary[columns].drop_duplicates().to_dict(orient="records")


def _method_records(summary: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "run_id",
        "method_id",
        "method_display_name",
        "family",
        "implementation_kind",
        "adaptations_json",
        "source_url",
    ]
    return summary[columns].drop_duplicates("run_id").to_dict(orient="records")


def _format_score_row(
    row: pd.DataFrame,
    *,
    mean_column: str,
    uncertainty_column: str,
    include_runs: bool = False,
) -> str:
    if row.empty:
        return "--"
    item = row.iloc[0]
    mean = float(item[mean_column])
    uncertainty = float(item[uncertainty_column])
    if not math.isfinite(mean):
        if include_runs and item.get("phase") == "formal":
            return f"-- [{int(item['successful_runs'])}/{int(item['requested_runs'])}]"
        return "--"
    value = f"{mean:.3f}"
    if math.isfinite(uncertainty):
        value += f" +/- {uncertainty:.3f}"
    if include_runs and (
        int(item["failed_runs"]) > 0 or int(item.get("missing_runs", 0)) > 0
    ):
        value += f" [{int(item['successful_runs'])}/{int(item['requested_runs'])}]"
    return value


def _format_latex_score(
    row: pd.DataFrame,
    *,
    mean_column: str,
    uncertainty_column: str,
) -> str:
    value = _format_score_row(
        row,
        mean_column=mean_column,
        uncertainty_column=uncertainty_column,
        include_runs="failed_runs" in row.columns,
    )
    if value == "--":
        return value
    return "$" + value.replace(" +/- ", " \\pm ") + "$"


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
    return "".join(replacements.get(character, character) for character in str(value))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def _score_columns() -> tuple[str, ...]:
    return (
        "raw_max_utility",
        "raw_median_utility",
        "raw_mean_utility",
        "refnorm_max_score",
        "refnorm_median_score",
        "refnorm_mean_score",
    )
