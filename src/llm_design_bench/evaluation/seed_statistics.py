"""Pure candidate diagnostics and per-method seed statistics."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import t as student_t  # type: ignore[import-untyped]

from llm_design_bench.evaluation.seed_contracts import (
    seed_group_contract,
    validate_seed_contracts,
)
from llm_design_bench.evaluation.seed_types import (
    FloatArray,
    SeedDiagnostics,
    SeedScores,
    _json_dumps,
)
from llm_design_bench.metrics.diversity import pairwise_diversity
from llm_design_bench.metrics.novelty import candidate_novelty
from llm_design_bench.problem import OfflineProblem
from llm_design_bench.spaces import BoxSpace, SimplexSpace

_SCORE_COLUMNS = (
    "raw_max_utility",
    "raw_median_utility",
    "raw_mean_utility",
    "refnorm_max_score",
    "refnorm_median_score",
    "refnorm_mean_score",
)

_SUMMARY_METRICS = _SCORE_COLUMNS + (
    "raw_min_loss",
    "raw_median_loss",
    "raw_mean_loss",
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
    validate_seed_contracts(per_seed)

    rows: list[dict[str, object]] = []
    group_columns = ["experiment_id"]
    group_columns.extend(
        column
        for column in (
            "suite",
            "task_id",
            "task_display_name",
            "category",
            "category_display_name",
        )
        if column in per_seed.columns
    )
    group_columns.append("run_id")
    groups = per_seed.groupby(group_columns, sort=False, dropna=False)
    for _, group in groups:
        first = group.iloc[0]
        successful = group[group["status"] == "success"]
        contract = seed_group_contract(group)
        required_seeds = contract.required_seeds
        observed_seeds = contract.observed_seeds
        environment_variants = (
            group["environment_json"].dropna().nunique()
            if "environment_json" in group
            else 0
        )
        row: dict[str, object] = {column: first[column] for column in group_columns}
        row.update(
            {
                "method_id": first["method_id"],
                "method_display_name": first.get(
                    "method_display_name",
                    first.get("display_name", first["method_id"]),
                ),
                "display_name": first.get(
                    "method_display_name",
                    first.get("display_name", first["method_id"]),
                ),
                "family": first.get("family"),
                "implementation_kind": first.get("implementation_kind"),
                "adaptations_json": first.get("adaptations_json"),
                "source_url": first.get("source_url"),
                "source_commit": first.get("source_commit"),
                "requested_method_config_json": first.get(
                    "requested_method_config_json"
                ),
                "method_config_json": first.get("method_config_json"),
                "package_commit": first.get("package_commit"),
                "normalization_reference_id": first.get("normalization_reference_id"),
                "phase": contract.phase,
                "required_seeds_json": _json_dumps(sorted(required_seeds)),
                "requested_runs": len(required_seeds),
                "attempted_runs": len(group),
                "successful_runs": len(successful),
                "failed_runs": int(len(group) - len(successful)),
                "missing_runs": len(required_seeds - observed_seeds),
                "complete_seed_set": contract.complete,
                "rank_eligible": contract.rank_eligible,
                "environment_variants": int(environment_variants),
                "mixed_environment": environment_variants > 1,
                "candidate_budget": int(first.get("candidate_budget", 0)),
            }
        )
        for metric in _SUMMARY_METRICS:
            values = pd.to_numeric(
                successful[metric] if metric in successful else pd.Series(dtype=float),
                errors="coerce",
            ).dropna()
            count = len(values)
            mean = float(values.mean()) if count else float("nan")
            sample_std = float(values.std(ddof=1)) if count > 1 else float("nan")
            standard_error = (
                sample_std / math.sqrt(count) if count > 1 else float("nan")
            )
            row[f"{metric}_n"] = count
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = sample_std
            row[f"{metric}_se"] = standard_error
            minimum = float(values.min()) if count else float("nan")
            maximum = float(values.max()) if count else float("nan")
            critical = (
                float(student_t.ppf(0.975, df=count - 1)) if count > 1 else float("nan")
            )
            row[f"{metric}_ci95_low"] = mean - critical * standard_error
            row[f"{metric}_ci95_high"] = mean + critical * standard_error
            row[f"{metric}_min"] = minimum
            row[f"{metric}_max"] = maximum
            row[f"{metric}_range"] = maximum - minimum
        rows.append(row)
    return pd.DataFrame(rows)


def _candidate_scores(
    utilities: FloatArray,
    *,
    reference_low: float,
    reference_high: float,
) -> SeedScores:
    maximum = float(np.max(utilities))
    median = float(np.median(utilities))
    mean = float(np.mean(utilities))
    return {
        "raw_max_utility": maximum,
        "raw_median_utility": median,
        "raw_mean_utility": mean,
        "refnorm_max_score": reference_normalize(
            maximum,
            reference_low=reference_low,
            reference_high=reference_high,
        ),
        "refnorm_median_score": reference_normalize(
            median,
            reference_low=reference_low,
            reference_high=reference_high,
        ),
        "refnorm_mean_score": reference_normalize(
            mean,
            reference_low=reference_low,
            reference_high=reference_high,
        ),
    }


def reference_normalize(
    value: float,
    *,
    reference_low: float,
    reference_high: float,
) -> float:
    width = reference_high - reference_low
    minimum_width = 1e-12 * max(
        1.0,
        abs(reference_low),
        abs(reference_high),
    )
    return 0.0 if width <= minimum_width else (value - reference_low) / width


def _candidate_diagnostics(
    candidates: FloatArray,
    problem: OfflineProblem,
) -> SeedDiagnostics:
    normalized_candidates = _normalized_designs(candidates, problem)
    logged = problem.train_designs.detach().cpu().numpy()
    normalized_logged = _normalized_designs(logged, problem)
    rounded = np.round(candidates, decimals=12)
    unique_count = len(np.unique(rounded, axis=0))
    diagnostics: SeedDiagnostics = {
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
    designs: FloatArray,
    problem: OfflineProblem,
) -> FloatArray:
    designs = np.asarray(designs, dtype=float)
    if isinstance(problem.design_space, BoxSpace):
        bounds = problem.design_space.bounds.detach().cpu().numpy()
        return cast(
            FloatArray, (designs - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        )
    return designs


def _empty_candidate_scores() -> SeedScores:
    """Keep the serialized column order stable for new and failed attempts."""
    return {
        "raw_max_utility": float("nan"),
        "raw_median_utility": float("nan"),
        "raw_mean_utility": float("nan"),
        "refnorm_max_score": float("nan"),
        "refnorm_median_score": float("nan"),
        "refnorm_mean_score": float("nan"),
    }
