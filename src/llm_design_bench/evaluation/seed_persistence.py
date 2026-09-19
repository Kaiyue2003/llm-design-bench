"""Seed-result identities, shard compatibility and atomic progress persistence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from llm_design_bench.evaluation.run_artifacts import (
    array_fingerprint,
    atomic_bytes,
)
from llm_design_bench.evaluation.seed_rendering import (
    render_seed_summary_latex,
    render_seed_summary_markdown,
)
from llm_design_bench.evaluation.seed_statistics import (
    _empty_candidate_scores,
    reference_normalize,
    summarize_seed_results,
)
from llm_design_bench.evaluation.seed_types import (
    RESULT_SCHEMA_VERSION,
    MethodSpec,
    SeedBenchmarkConfig,
    SeedResultRow,
    _json_dumps,
    _update_scores,
)
from llm_design_bench.optimizers.base import MethodMetadata
from llm_design_bench.problem import OfflineProblem
from llm_design_bench.spaces import BoxSpace


def _base_row(
    *,
    spec: MethodSpec,
    metadata: MethodMetadata,
    problem: OfflineProblem,
    config: SeedBenchmarkConfig,
    seed: int,
    reference_low: float,
    reference_high: float,
) -> SeedResultRow:
    row: SeedResultRow = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_source": "unified_runner",
        "run_id": cast(str, spec.run_id),
        "experiment_id": config.experiment_id,
        "method_id": spec.method_id,
        "method_display_name": metadata.display_name,
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
        "requested_method_config_json": _json_dumps(spec.kwargs),
        "method_config_json": _json_dumps(spec.kwargs),
        "resolved_method_config_json": None,
        "package_commit": config.package_commit,
        "provenance_json": _json_dumps(config.provenance),
        "phase": config.phase,
        "required_seeds_json": _json_dumps(config.required_seeds),
        "task_id": config.task_id or problem.metadata.task_name,
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
        "d_best_utility": float(problem.train_utility.max().detach().cpu()),
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
        "environment_json": None,
        "artifact_dir": None,
        "artifact_relative_dir": None,
        "peak_gpu_memory_bytes": None,
        "logical_fingerprint": None,
        "infrastructure_retry_reason": None,
        "failure_stage": None,
        "raw_min_loss": float("nan"),
        "raw_median_loss": float("nan"),
        "raw_mean_loss": float("nan"),
        "unique_candidate_count": float("nan"),
        "unique_candidate_fraction": float("nan"),
        "candidate_diversity": float("nan"),
        "candidate_novelty": float("nan"),
    }
    _update_scores(row, _empty_candidate_scores())
    row["refnorm_d_best_score"] = reference_normalize(
        row["d_best_utility"],
        reference_low=reference_low,
        reference_high=reference_high,
    )
    return row


def _logical_config(
    row: Mapping[str, object],
    problem: OfflineProblem,
    reference: NDArray[np.generic],
) -> dict[str, object]:
    fields = (
        "schema_version",
        "experiment_id",
        "task_id",
        "run_id",
        "method_id",
        "method_config_json",
        "package_commit",
        "provenance_json",
        "phase",
        "required_seeds_json",
        "method_seed",
        "dataset_seed",
        "split_seed",
        "candidate_budget",
        "dtype",
        "device",
        "normalization_reference_id",
        "problem_metadata_json",
        "objective_name",
        "target_context_json",
        "reference_min_utility",
        "reference_max_utility",
    )
    return {
        **{field: row[field] for field in fields},
        "data_fingerprints": {
            field: array_fingerprint(getattr(problem, field))
            for field in (
                "train_designs",
                "train_context",
                "train_utility",
                "target_context",
            )
        },
        "reference_fingerprint": array_fingerprint(reference),
        "design_space": {
            "type": type(problem.design_space).__name__,
            "dimension": problem.design_space.dimension,
            "tolerance": getattr(problem.design_space, "tolerance", None),
            "bounds": (
                array_fingerprint(problem.design_space.bounds)
                if isinstance(problem.design_space, BoxSpace)
                else None
            ),
        },
    }


def merge_result_rows(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    resume: bool,
    infrastructure_retry_reason: str | None,
) -> pd.DataFrame:
    """Merge disjoint shards, replacing an attempted seed only under explicit policy."""
    keys = ["experiment_id", "task_id", "run_id", "method_seed"]
    for frame in (previous, current):
        if frame.duplicated(keys).any():
            raise ValueError("duplicate method/task/seed rows are not allowed")
    for column in ("experiment_id", "phase", "required_seeds_json", "provenance_json"):
        combined = pd.concat([previous, current], ignore_index=True)
        if column not in combined or combined[column].nunique(dropna=False) != 1:
            raise ValueError(f"cannot merge results with incompatible {column}")
    old = previous.set_index(keys, drop=False)
    new = current.set_index(keys, drop=False)
    overlap = old.index.intersection(new.index)
    for key in overlap:
        old_row, new_row = old.loc[key], new.loc[key]
        fingerprint = old_row.get("logical_fingerprint")
        if not isinstance(fingerprint, str) or fingerprint != new_row.get(
            "logical_fingerprint"
        ):
            raise ValueError(
                "refusing to replace a row with a different logical configuration"
            )
        if old_row["status"] == "success":
            if not resume or _artifact_identity(old_row) != _artifact_identity(new_row):
                raise FileExistsError(
                    "successful result requires explicit resume of its saved attempt"
                )
        elif not (
            infrastructure_retry_reason
            or _artifact_identity(old_row) == _artifact_identity(new_row)
        ):
            raise FileExistsError(
                "replacing a failure requires an explicit infrastructure retry reason"
            )
    return pd.concat([old.drop(index=overlap), new], ignore_index=True)


def validate_existing_result_config(config: SeedBenchmarkConfig) -> pd.DataFrame | None:
    """Reject incompatible experiment shards before allocating another trial."""
    path = config.results_dir / "method_seed_results.csv"
    if not config.save_artifacts or not path.exists():
        return None
    previous = pd.read_csv(path)
    expected = {
        "experiment_id": config.experiment_id,
        "phase": config.phase,
        "required_seeds_json": _json_dumps(config.required_seeds),
        "provenance_json": _json_dumps(config.provenance),
    }
    for key, value in expected.items():
        if (
            key not in previous
            or previous[key].nunique(dropna=False) != 1
            or previous.iloc[0][key] != value
        ):
            raise ValueError(f"existing result directory has incompatible {key}")
    return previous


def _validate_existing_method_rows(
    row: Mapping[str, object], previous: pd.DataFrame
) -> None:
    matching = previous[previous["run_id"] == row["run_id"]]
    if matching.empty:
        return
    for column in (
        "method_id",
        "method_config_json",
        "requested_method_config_json",
        "candidate_budget",
        "dtype",
        "device",
        "package_commit",
    ):
        for value in matching[column]:
            expected = row[column]
            if pd.isna(value) and expected is None:
                continue
            if value != expected:
                raise ValueError(
                    f"logical configuration changed: existing run ID has incompatible {column}; "
                    "use a new experiment"
                )


def _artifact_identity(row: Mapping[str, object]) -> object:
    relative = row.get("artifact_relative_dir")
    return relative if isinstance(relative, str) else row.get("artifact_dir")


def _write_seed_progress(
    rows: list[SeedResultRow], config: SeedBenchmarkConfig
) -> None:
    """Publish the public seed reports while the caller holds the directory lock."""
    frame = pd.DataFrame(rows)
    path = config.results_dir / "method_seed_results.csv"
    if config.save_artifacts and path.exists():
        # Existing rows from this invocation are derived from the same immutable
        # attempts; merging them is equivalent to resuming those saved rows.
        frame = merge_result_rows(
            pd.read_csv(path),
            frame,
            resume=True,
            infrastructure_retry_reason=config.infrastructure_retry_reason,
        )
    summary = summarize_seed_results(frame)
    # Prepare every derived representation before publishing any file. Each
    # replacement remains atomic; this is not a transaction across four files.
    reports = {
        "method_seed_results.csv": frame.to_csv(index=False),
        "method_seed_summary.csv": summary.to_csv(index=False),
        "method_seed_table.md": render_seed_summary_markdown(summary),
        "method_seed_table.tex": render_seed_summary_latex(summary),
    }
    for filename, content in reports.items():
        atomic_bytes(
            config.results_dir / filename, content.encode("utf-8"), replace=True
        )
