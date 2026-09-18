"""Execution orchestration for paired method seeds.

Configuration/record types, pure statistics and persistence each have their own
module. Re-exports retain the existing import surface for callers.
"""

from __future__ import annotations

import inspect
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from llm_design_bench.evaluation.run_artifacts import (
    RunAttempt,
    atomic_npz,
    capture_environment,
    file_sha256,
    result_directory_lock,
)
from llm_design_bench.evaluation.seed_persistence import (
    _artifact_identity as _artifact_identity,
    _base_row as _base_row,
    _logical_config as _logical_config,
    _validate_existing_method_rows as _validate_existing_method_rows,
    _write_seed_progress as _write_seed_progress,
    merge_result_rows as merge_result_rows,
    validate_existing_result_config as validate_existing_result_config,
)
from llm_design_bench.evaluation.seed_statistics import (
    _SCORE_COLUMNS as _SCORE_COLUMNS,
    _SUMMARY_METRICS as _SUMMARY_METRICS,
    _candidate_diagnostics as _candidate_diagnostics,
    _candidate_scores as _candidate_scores,
    _empty_candidate_scores,
    _normalized_designs as _normalized_designs,
    reference_normalize as reference_normalize,
    summarize_seed_results as summarize_seed_results,
)
from llm_design_bench.evaluation.seed_types import (
    DEFAULT_METHOD_SEEDS as DEFAULT_METHOD_SEEDS,
    DEFAULT_SEED_BENCHMARK_CONFIG as DEFAULT_SEED_BENCHMARK_CONFIG,
    RESULT_SCHEMA_VERSION as RESULT_SCHEMA_VERSION,
    FloatArray,
    MethodSpec as MethodSpec,
    SeedBenchmarkConfig as SeedBenchmarkConfig,
    SeedBenchmarkResult as SeedBenchmarkResult,
    SeedEvaluator,
    SeedResultRow,
    _json_dumps as _json_dumps,
    _update_diagnostics,
    _update_scores,
    _validate_dtype as _validate_dtype,
)
from llm_design_bench.optimizers.base import MethodCapabilities, OfflineBBOMethod
from llm_design_bench.optimizers.registry import (
    get_method_capabilities,
    get_method_metadata,
    make_method,
)
from llm_design_bench.problem import MethodResult, OfflineProblem, RunContext
from llm_design_bench.spaces import BoxSpace, SimplexSpace

_Batch = TypeVar("_Batch")


def run_method_seed_benchmark(
    evaluator_task: SeedEvaluator[_Batch],
    problem: OfflineProblem,
    methods: Sequence[MethodSpec | str],
    *,
    reference_utility: NDArray[np.generic],
    config: SeedBenchmarkConfig = DEFAULT_SEED_BENCHMARK_CONFIG,
    write_results: bool = True,
) -> SeedBenchmarkResult:
    """Reserve the output directory whenever this runner writes aggregate CSVs."""
    if write_results:
        with result_directory_lock(config.results_dir):
            return _run_method_seed_benchmark_unlocked(
                evaluator_task,
                problem,
                methods,
                reference_utility=reference_utility,
                config=config,
                write_results=write_results,
            )
    return _run_method_seed_benchmark_unlocked(
        evaluator_task,
        problem,
        methods,
        reference_utility=reference_utility,
        config=config,
        write_results=write_results,
    )


def _run_method_seed_benchmark_unlocked(
    evaluator_task: SeedEvaluator[_Batch],
    problem: OfflineProblem,
    methods: Sequence[MethodSpec | str],
    *,
    reference_utility: NDArray[np.generic],
    config: SeedBenchmarkConfig,
    write_results: bool,
) -> SeedBenchmarkResult:
    """Validate, dispatch paired seeds and summarize their completed attempts.

    Only the offline problem reaches each method; the evaluator and reference
    remain owned by the attempt lifecycle, after ordered candidate persistence.
    """
    specs = _normalize_method_specs(methods)
    _validate_specs(specs)
    reference = _validate_reference(reference_utility)
    reference_low = float(reference.min())
    reference_high = float(reference.max())
    existing_results = validate_existing_result_config(config)

    rows: list[SeedResultRow] = []
    for spec in specs:
        method_config = replace(config, dtype=spec.dtype or config.dtype)
        metadata = get_method_metadata(spec.method_id)
        capabilities = get_method_capabilities(spec.method_id)
        _validate_method_compatibility(
            spec.method_id,
            capabilities,
            problem,
            method_config,
        )
        for seed in config.seeds:
            row = _base_row(
                spec=spec,
                metadata=metadata,
                problem=problem,
                config=method_config,
                seed=seed,
                reference_low=reference_low,
                reference_high=reference_high,
            )
            _run_seed_attempt(
                evaluator_task,
                problem,
                spec,
                row,
                reference=reference,
                config=config,
                method_config=method_config,
                existing_results=existing_results,
                rows=rows,
                write_results=write_results,
            )

    per_seed = pd.DataFrame(rows)
    summary = summarize_seed_results(per_seed)
    if write_results:
        _write_seed_progress(rows, config)
    return SeedBenchmarkResult(per_seed=per_seed, summary=summary)


def _run_seed_attempt(
    evaluator_task: SeedEvaluator[_Batch],
    problem: OfflineProblem,
    spec: MethodSpec,
    row: SeedResultRow,
    *,
    reference: NDArray[np.generic],
    config: SeedBenchmarkConfig,
    method_config: SeedBenchmarkConfig,
    existing_results: pd.DataFrame | None,
    rows: list[SeedResultRow],
    write_results: bool,
) -> None:
    """Own one seed's recovery, failure stages and unconditional finalization."""
    _seed_everything(row["method_seed"])
    method, construction_error = _construct_seed_method(spec, row)
    if existing_results is not None:
        _validate_existing_method_rows(row, existing_results)
    attempt = _begin_seed_attempt(row, spec, problem, reference, config)
    pending_error: BaseException | None = None
    try:
        if attempt is not None:
            if attempt.previous_result is not None:
                # Verified artifacts from this runner are restored without training,
                # evaluation or rewriting the immutable successful attempt.
                restored = cast(SeedResultRow, attempt.previous_result.copy())
                restored["artifact_dir"] = str(attempt.path.resolve())
                restored["artifact_relative_dir"] = attempt.path.relative_to(
                    config.results_dir
                ).as_posix()
                rows.append(restored)
                attempt.close()
                return
            row["artifact_dir"] = str(attempt.path.resolve())
            row["artifact_relative_dir"] = attempt.path.relative_to(
                config.results_dir
            ).as_posix()
            row["logical_fingerprint"] = attempt.fingerprint
            row["infrastructure_retry_reason"] = config.infrastructure_retry_reason
        started = time.perf_counter()
        stage = "construction"
        try:
            if construction_error is not None:
                raise construction_error
            assert method is not None
            context = RunContext(
                method_seed=row["method_seed"],
                candidate_budget=config.candidate_budget,
                device=config.device,
                dtype=method_config.dtype,
                dataset_seed=config.dataset_seed,
                split_seed=config.split_seed,
            )
            stage = "method"
            method_result = _run_timed_method(method, problem, context, row)
            stage = "candidate_persistence"
            candidates = _persist_candidates(method_result, problem, attempt, row)
            # No evaluator call, including target-fidelity conversion, occurs until
            # the exact ordered candidate batch is safely committed above.
            stage = "evaluation"
            utilities, negative_loss = _evaluate_candidates(
                evaluator_task, candidates, problem, attempt, row, config
            )
            stage = "diagnostics"
            _record_candidate_diagnostics(
                candidates, utilities, problem, row, negative_loss
            )
            row["status"] = "success"
        except (Exception, KeyboardInterrupt) as exc:
            row["status"] = "failed"
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
            row["failure_stage"] = stage
            _update_scores(row, _empty_candidate_scores())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if config.fail_fast or isinstance(exc, KeyboardInterrupt):
                pending_error = exc
                raise
        except BaseException as exc:
            # Track unclassified exits only for cleanup; leave their row fields
            # and propagation unchanged instead of treating them as method errors.
            pending_error = exc
            raise
        finally:
            # This must run before an exception escapes to the outer seed loop:
            # fail-fast and interrupts still commit their failure row and progress.
            _finish_seed_attempt(
                row,
                attempt,
                rows,
                config,
                started=started,
                write_results=write_results,
                pending_error=pending_error,
            )
    except BaseException as exc:
        # Resource cleanup must also cover setup/resume and unhandled interrupts;
        # this does not classify or suppress any execution failure.
        pending_error = exc
        raise
    finally:
        _close_seed_attempt(attempt, pending_error)


def _close_seed_attempt(
    attempt: RunAttempt | None, pending_error: BaseException | None
) -> None:
    """Release an owned attempt without replacing an exception already escaping."""
    if attempt is not None:
        attempt.close_preserving_error(pending_error)


def _construct_seed_method(
    spec: MethodSpec, row: SeedResultRow
) -> tuple[OfflineBBOMethod | None, Exception | None]:
    method = None
    construction_error = None
    try:
        method = make_method(spec.method_id, **spec.kwargs)
        row["method_config_json"] = _json_dumps(
            resolved_method_config(method, spec.kwargs)
        )
    except Exception as exc:  # noqa: BLE001 -- Persist arbitrary method construction failures.
        construction_error = exc
    return method, construction_error


def _begin_seed_attempt(
    row: SeedResultRow,
    spec: MethodSpec,
    problem: OfflineProblem,
    reference: NDArray[np.generic],
    config: SeedBenchmarkConfig,
) -> RunAttempt | None:
    if not config.save_artifacts:
        return None
    environment = capture_environment(cast(torch.device, config.device))
    row["environment_json"] = _json_dumps(environment)
    return RunAttempt.begin(
        config.results_dir,
        experiment_id=config.experiment_id,
        task_id=row["task_id"],
        run_id=str(spec.run_id),
        seed=row["method_seed"],
        logical_config=_logical_config(row, problem, reference),
        environment=environment,
        resume=config.resume,
        infrastructure_retry_reason=config.infrastructure_retry_reason,
    )


def _run_timed_method(
    method: OfflineBBOMethod,
    problem: OfflineProblem,
    context: RunContext,
    row: SeedResultRow,
) -> MethodResult:
    if cast(torch.device, context.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    method_started = time.perf_counter()
    method_result = method.run(problem, context)
    if cast(torch.device, context.device).type == "cuda":
        torch.cuda.synchronize(context.device)
    row["method_seconds"] = time.perf_counter() - method_started
    return method_result


def _persist_candidates(
    method_result: MethodResult,
    problem: OfflineProblem,
    attempt: RunAttempt | None,
    row: SeedResultRow,
) -> FloatArray:
    candidates = method_result.candidates.detach().cpu().numpy().copy()
    row["training_summary_json"] = _json_dumps(method_result.training_summary)
    row["diagnostics_json"] = _json_dumps(method_result.diagnostics)
    if attempt is not None:
        atomic_npz(
            attempt.path / "candidates.npz",
            candidates=candidates,
            target_context=problem.target_context.detach().cpu().numpy(),
        )
    return candidates


def _evaluate_candidates(
    evaluator_task: SeedEvaluator[_Batch],
    candidates: FloatArray,
    problem: OfflineProblem,
    attempt: RunAttempt | None,
    row: SeedResultRow,
    config: SeedBenchmarkConfig,
) -> tuple[NDArray[np.float64], bool]:
    evaluation_started = time.perf_counter()
    candidate_batch = evaluator_task.at_target_fidelity(candidates)
    utilities = np.asarray(evaluator_task.predict(candidate_batch), dtype=float)
    _validate_oracle_output(utilities, expected=config.candidate_budget)
    row["evaluation_seconds"] = time.perf_counter() - evaluation_started
    negative_loss = problem.metadata.extra.get("utility_transform") == "negative_loss"
    if attempt is not None:
        evaluation_arrays = {
            "utility": utilities,
            "refnorm_score": np.asarray(
                [
                    reference_normalize(
                        value,
                        reference_low=row["reference_min_utility"],
                        reference_high=row["reference_max_utility"],
                    )
                    for value in utilities
                ]
            ),
        }
        if negative_loss:
            evaluation_arrays["raw_loss"] = -utilities
        atomic_npz(attempt.path / "evaluation.npz", **evaluation_arrays)
    return utilities, negative_loss


def _record_candidate_diagnostics(
    candidates: FloatArray,
    utilities: NDArray[np.float64],
    problem: OfflineProblem,
    row: SeedResultRow,
    negative_loss: bool,
) -> None:
    _update_scores(
        row,
        _candidate_scores(
            utilities,
            reference_low=row["reference_min_utility"],
            reference_high=row["reference_max_utility"],
        ),
    )
    _update_diagnostics(row, _candidate_diagnostics(candidates, problem))
    if negative_loss:
        row.update(
            {
                "raw_min_loss": float((-utilities).min()),
                "raw_median_loss": float(np.median(-utilities)),
                "raw_mean_loss": float((-utilities).mean()),
            }
        )


def _finish_seed_attempt(
    row: SeedResultRow,
    attempt: RunAttempt | None,
    rows: list[SeedResultRow],
    config: SeedBenchmarkConfig,
    *,
    started: float,
    write_results: bool,
    pending_error: BaseException | None = None,
) -> None:
    row["total_seconds"] = time.perf_counter() - started
    if cast(torch.device, config.device).type == "cuda":
        row["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(config.device)
    rows.append(row)
    if attempt is not None:
        row["artifact_sha256"] = {
            name: file_sha256(attempt.path / name)
            for name in ("candidates.npz", "evaluation.npz")
            if (attempt.path / name).is_file()
        }
        attempt.finish(row, pending_error=pending_error)
    if write_results:
        _write_seed_progress(rows, config)


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
    capabilities: MethodCapabilities,
    problem: OfflineProblem,
    config: SeedBenchmarkConfig,
) -> None:
    if (
        isinstance(problem.design_space, SimplexSpace)
        and not capabilities.supports_simplex
    ):
        raise ValueError(f"method {method_id!r} does not support simplex designs")
    if isinstance(problem.design_space, BoxSpace) and not capabilities.supports_box:
        raise ValueError(f"method {method_id!r} does not support box designs")
    if capabilities.requires_gpu and cast(torch.device, config.device).type != "cuda":
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


def _validate_reference(values: NDArray[np.generic]) -> NDArray[np.generic]:
    reference = np.asarray(values, dtype=float).reshape(-1)
    if reference.size == 0:
        raise ValueError("reference_utility must not be empty")
    if not np.isfinite(reference).all():
        raise ValueError("reference_utility must be finite")
    return reference


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_oracle_output(utilities: NDArray[np.generic], expected: int) -> None:
    if utilities.shape != (expected,):
        raise ValueError(
            f"oracle returned utility shape {utilities.shape}; expected {(expected,)}"
        )
    if not np.isfinite(utilities).all():
        raise ValueError("oracle returned non-finite utilities")


def resolved_method_config(
    method: object,
    requested: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture constructor defaults without serializing learned model state."""

    resolved = dict(requested)
    try:
        signature = inspect.signature(type(method).__init__)
    except (TypeError, ValueError):
        return resolved
    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if hasattr(method, name):
            resolved[name] = getattr(method, name)
        elif parameter.default is not inspect.Parameter.empty:
            resolved.setdefault(name, parameter.default)
    return resolved


# Keep the private spelling available to older local callers.
_resolved_method_config = resolved_method_config
