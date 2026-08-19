from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from llm_design_bench.evaluation.report import save_summary_plot
from llm_design_bench.metrics.diversity import mixture_diagnostics
from llm_design_bench.metrics.novelty import candidate_novelty
from llm_design_bench.metrics.usefulness import (
    add_reference_normalized_score_columns,
    usefulness_summary,
)
from llm_design_bench.optimizers.bdi import BackwardDistillationOptimizer
from llm_design_bench.optimizers.best_logged import BestLoggedOptimizer
from llm_design_bench.optimizers.coms import ConservativeObjectiveModelOptimizer
from llm_design_bench.optimizers.mlp_surrogate import OfflineMLPOptimizer
from llm_design_bench.optimizers.random_search import DirichletRandomSearch
from llm_design_bench.types import CandidateBatch


@dataclass(frozen=True)
class OfflineBenchmarkConfig:
    reference_queries: int = 2048
    recommendations: int = 128
    seed: int = 38
    epochs: int = 100
    particle_steps: int = 100
    bdi_steps: int = 100
    train_min_percentile: float = 0.0
    train_max_percentile: float = 40.0
    results_dir: Path = Path("results")


@dataclass(frozen=True)
class LoggedPercentileSplit:
    task: object
    reference_y: np.ndarray
    train_min_percentile: float
    train_max_percentile: float
    train_size: int
    train_min_utility: float
    train_max_utility: float
    d_best_utility: float
    refnorm_d_best_score: float


class LoggedDatasetView:
    def __init__(self, task, logged_x: CandidateBatch, logged_y: np.ndarray) -> None:
        self._task = task
        self._logged_x = logged_x
        self._logged_y = logged_y

    @property
    def logged_x(self) -> CandidateBatch:
        return self._logged_x

    @property
    def logged_y(self) -> np.ndarray:
        return self._logged_y

    def __getattr__(self, name: str):
        return getattr(self._task, name)


def run_offline_benchmarks(
    task,
    config: OfflineBenchmarkConfig = OfflineBenchmarkConfig(),
    optimizers=None,
) -> pd.DataFrame:
    split = make_logged_percentile_split(
        task,
        min_percentile=config.train_min_percentile,
        max_percentile=config.train_max_percentile,
    )
    reference = DirichletRandomSearch(
        queries=config.reference_queries,
        recommendations=1,
        seed=config.seed + 10_000,
    ).optimize(task)
    oracle_best = float(reference.query_utility.max())
    if optimizers is None:
        optimizers = (
            BestLoggedOptimizer(recommendations=config.recommendations),
            OfflineMLPOptimizer(
                recommendations=config.recommendations,
                seed=config.seed,
                epochs=config.epochs,
                particle_steps=config.particle_steps,
            ),
            ConservativeObjectiveModelOptimizer(
                recommendations=config.recommendations,
                seed=config.seed,
                epochs=config.epochs,
                particle_steps=config.particle_steps,
            ),
            BackwardDistillationOptimizer(
                recommendations=config.recommendations,
                seed=config.seed,
                steps=config.bdi_steps,
            ),
        )

    rows = []
    for optimizer in optimizers:
        trace = optimizer.optimize(split.task)
        row = {
            "optimizer": trace.name,
            "mode": "offline",
            "train_min_percentile": split.train_min_percentile,
            "train_max_percentile": split.train_max_percentile,
            "train_size": split.train_size,
            "train_min_utility": split.train_min_utility,
            "train_max_utility": split.train_max_utility,
            "d_best_utility": split.d_best_utility,
            "refnorm_d_best_score": split.refnorm_d_best_score,
            "query_count": len(trace.queried),
            "recommendation_count": len(trace.recommendations),
            "cumulative_simulated_cost": trace.cumulative_cost,
        }
        row.update(usefulness_summary(trace.recommendation_utility, split.reference_y, oracle_best))
        row.update(mixture_diagnostics(trace.recommendations.mixtures))
        row["candidate_novelty"] = candidate_novelty(
            trace.recommendations.mixtures,
            split.task.logged_x.mixtures,
        )
        rows.append(row)

    frame = add_reference_normalized_score_columns(pd.DataFrame(rows))
    config.results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.results_dir / "offline_results.csv", index=False)
    save_summary_plot(frame, config.results_dir / "offline_summary.png")
    return frame


def make_logged_percentile_split(
    task,
    min_percentile: float = 0.0,
    max_percentile: float = 40.0,
) -> LoggedPercentileSplit:
    if not (0.0 <= min_percentile <= max_percentile <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= min <= max <= 100")

    reference_y = np.asarray(task.logged_y, dtype=float)
    lower = np.percentile(reference_y, min_percentile)
    upper = np.percentile(reference_y, max_percentile)
    if min_percentile == max_percentile:
        mask = np.isclose(reference_y, lower)
    else:
        mask = (reference_y >= lower) & (reference_y <= upper)
    if not mask.any():
        raise ValueError(
            f"logged percentile split [{min_percentile}, {max_percentile}] is empty"
        )

    logged_x = CandidateBatch(
        mixtures=task.logged_x.mixtures[mask],
        model_scales=task.logged_x.model_scales[mask],
        training_steps=task.logged_x.training_steps[mask],
    )
    logged_y = reference_y[mask]
    reference_min = float(reference_y.min())
    reference_max = float(reference_y.max())
    reference_width = reference_max - reference_min
    d_best = float(logged_y.max())
    if reference_width == 0:
        refnorm_d_best_score = 0.0
    else:
        refnorm_d_best_score = (d_best - reference_min) / reference_width

    return LoggedPercentileSplit(
        task=LoggedDatasetView(task, logged_x, logged_y),
        reference_y=reference_y,
        train_min_percentile=min_percentile,
        train_max_percentile=max_percentile,
        train_size=int(mask.sum()),
        train_min_utility=float(logged_y.min()),
        train_max_utility=float(logged_y.max()),
        d_best_utility=d_best,
        refnorm_d_best_score=float(refnorm_d_best_score),
    )
