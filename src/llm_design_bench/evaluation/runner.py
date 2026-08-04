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
from llm_design_bench.optimizers.best_logged import BestLoggedOptimizer
from llm_design_bench.optimizers.random_search import DirichletRandomSearch
from llm_design_bench.optimizers.sobol_search import SobolSearch


@dataclass(frozen=True)
class BenchmarkConfig:
    queries: int = 256
    reference_queries: int = 2048
    recommendations: int = 128
    seed: int = 38
    results_dir: Path = Path("results")


def run_baselines(task, config: BenchmarkConfig = BenchmarkConfig()) -> pd.DataFrame:
    reference = DirichletRandomSearch(
        queries=config.reference_queries,
        recommendations=1,
        seed=config.seed + 10_000,
    ).optimize(task)
    oracle_best = float(reference.query_utility.max())
    target_utility = float(np.quantile(reference.query_utility, 0.95))
    optimizers = (
        BestLoggedOptimizer(recommendations=config.recommendations),
        DirichletRandomSearch(
            queries=config.queries,
            recommendations=config.recommendations,
            seed=config.seed,
        ),
        SobolSearch(
            queries=config.queries,
            recommendations=config.recommendations,
            seed=config.seed,
        ),
    )

    rows = []
    for optimizer in optimizers:
        trace = optimizer.optimize(task)
        row = {
            "optimizer": trace.name,
            "query_count": len(trace.queried),
            "recommendation_count": len(trace.recommendations),
            "cumulative_simulated_cost": trace.cumulative_cost,
            "target_utility": target_utility,
            "cost_to_target_utility": _cost_to_reach(trace, target_utility),
        }
        row.update(usefulness_summary(trace.recommendation_utility, task.logged_y, oracle_best))
        row.update(mixture_diagnostics(trace.recommendations.mixtures))
        row["candidate_novelty"] = candidate_novelty(
            trace.recommendations.mixtures,
            task.logged_x.mixtures,
        )
        rows.append(row)

    frame = add_reference_normalized_score_columns(pd.DataFrame(rows))
    config.results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.results_dir / "baseline_results.csv", index=False)
    save_summary_plot(frame, config.results_dir / "baseline_summary.png")
    return frame


def _cost_to_reach(trace, target_utility: float) -> float:
    if len(trace.queried) == 0:
        return float("nan")
    hits = np.flatnonzero(trace.query_utility >= target_utility)
    if len(hits) == 0:
        return float("nan")
    return float(np.cumsum(trace.query_cost)[hits[0]])
