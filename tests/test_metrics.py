import numpy as np
import pandas as pd

from llm_design_bench.metrics.diversity import (
    active_domain_count,
    mixture_entropy,
    pairwise_diversity,
)
from llm_design_bench.metrics.novelty import candidate_novelty
from llm_design_bench.metrics.usefulness import (
    add_reference_normalized_score_columns,
    usefulness_summary,
)


def test_usefulness_summary() -> None:
    summary = usefulness_summary(np.array([1.0, 2.0, 3.0]), np.array([0.0, 4.0]), 3.5)
    assert summary["raw_max_utility"] == 3.0
    assert summary["raw_median_utility"] == 2.0
    assert summary["refnorm_max_score"] == 0.75
    assert summary["regret"] == 0.5


def test_report_scores_use_reference_normalization() -> None:
    rows = [
        usefulness_summary(np.array([2.0, 1.0, 0.0]), np.array([0.0, 4.0]), 3.5),
        usefulness_summary(np.array([3.0, 1.0, 0.0]), np.array([0.0, 4.0]), 3.5),
    ]
    frame = add_reference_normalized_score_columns(pd.DataFrame(rows))
    assert np.allclose(frame["refnorm_max_score"], [0.5, 0.75])
    assert np.all(frame["refnorm_max_score"] >= frame["refnorm_median_score"])


def test_reference_normalization_treats_flat_reference_as_degenerate() -> None:
    summary = usefulness_summary(
        np.array([1e-30, 2e-30]),
        np.array([0.0, 1e-65]),
        2e-30,
    )
    assert summary["refnorm_max_score"] == 0.0
    assert summary["refnorm_median_score"] == 0.0


def test_diagnostic_metrics() -> None:
    mixtures = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(pairwise_diversity(mixtures), np.sqrt(2))
    assert np.allclose(mixture_entropy(mixtures), 0.0)
    assert np.array_equal(active_domain_count(mixtures), [1, 1])
    assert candidate_novelty(mixtures, mixtures) == 0.0
