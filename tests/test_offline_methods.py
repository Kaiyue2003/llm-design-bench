import numpy as np
import pytest

from llm_design_bench.evaluation.data_manifest import stratified_percentile_mask
from llm_design_bench.evaluation.logged_data import LoggedDatasetView
from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import OfflineProblem, RunContext
from llm_design_bench.types import CandidateBatch
from toy_offline_task import ToyOfflineTask


@pytest.mark.parametrize(
    "method_id,kwargs",
    [
        (
            "offline_mlp",
            {"hidden_size": 16, "epochs": 2, "batch_size": 4, "particle_steps": 2},
        ),
        (
            "coms",
            {
                "hidden_size": 16,
                "epochs": 2,
                "batch_size": 4,
                "adversarial_steps": 2,
                "particle_steps": 2,
            },
        ),
        ("bdi", {"steps": 2}),
    ],
)
def test_offline_method_generates_simplex_before_evaluation(method_id, kwargs) -> None:
    task = ToyOfflineTask()
    problem = OfflineProblem.from_task(task)
    result = make_method(method_id, **kwargs).run(
        problem, RunContext(method_seed=7, candidate_budget=4)
    )

    assert task.predict_calls == 0
    assert result.candidates.shape == (4, task.mixture_dim)
    problem.design_space.validate(result.candidates)
    batch = task.at_target_fidelity(result.candidates.detach().cpu().numpy())
    utility = task.predict(batch)
    assert task.predict_calls == 1
    assert utility.shape == (4,)
    assert np.isfinite(utility).all()
    np.testing.assert_array_equal(batch.model_scales, [1000] * 4)
    np.testing.assert_array_equal(batch.training_steps, [19_500] * 4)


def test_scale_stratified_view_exposes_only_low_outcomes_per_scale() -> None:
    task = ToyOfflineTask()
    reference_utility = task.logged_y.copy()
    logged = task.logged_x
    mask = stratified_percentile_mask(task.logged_y, logged.model_scales)
    view = LoggedDatasetView(
        task,
        CandidateBatch(
            logged.mixtures[mask],
            logged.model_scales[mask],
            logged.training_steps[mask],
        ),
        task.logged_y[mask],
    )
    problem = OfflineProblem.from_task(view)

    assert 0 < problem.sample_count < len(reference_utility)
    assert task.predict_calls == 0
    for scale in np.unique(logged.model_scales):
        threshold = np.percentile(reference_utility[logged.model_scales == scale], 40)
        assert np.all(view.logged_y[view.logged_x.model_scales == scale] <= threshold)
    # The view must not replace the evaluator's full logged reference.
    np.testing.assert_array_equal(task.logged_y, reference_utility)
    assert view.target_model_scale == task.target_model_scale
    assert view.target_training_steps == task.target_training_steps
