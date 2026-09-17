import numpy as np

from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import OfflineProblem, RunContext
from toy_offline_task import ToyOfflineTask


def test_random_search_returns_unevaluated_simplex_candidates() -> None:
    task = ToyOfflineTask()
    result = make_method("random_search").run(
        OfflineProblem.from_task(task),
        RunContext(method_seed=7, candidate_budget=8),
    )

    assert task.predict_calls == 0
    designs = result.candidates.detach().cpu().numpy()
    assert designs.shape == (8, task.mixture_dim)
    np.testing.assert_allclose(designs.sum(axis=1), 1.0, atol=1e-6)
    assert (designs >= 0).all()
    batch = task.at_target_fidelity(designs)
    assert np.isfinite(task.predict(batch)).all()
    assert task.predict_calls == 1
