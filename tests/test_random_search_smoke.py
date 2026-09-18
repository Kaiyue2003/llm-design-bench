"""Random search obeys the offline candidate-only method boundary."""

import torch
from toy_offline_task import ToyOfflineTask

from llm_design_bench.optimizers import make_method
from llm_design_bench.problem import OfflineProblem, RunContext


def test_random_search_returns_seeded_simplex_candidates_without_queries():
    task = ToyOfflineTask()
    problem = OfflineProblem.from_task(task)
    context = RunContext(method_seed=7, candidate_budget=8)
    first = make_method("random_search").run(problem, context)
    second = make_method("random_search").run(problem, context)
    assert first.candidates.shape == (8, 3)
    assert torch.equal(first.candidates, second.candidates)
    problem.design_space.validate(first.candidates)
    assert task.predict_calls == 0
