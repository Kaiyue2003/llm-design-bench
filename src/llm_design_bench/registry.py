from collections.abc import Callable
from typing import Any

from llm_design_bench.tasks.base import Task
from llm_design_bench.tasks.data_recipes import DataRecipesTask
from llm_design_bench.tasks.synthetic_functions import (
    DEFAULT_SYNTHETIC_FUNCTIONS,
    SyntheticFunctionTask,
)

TaskFactory = Callable[..., Task]

_TASKS: dict[str, TaskFactory] = {
    "data-recipes": DataRecipesTask,
    "data-recipes-stack-exchange": DataRecipesTask,
    "data-recipes-1b": lambda **kwargs: DataRecipesTask(
        logged_model_scale=1000,
        **kwargs,
    ),
}
_TASKS.update(
    {
        f"synthetic-{name}": (
            lambda function_name=name, **kwargs: SyntheticFunctionTask(
                function_name, **kwargs
            )
        )
        for name in DEFAULT_SYNTHETIC_FUNCTIONS
    }
)


def make(name: str, **kwargs: Any) -> Task:
    try:
        factory = _TASKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TASKS))
        raise KeyError(f"unknown task {name!r}; available tasks: {available}") from exc
    return factory(**kwargs)
