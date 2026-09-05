from __future__ import annotations

from pathlib import Path

from llm_design_bench.evaluation.offline_runner import make_logged_percentile_split
from llm_design_bench.evaluation.unified_report import (
    BenchmarkTaskSpec,
    BenchmarkTrial,
)
from llm_design_bench.problem import OfflineProblem, ProblemMetadata
from llm_design_bench.tasks.data_recipes import MODEL_SCALE_LABELS, DataRecipesTask
from llm_design_bench.tasks.synthetic_functions import (
    CATEGORY_DISPLAY_NAMES,
    SYNTHETIC_FUNCTIONS,
    SyntheticFunctionTask,
)


def make_synthetic_task_spec(
    function_name: str,
    *,
    logged_samples: int = 256,
) -> BenchmarkTaskSpec:
    """Build a task spec whose logged dataset is paired with the method seed."""

    if logged_samples < 1:
        raise ValueError("logged_samples must be positive")
    try:
        function_spec = SYNTHETIC_FUNCTIONS[function_name]
    except KeyError as exc:
        available = ", ".join(sorted(SYNTHETIC_FUNCTIONS))
        raise KeyError(
            f"unknown synthetic function {function_name!r}; available: {available}"
        ) from exc

    def trial_factory(seed: int) -> BenchmarkTrial:
        task = SyntheticFunctionTask(
            function_name,
            logged_samples=logged_samples,
            seed=seed,
        )
        problem = OfflineProblem.from_task(
            task,
            metadata=ProblemMetadata(
                task_name=task.name,
                objective_name=task.name,
                source=task.spec.source,
                extra={
                    "logged_samples": logged_samples,
                    "utility_transform": "negative_objective",
                },
            ),
        )
        return BenchmarkTrial(
            evaluator_task=task,
            problem=problem,
            reference_utility=task.logged_y,
            dataset_seed=seed,
            d_best_utility=float(task.logged_y.max()),
        )

    return BenchmarkTaskSpec(
        task_id=function_spec.name,
        display_name=function_spec.display_name,
        suite="synthetic",
        category=function_spec.category,
        category_display_name=CATEGORY_DISPLAY_NAMES[function_spec.category],
        trial_factory=trial_factory,
        normalization_reference_id=(
            f"synthetic_{function_spec.name}_paired_logged"
        ),
    )


def make_data_recipes_task_spec(
    *,
    data_recipes_root: str | Path | None = None,
    metric_index: int = 4,
    logged_model_scale: float | None = None,
    train_min_percentile: float = 0.0,
    train_max_percentile: float = 40.0,
) -> BenchmarkTaskSpec:
    """Build the primary multi-scale or fixed-scale LLM-DM task spec.

    Both settings normalize against the same unfiltered logged reference for
    the chosen metric. The fixed-scale filter changes only optimizer-visible
    data, not the evaluation denominator or target fidelity.
    """

    full_task = DataRecipesTask(
        data_recipes_root=data_recipes_root,
        metric_index=metric_index,
    )
    visible_task = full_task
    if logged_model_scale is not None:
        visible_task = DataRecipesTask(
            data_recipes_root=full_task.root,
            metric_index=metric_index,
            logged_model_scale=logged_model_scale,
        )
    split = make_logged_percentile_split(
        visible_task,
        min_percentile=train_min_percentile,
        max_percentile=train_max_percentile,
    )
    setting = "multi_scale" if logged_model_scale is None else "fixed_scale"
    metric_key = visible_task.metric.name.removesuffix("_cross_entropy")
    task_id = f"data_recipes_{metric_key}"
    display_name = "LLM-DM"
    if metric_index != 4:
        display_name += f" ({visible_task.metric.name})"
    if logged_model_scale is not None:
        scale_label = MODEL_SCALE_LABELS[int(logged_model_scale)]
        task_id += f"_{scale_label.lower()}"
        display_name += f" ({scale_label} logged)"
    problem = OfflineProblem.from_task(
        split.task,
        metadata=ProblemMetadata(
            task_name=task_id,
            objective_name=visible_task.metric.name,
            source="data-recipes",
            extra={
                "setting": setting,
                "logged_model_scale": logged_model_scale,
                "train_min_percentile": train_min_percentile,
                "train_max_percentile": train_max_percentile,
                "target_model_scale": visible_task.target_model_scale,
                "target_training_steps": visible_task.target_training_steps,
                "utility_transform": (
                    "identity" if visible_task.metric.maximize else "negative_loss"
                ),
            },
        ),
    )
    trial = BenchmarkTrial(
        evaluator_task=full_task,
        problem=problem,
        reference_utility=full_task.logged_y,
        dataset_seed=None,
        split_seed=None,
        d_best_utility=split.d_best_utility,
    )

    def trial_factory(seed: int) -> BenchmarkTrial:
        del seed
        return trial

    return BenchmarkTaskSpec(
        task_id=task_id,
        display_name=display_name,
        suite="data_mixture",
        category="data_mixture",
        category_display_name="Data Mixture",
        trial_factory=trial_factory,
        normalization_reference_id=f"data_recipes_metric_{metric_index}_full_logged",
    )
