from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from llm_design_bench.optimizers.base import MethodFamily


class IntegrationStatus(str, Enum):
    PLANNED = "planned"
    PORT_IN_PROGRESS = "port_in_progress"
    PARITY_VALIDATED = "parity_validated"
    IMPLEMENTED_ADAPTATION = "implemented_adaptation"


@dataclass(frozen=True)
class MethodBlueprint:
    """Integration requirements and implementation status for a method."""

    method_id: str
    display_name: str
    family: MethodFamily
    mechanism: str
    required_components: tuple[str, ...]
    paper_url: str
    source_url: str | None = None
    original_framework: str | None = None
    context_strategy: str = "concatenate transformed context to every conditional model"
    simplex_strategy: str = "train in additive log-ratio coordinates; decode before evaluation"
    box_strategy: str = "train in unit-box coordinates; decode before evaluation"
    status: IntegrationStatus = IntegrationStatus.PLANNED


_BLUEPRINTS = {
    blueprint.method_id: blueprint
    for blueprint in (
        MethodBlueprint(
            method_id="cbas",
            display_name="CbAS",
            family=MethodFamily.INVERSE_GENERATIVE,
            mechanism="iterative adaptive-threshold conditioning with density-ratio weights",
            required_components=(
                "conditional VAE",
                "probabilistic forward ensemble",
                "adaptive quantile schedule",
                "log-density ratio",
            ),
            paper_url="https://proceedings.mlr.press/v97/brookes19a.html",
            source_url="https://github.com/dhbrookes/CbAS",
            original_framework="TensorFlow",
        ),
        MethodBlueprint(
            method_id="mins",
            display_name="MINs",
            family=MethodFamily.INVERSE_GENERATIVE,
            mechanism="conditional inverse generator trained to model designs given utility",
            required_components=(
                "conditional generator",
                "discriminator",
                "target-utility search",
            ),
            paper_url=(
                "https://proceedings.neurips.cc/paper/2020/hash/"
                "373e4c5d8edfa8b74fd4b6791d0cf6dc-Abstract.html"
            ),
            source_url="https://github.com/rail-berkeley/design-baselines",
            original_framework="TensorFlow",
        ),
        MethodBlueprint(
            method_id="ddom",
            display_name="DDOM",
            family=MethodFamily.INVERSE_GENERATIVE,
            mechanism="classifier-free diffusion of designs conditioned on utility",
            required_components=(
                "design diffusion process",
                "utility conditioning",
                "classifier-free guidance",
            ),
            paper_url="https://proceedings.mlr.press/v202/krishnamoorthy23a.html",
            source_url="https://github.com/siddarthk97/ddom",
            original_framework="PyTorch",
        ),
        MethodBlueprint(
            method_id="gabo",
            display_name="GABO",
            family=MethodFamily.HYBRID,
            mechanism="Bayesian optimization in GAN latent space with source-critic regularization",
            required_components=(
                "conditional VAE latent representation",
                "source critic",
                "latent Gaussian process",
                "acquisition optimizer",
            ),
            paper_url="https://openreview.net/forum?id=3RxcarQFRn",
            source_url="https://github.com/michael-s-yao/gabo",
            original_framework="PyTorch",
        ),
        MethodBlueprint(
            method_id="gtg",
            display_name="GTG",
            family=MethodFamily.TRAJECTORY_MODEL,
            mechanism="guided diffusion over synthetic improvement trajectories",
            required_components=(
                "trajectory constructor",
                "conditional diffusion model",
                "return guidance",
            ),
            paper_url=(
                "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
                "98904db124b2a2463e8c59ec33fc7150-Abstract-Conference.html"
            ),
            source_url="https://github.com/dbsxodud-11/GTG",
            original_framework="PyTorch",
        ),
        MethodBlueprint(
            method_id="rgd",
            display_name="RGD",
            family=MethodFamily.INVERSE_GENERATIVE,
            mechanism="proxy-guided design diffusion with iterative proxy refinement",
            required_components=(
                "design diffusion process",
                "surrogate ensemble",
                "robust guidance",
                "proxy refinement",
            ),
            paper_url="https://arxiv.org/abs/2410.00983",
            source_url="https://github.com/GGchen1997/RGD",
            original_framework="PyTorch",
        ),
        MethodBlueprint(
            method_id="bonet",
            display_name="BONET",
            family=MethodFamily.TRAJECTORY_MODEL,
            mechanism="autoregressive generation from regret-budget-conditioned trajectories",
            required_components=(
                "trajectory constructor",
                "autoregressive transformer",
                "regret-budget conditioning",
            ),
            paper_url="https://proceedings.mlr.press/v202/mashkaria23a.html",
            source_url="https://github.com/siddarthk97/bonet",
            original_framework="PyTorch",
        ),
        MethodBlueprint(
            method_id="demo",
            display_name="DEMO",
            family=MethodFamily.INVERSE_GENERATIVE,
            mechanism="diffusion-based design editing toward higher target utility",
            required_components=(
                "design diffusion process",
                "editing direction",
                "utility-conditioned sampler",
            ),
            paper_url="https://openreview.net/forum?id=OPFnpl7KiF",
            source_url="https://github.com/mila-iqia/Design-Editing-for-Offline-MBO",
            original_framework="PyTorch",
        ),
        MethodBlueprint(
            method_id="root",
            display_name="ROOT",
            family=MethodFamily.TRANSPORT,
            mechanism="probabilistic bridge translating low-score designs into high-score designs",
            required_components=(
                "source and target strata",
                "probabilistic bridge",
                "distributional transport sampler",
            ),
            paper_url=(
                "https://papers.nips.cc/paper_files/paper/2025/hash/"
                "8e357df853b7352fe91d92c83e2f90b2-Abstract-Conference.html"
            ),
            source_url="https://github.com/cuong-dm/ROOT",
            original_framework="PyTorch",
        ),
        MethodBlueprint(
            method_id="spade",
            display_name="SPADE",
            family=MethodFamily.FORWARD_SURROGATE,
            mechanism=(
                "conditional diffusion surrogate with calibration, support-proximity "
                "regularization, and conservative acquisition search"
            ),
            required_components=(
                "conditional score diffusion for p(y|x)",
                "moment and rank calibration",
                "k-nearest-neighbor support index",
                "LCB acquisition",
                "evolutionary search",
            ),
            paper_url="https://arxiv.org/abs/2605.11246",
            source_url="https://github.com/HarryYoung2018/spade",
            original_framework="PyTorch",
        ),
    )
}


def planned_method_names() -> tuple[str, ...]:
    return tuple(sorted(name for name, blueprint in _BLUEPRINTS.items()
                        if blueprint.status in {IntegrationStatus.PLANNED, IntegrationStatus.PORT_IN_PROGRESS}))


def list_method_blueprints() -> tuple[MethodBlueprint, ...]:
    return tuple(_BLUEPRINTS[name] for name in sorted(_BLUEPRINTS))


def get_method_blueprint(method_id: str) -> MethodBlueprint:
    try:
        return _BLUEPRINTS[method_id]
    except KeyError as exc:
        available = ", ".join(sorted(_BLUEPRINTS))
        raise KeyError(
            f"unknown catalog method {method_id!r}; available methods: {available}"
        ) from exc


# Component equations and end-to-end contracts are tested. Full upstream
# architecture/checkpoint and published-table parity are separate claims.
_BLUEPRINTS = {name: replace(blueprint, status=IntegrationStatus.IMPLEMENTED_ADAPTATION)
               for name, blueprint in _BLUEPRINTS.items()}
