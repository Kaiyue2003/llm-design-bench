from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import bayeso_benchmarks as bayeso
import numpy as np

from llm_design_bench.types import CandidateBatch


Objective = Callable[[np.ndarray], np.ndarray]

CATEGORY_DISPLAY_NAMES = {
    "many_local_minima": "Many Local Minima",
    "bowl_shaped": "Bowl-Shaped",
    "plate_shaped": "Plate-Shaped",
    "valley_shaped": "Valley-Shaped",
    "steep_ridges_drops": "Steep Ridges/Drops",
    "other": "Other",
}


@dataclass(frozen=True)
class SyntheticFunctionSpec:
    name: str
    display_name: str
    category: str
    bounds: np.ndarray
    objective: Objective
    global_minimizer: np.ndarray
    global_minimum_value: float
    source: str = "local"


SYNTHETIC_CATEGORIES: dict[str, tuple[str, ...]] = {
    "many_local_minima": (
        "ackley",
        "bukin6",
        "cross_in_tray",
        "drop_wave",
        "eggholder",
        "gramacy_lee_2012",
        "griewank",
        "holder_table",
        "langermann",
        "levy",
        "levy13",
        "rastrigin",
        "schaffer2",
        "schaffer4",
        "schwefel",
        "shubert",
    ),
    "bowl_shaped": (
        "bohachevsky",
        "perm0db",
        "rotated_hyper_ellipsoid",
        "sphere",
        "sum_different_powers",
        "sum_squares",
        "trid",
    ),
    "plate_shaped": (
        "booth",
        "matyas",
        "mccormick",
        "power_sum",
        "zakharov",
    ),
    "valley_shaped": (
        "three_hump_camel",
        "six_hump_camel",
        "dixon_price",
        "rosenbrock",
    ),
    "steep_ridges_drops": (
        "dejong5",
        "easom",
        "michalewicz",
    ),
    "other": (
        "beale",
        "branin",
        "colville",
        "forrester",
        "goldstein_price",
        "hartmann3",
        "hartmann4",
        "hartmann6",
        "permdb",
        "powell",
        "shekel",
        "styblinski_tang",
    ),
}

DEFAULT_SYNTHETIC_FUNCTIONS = tuple(
    function_name
    for category_functions in SYNTHETIC_CATEGORIES.values()
    for function_name in category_functions
)


class SyntheticFunctionTask:
    target_model_scale = 1.0
    target_training_steps = 1.0
    max_training_steps = 1.0

    def __init__(
        self,
        function_name: str,
        logged_samples: int = 256,
        seed: int = 0,
    ) -> None:
        try:
            spec = SYNTHETIC_FUNCTIONS[function_name]
        except KeyError as exc:
            available = ", ".join(sorted(SYNTHETIC_FUNCTIONS))
            raise KeyError(
                f"unknown synthetic function {function_name!r}; available: {available}"
            ) from exc
        if logged_samples <= 0:
            raise ValueError("logged_samples must be positive")

        self.spec = spec
        self.name = spec.name
        self.display_name = spec.display_name
        self.category = spec.category
        self.category_display_name = CATEGORY_DISPLAY_NAMES[spec.category]
        self.design_bounds = np.asarray(spec.bounds, dtype=float)
        self.mixture_dim = self.design_bounds.shape[0]
        self.oracle_utility = -float(spec.global_minimum_value)

        designs = self.sample_uniform(logged_samples, seed=seed)
        self._logged_x = self.at_target_fidelity(designs)
        self._logged_y = self.predict(self._logged_x)

    @property
    def logged_x(self) -> CandidateBatch:
        return self._logged_x

    @property
    def logged_y(self) -> np.ndarray:
        return self._logged_y

    def sample_uniform(self, count: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        lower = self.design_bounds[:, 0]
        upper = self.design_bounds[:, 1]
        return rng.uniform(lower, upper, size=(count, self.mixture_dim))

    def at_target_fidelity(self, designs: np.ndarray) -> CandidateBatch:
        return CandidateBatch.at_fidelity(
            designs,
            self.target_model_scale,
            self.target_training_steps,
        )

    def validate(self, batch: CandidateBatch) -> None:
        designs = np.asarray(batch.mixtures, dtype=float)
        if designs.ndim != 2 or designs.shape[1] != self.mixture_dim:
            raise ValueError(f"designs must have shape (n, {self.mixture_dim})")
        lower = self.design_bounds[:, 0]
        upper = self.design_bounds[:, 1]
        if not np.all(np.isfinite(designs)):
            raise ValueError("designs must be finite")
        if np.any(designs < lower - 1e-9) or np.any(designs > upper + 1e-9):
            raise ValueError("designs are outside the task bounds")

    def predict(self, batch: CandidateBatch) -> np.ndarray:
        self.validate(batch)
        return -self.objective(batch.mixtures)

    def objective(self, designs: np.ndarray) -> np.ndarray:
        designs = np.asarray(designs, dtype=float)
        return self.spec.objective(designs)

    def cost(self, batch: CandidateBatch) -> np.ndarray:
        self.validate(batch)
        return np.ones(len(batch), dtype=float)

    def normalize_designs(self, designs: np.ndarray) -> np.ndarray:
        designs = np.asarray(designs, dtype=float)
        lower = self.design_bounds[:, 0]
        upper = self.design_bounds[:, 1]
        return (designs - lower) / (upper - lower)


def _bayeso_spec(
    name: str,
    display_name: str,
    category: str,
    factory: Callable[[], object],
) -> SyntheticFunctionSpec:
    benchmark = factory()

    def objective(designs: np.ndarray, benchmark=benchmark) -> np.ndarray:
        return np.asarray(benchmark.output(designs), dtype=float).reshape(-1)

    minimizers = np.asarray(benchmark.get_global_minimizers(), dtype=float)
    if minimizers.ndim == 1:
        minimizer = minimizers
    else:
        minimizer = minimizers[0]
    return SyntheticFunctionSpec(
        name=name,
        display_name=display_name,
        category=category,
        bounds=np.asarray(benchmark.get_bounds(), dtype=float),
        objective=objective,
        global_minimizer=np.asarray(minimizer, dtype=float),
        global_minimum_value=float(benchmark.global_minimum),
        source="bayeso-benchmarks",
    )


def _local_spec(
    name: str,
    display_name: str,
    category: str,
    bounds: list[list[float]],
    objective: Objective,
    global_minimizer: list[float],
    global_minimum_value: float,
) -> SyntheticFunctionSpec:
    return SyntheticFunctionSpec(
        name=name,
        display_name=display_name,
        category=category,
        bounds=np.asarray(bounds, dtype=float),
        objective=objective,
        global_minimizer=np.asarray(global_minimizer, dtype=float),
        global_minimum_value=float(global_minimum_value),
    )


def _x(designs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return designs[:, 0], designs[:, 1]


def _cross_in_tray(designs: np.ndarray) -> np.ndarray:
    x1, x2 = _x(designs)
    radius = np.sqrt(x1**2 + x2**2)
    inner = np.abs(np.sin(x1) * np.sin(x2) * np.exp(np.abs(100.0 - radius / np.pi)))
    return -0.0001 * (inner + 1.0) ** 0.1


def _langermann(designs: np.ndarray) -> np.ndarray:
    a = np.array([[3.0, 5.0], [5.0, 2.0], [2.0, 1.0], [1.0, 4.0], [7.0, 9.0]])
    c = np.array([1.0, 2.0, 5.0, 2.0, 3.0])
    result = np.zeros(len(designs), dtype=float)
    for ai, ci in zip(a, c, strict=True):
        squared = np.square(designs - ai).sum(axis=1)
        result -= ci * np.exp(-squared / np.pi) * np.cos(np.pi * squared)
    return result


def _levy13(designs: np.ndarray) -> np.ndarray:
    x1, x2 = _x(designs)
    return (
        np.sin(3.0 * np.pi * x1) ** 2
        + (x1 - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x2) ** 2)
        + (x2 - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x2) ** 2)
    )


def _schaffer2(designs: np.ndarray) -> np.ndarray:
    x1, x2 = _x(designs)
    numerator = np.sin(x1**2 - x2**2) ** 2 - 0.5
    denominator = (1.0 + 0.001 * (x1**2 + x2**2)) ** 2
    return 0.5 + numerator / denominator


def _schaffer4(designs: np.ndarray) -> np.ndarray:
    x1, x2 = _x(designs)
    numerator = np.cos(np.sin(np.abs(x1**2 - x2**2))) ** 2 - 0.5
    denominator = (1.0 + 0.001 * (x1**2 + x2**2)) ** 2
    return 0.5 + numerator / denominator


def _schwefel(designs: np.ndarray) -> np.ndarray:
    dimension = designs.shape[1]
    return 418.9829 * dimension - np.sum(
        designs * np.sin(np.sqrt(np.abs(designs))),
        axis=1,
    )


def _perm0db(designs: np.ndarray, beta: float = 10.0) -> np.ndarray:
    dimension = designs.shape[1]
    result = np.zeros(len(designs), dtype=float)
    indices = np.arange(1, dimension + 1, dtype=float)
    for i in range(1, dimension + 1):
        inner = np.sum((indices + beta) * (designs**i - (1.0 / indices) ** i), axis=1)
        result += inner**2
    return result


def _rotated_hyper_ellipsoid(designs: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(designs, axis=1)
    return np.sum(cumulative**2, axis=1)


def _sum_different_powers(designs: np.ndarray) -> np.ndarray:
    powers = np.arange(2, designs.shape[1] + 2, dtype=float)
    return np.sum(np.abs(designs) ** powers, axis=1)


def _sum_squares(designs: np.ndarray) -> np.ndarray:
    weights = np.arange(1, designs.shape[1] + 1, dtype=float)
    return np.sum(weights * designs**2, axis=1)


def _trid(designs: np.ndarray) -> np.ndarray:
    first = np.sum((designs - 1.0) ** 2, axis=1)
    second = np.sum(designs[:, 1:] * designs[:, :-1], axis=1)
    return first - second


def _matyas(designs: np.ndarray) -> np.ndarray:
    x1, x2 = _x(designs)
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def _mccormick(designs: np.ndarray) -> np.ndarray:
    x1, x2 = _x(designs)
    return np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1.0


def _power_sum(designs: np.ndarray) -> np.ndarray:
    b = np.array([8.0, 18.0, 44.0, 114.0])
    result = np.zeros(len(designs), dtype=float)
    for power, target in enumerate(b, start=1):
        result += (np.sum(designs**power, axis=1) - target) ** 2
    return result


def _dixon_price(designs: np.ndarray) -> np.ndarray:
    result = (designs[:, 0] - 1.0) ** 2
    for index in range(1, designs.shape[1]):
        result += (index + 1.0) * (
            2.0 * designs[:, index] ** 2 - designs[:, index - 1]
        ) ** 2
    return result


def _forrester(designs: np.ndarray) -> np.ndarray:
    x1 = designs[:, 0]
    return (6.0 * x1 - 2.0) ** 2 * np.sin(12.0 * x1 - 4.0)


def _hartmann4(designs: np.ndarray) -> np.ndarray:
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    a = np.array(
        [
            [10.0, 3.0, 17.0, 3.5],
            [0.05, 10.0, 17.0, 0.1],
            [3.0, 3.5, 1.7, 10.0],
            [17.0, 8.0, 0.05, 10.0],
        ]
    )
    p = 1e-4 * np.array(
        [
            [1312.0, 1696.0, 5569.0, 124.0],
            [2329.0, 4135.0, 8307.0, 3736.0],
            [2348.0, 1451.0, 3522.0, 2883.0],
            [4047.0, 8828.0, 8732.0, 5743.0],
        ]
    )
    raw_sum = np.zeros(len(designs), dtype=float)
    for alpha_i, a_i, p_i in zip(alpha, a, p, strict=True):
        raw_sum += alpha_i * np.exp(-np.sum(a_i * (designs - p_i) ** 2, axis=1))
    return (1.1 - raw_sum) / 0.839


def _permdb(designs: np.ndarray, beta: float = 0.5) -> np.ndarray:
    dimension = designs.shape[1]
    result = np.zeros(len(designs), dtype=float)
    indices = np.arange(1, dimension + 1, dtype=float)
    for i in range(1, dimension + 1):
        inner = np.sum((indices**i + beta) * ((designs / indices) ** i - 1.0), axis=1)
        result += inner**2
    return result


def _powell(designs: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4 = designs[:, 0], designs[:, 1], designs[:, 2], designs[:, 3]
    return (
        (x1 + 10.0 * x2) ** 2
        + 5.0 * (x3 - x4) ** 2
        + (x2 - 2.0 * x3) ** 4
        + 10.0 * (x1 - x4) ** 4
    )


def _shekel(designs: np.ndarray) -> np.ndarray:
    a = np.array(
        [
            [4.0, 4.0, 4.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0, 8.0],
            [6.0, 6.0, 6.0, 6.0],
            [3.0, 7.0, 3.0, 7.0],
            [2.0, 9.0, 2.0, 9.0],
            [5.0, 5.0, 3.0, 3.0],
            [8.0, 1.0, 8.0, 1.0],
            [6.0, 2.0, 6.0, 2.0],
            [7.0, 3.6, 7.0, 3.6],
        ]
    )
    c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    result = np.zeros(len(designs), dtype=float)
    for a_i, c_i in zip(a, c, strict=True):
        result -= 1.0 / (np.sum((designs - a_i) ** 2, axis=1) + c_i)
    return result


def _styblinski_tang(designs: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(designs**4 - 16.0 * designs**2 + 5.0 * designs, axis=1)


def _local_specs() -> dict[str, SyntheticFunctionSpec]:
    return {
        "cross_in_tray": _local_spec(
            "cross_in_tray",
            "Cross-in-Tray",
            "many_local_minima",
            [[-10.0, 10.0], [-10.0, 10.0]],
            _cross_in_tray,
            [1.34941, -1.34941],
            -2.0626118708,
        ),
        "langermann": _local_spec(
            "langermann",
            "Langermann",
            "many_local_minima",
            [[0.0, 10.0], [0.0, 10.0]],
            _langermann,
            [2.00299219, 1.006096],
            -5.1621259,
        ),
        "levy13": _local_spec(
            "levy13",
            "Levy N. 13",
            "many_local_minima",
            [[-10.0, 10.0], [-10.0, 10.0]],
            _levy13,
            [1.0, 1.0],
            0.0,
        ),
        "schaffer2": _local_spec(
            "schaffer2",
            "Schaffer N. 2",
            "many_local_minima",
            [[-100.0, 100.0], [-100.0, 100.0]],
            _schaffer2,
            [0.0, 0.0],
            0.0,
        ),
        "schaffer4": _local_spec(
            "schaffer4",
            "Schaffer N. 4",
            "many_local_minima",
            [[-100.0, 100.0], [-100.0, 100.0]],
            _schaffer4,
            [0.0, 1.253115],
            0.292579,
        ),
        "schwefel": _local_spec(
            "schwefel",
            "Schwefel",
            "many_local_minima",
            [[-500.0, 500.0], [-500.0, 500.0]],
            _schwefel,
            [420.9687, 420.9687],
            0.0,
        ),
        "perm0db": _local_spec(
            "perm0db",
            "Perm 0,d,beta",
            "bowl_shaped",
            [[-2.0, 2.0], [-2.0, 2.0]],
            _perm0db,
            [1.0, 0.5],
            0.0,
        ),
        "rotated_hyper_ellipsoid": _local_spec(
            "rotated_hyper_ellipsoid",
            "Rotated Hyper-Ellipsoid",
            "bowl_shaped",
            [[-65.536, 65.536], [-65.536, 65.536]],
            _rotated_hyper_ellipsoid,
            [0.0, 0.0],
            0.0,
        ),
        "sum_different_powers": _local_spec(
            "sum_different_powers",
            "Sum of Different Powers",
            "bowl_shaped",
            [[-1.0, 1.0], [-1.0, 1.0]],
            _sum_different_powers,
            [0.0, 0.0],
            0.0,
        ),
        "sum_squares": _local_spec(
            "sum_squares",
            "Sum Squares",
            "bowl_shaped",
            [[-10.0, 10.0], [-10.0, 10.0]],
            _sum_squares,
            [0.0, 0.0],
            0.0,
        ),
        "trid": _local_spec(
            "trid",
            "Trid",
            "bowl_shaped",
            [[-4.0, 4.0], [-4.0, 4.0]],
            _trid,
            [2.0, 2.0],
            -2.0,
        ),
        "matyas": _local_spec(
            "matyas",
            "Matyas",
            "plate_shaped",
            [[-10.0, 10.0], [-10.0, 10.0]],
            _matyas,
            [0.0, 0.0],
            0.0,
        ),
        "booth": _local_spec(
            "booth",
            "Booth",
            "plate_shaped",
            [[-10.0, 10.0], [-10.0, 10.0]],
            lambda designs: (
                (designs[:, 0] + 2.0 * designs[:, 1] - 7.0) ** 2
                + (2.0 * designs[:, 0] + designs[:, 1] - 5.0) ** 2
            ),
            [1.0, 3.0],
            0.0,
        ),
        "mccormick": _local_spec(
            "mccormick",
            "McCormick",
            "plate_shaped",
            [[-1.5, 4.0], [-3.0, 4.0]],
            _mccormick,
            [-0.54719, -1.54719],
            -1.9133,
        ),
        "power_sum": _local_spec(
            "power_sum",
            "Power Sum",
            "plate_shaped",
            [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0]],
            _power_sum,
            [1.0, 2.0, 2.0, 3.0],
            0.0,
        ),
        "dixon_price": _local_spec(
            "dixon_price",
            "Dixon-Price",
            "valley_shaped",
            [[-10.0, 10.0], [-10.0, 10.0]],
            _dixon_price,
            [1.0, 1.0 / np.sqrt(2.0)],
            0.0,
        ),
        "forrester": _local_spec(
            "forrester",
            "Forrester et al. (2008)",
            "other",
            [[0.0, 1.0]],
            _forrester,
            [0.75724875],
            -6.0207400558,
        ),
        "hartmann4": _local_spec(
            "hartmann4",
            "Hartmann 4-D",
            "other",
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            _hartmann4,
            [0.1873, 0.1906, 0.5566, 0.2647],
            -3.135474,
        ),
        "permdb": _local_spec(
            "permdb",
            "Perm d,beta",
            "other",
            [[-2.0, 2.0], [-2.0, 2.0]],
            _permdb,
            [1.0, 2.0],
            0.0,
        ),
        "powell": _local_spec(
            "powell",
            "Powell",
            "other",
            [[-4.0, 5.0], [-4.0, 5.0], [-4.0, 5.0], [-4.0, 5.0]],
            _powell,
            [0.0, 0.0, 0.0, 0.0],
            0.0,
        ),
        "shekel": _local_spec(
            "shekel",
            "Shekel",
            "other",
            [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
            _shekel,
            [4.0, 4.0, 4.0, 4.0],
            -10.5364,
        ),
        "styblinski_tang": _local_spec(
            "styblinski_tang",
            "Styblinski-Tang",
            "other",
            [[-5.0, 5.0], [-5.0, 5.0]],
            _styblinski_tang,
            [-2.903534, -2.903534],
            -78.3323314075,
        ),
    }


SYNTHETIC_FUNCTIONS: dict[str, SyntheticFunctionSpec] = {
    "ackley": _bayeso_spec(
        "ackley", "Ackley", "many_local_minima", lambda: bayeso.Ackley(2)
    ),
    "bukin6": _bayeso_spec("bukin6", "Bukin N. 6", "many_local_minima", bayeso.Bukin6),
    "drop_wave": _bayeso_spec(
        "drop_wave", "Drop-Wave", "many_local_minima", bayeso.DropWave
    ),
    "eggholder": _bayeso_spec(
        "eggholder", "Eggholder", "many_local_minima", bayeso.Eggholder
    ),
    "gramacy_lee_2012": _bayeso_spec(
        "gramacy_lee_2012",
        "Gramacy & Lee (2012)",
        "many_local_minima",
        bayeso.GramacyAndLee2012,
    ),
    "griewank": _bayeso_spec(
        "griewank", "Griewank", "many_local_minima", lambda: bayeso.Griewank(2)
    ),
    "holder_table": _bayeso_spec(
        "holder_table", "Holder Table", "many_local_minima", bayeso.HolderTable
    ),
    "levy": _bayeso_spec("levy", "Levy", "many_local_minima", lambda: bayeso.Levy(2)),
    "rastrigin": _bayeso_spec(
        "rastrigin", "Rastrigin", "many_local_minima", lambda: bayeso.Rastrigin(2)
    ),
    "shubert": _bayeso_spec("shubert", "Shubert", "many_local_minima", bayeso.Shubert),
    "bohachevsky": _bayeso_spec(
        "bohachevsky", "Bohachevsky 1", "bowl_shaped", bayeso.Bohachevsky
    ),
    "sphere": _bayeso_spec("sphere", "Sphere", "bowl_shaped", lambda: bayeso.Sphere(2)),
    "zakharov": _bayeso_spec(
        "zakharov", "Zakharov", "plate_shaped", lambda: bayeso.Zakharov(2)
    ),
    "three_hump_camel": _bayeso_spec(
        "three_hump_camel", "Three-Hump Camel", "valley_shaped", bayeso.ThreeHumpCamel
    ),
    "six_hump_camel": _bayeso_spec(
        "six_hump_camel", "Six-Hump Camel", "valley_shaped", bayeso.SixHumpCamel
    ),
    "rosenbrock": _bayeso_spec(
        "rosenbrock", "Rosenbrock", "valley_shaped", lambda: bayeso.Rosenbrock(2)
    ),
    "dejong5": _bayeso_spec(
        "dejong5", "De Jong N. 5", "steep_ridges_drops", bayeso.DeJong5
    ),
    "easom": _bayeso_spec("easom", "Easom", "steep_ridges_drops", bayeso.Easom),
    "michalewicz": _bayeso_spec(
        "michalewicz", "Michalewicz", "steep_ridges_drops", bayeso.Michalewicz
    ),
    "beale": _bayeso_spec("beale", "Beale", "other", bayeso.Beale),
    "branin": _bayeso_spec("branin", "Branin", "other", bayeso.Branin),
    "colville": _bayeso_spec("colville", "Colville", "other", bayeso.Colville),
    "goldstein_price": _bayeso_spec(
        "goldstein_price", "Goldstein-Price", "other", bayeso.GoldsteinPrice
    ),
    "hartmann3": _bayeso_spec("hartmann3", "Hartmann 3-D", "other", bayeso.Hartmann3D),
    "hartmann6": _bayeso_spec("hartmann6", "Hartmann 6-D", "other", bayeso.Hartmann6D),
}
SYNTHETIC_FUNCTIONS.update(_local_specs())
