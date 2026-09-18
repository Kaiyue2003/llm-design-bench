from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

from llm_design_bench.types import CandidateBatch


@dataclass(frozen=True)
class MetricSpec:
    name: str
    history_column: str
    maximize: bool


METRICS = (
    MetricSpec("train_cross_entropy", "train/CrossEntropyLoss", False),
    MetricSpec(
        "common_crawl_cross_entropy",
        "eval/RedPajamaCommonCrawl/CrossEntropyLoss",
        False,
    ),
    MetricSpec("c4_cross_entropy", "eval/RedPajamaC4/CrossEntropyLoss", False),
    MetricSpec(
        "wikipedia_cross_entropy", "eval/RedPajamaWikipedia/CrossEntropyLoss", False
    ),
    MetricSpec(
        "stack_exchange_cross_entropy",
        "eval/RedPajamaStackExchange/CrossEntropyLoss",
        False,
    ),
    MetricSpec("github_cross_entropy", "eval/RedPajamaGithub/CrossEntropyLoss", False),
    MetricSpec("arxiv_cross_entropy", "eval/RedPajamaArXiv/CrossEntropyLoss", False),
    MetricSpec("book_cross_entropy", "eval/RedPajamaBook/CrossEntropyLoss", False),
    MetricSpec("hellaswag_accuracy", "eval/downstream/hellaswag_len_norm", True),
    MetricSpec("piqa_accuracy", "eval/downstream/piqa_len_norm", True),
    MetricSpec("arc_easy_accuracy", "eval/downstream/arc_easy_acc", True),
)

MODEL_SCALE_LABELS = {
    20: "20M",
    60: "60M",
    150: "150M",
    300: "300M",
    500: "500M",
    700: "700M",
    1000: "1B",
}

# Upstream DataModelBenchmark.feature_names, indices 0..4. The frozen-data
# preparation additionally verifies these names against the selected source.
DOMAIN_ORDER = (
    "RedPajamaWikipedia",
    "RedPajamaStackExchange",
    "RedPajamaGithub",
    "RedPajamaArXiv",
    "RedPajamaBook",
)


class DataRecipesTask:
    mixture_dim = 5
    allowed_model_scales = np.array(sorted(MODEL_SCALE_LABELS), dtype=float)
    min_training_steps = 100
    max_training_steps = 19_600
    training_step_increment = 100
    target_model_scale = 1000
    target_training_steps = 19_500

    def __init__(
        self,
        data_recipes_root: str | Path | None = None,
        metric_index: int = 4,
        logged_model_scale: float | None = None,
        device: str = "cpu",
        simplex_tolerance: float = 1e-6,
        checked_checkpoint_hashes: Mapping[str, str] | None = None,
        checked_file_hashes: Mapping[str, str] | None = None,
    ) -> None:
        if metric_index < 0 or metric_index >= len(METRICS):
            raise ValueError(f"metric_index must be between 0 and {len(METRICS) - 1}")
        if (
            logged_model_scale is not None
            and logged_model_scale not in MODEL_SCALE_LABELS
        ):
            allowed = ", ".join(str(scale) for scale in MODEL_SCALE_LABELS)
            raise ValueError(f"logged_model_scale must be one of: {allowed}")
        self.root = self._resolve_root(data_recipes_root)
        self.metric_index = metric_index
        self.metric = METRICS[metric_index]
        self.logged_model_scale = logged_model_scale
        self.device = device
        self.simplex_tolerance = simplex_tolerance
        self.checked_checkpoint_hashes = (
            None
            if checked_checkpoint_hashes is None
            else dict(checked_checkpoint_hashes)
        )
        self.checked_file_hashes = dict(checked_file_hashes or {})
        self._benchmark = None

    @staticmethod
    def _resolve_root(value: str | Path | None) -> Path:
        candidates: list[Path] = []
        if value is not None:
            resolved = Path(value).expanduser().resolve()
            if not (resolved / "opt_algos" / "benchmarks.py").is_file():
                raise FileNotFoundError(
                    f"invalid explicit data-recipes root: {resolved}"
                )
            return resolved
        if env_value := os.environ.get("DATA_RECIPES_ROOT"):
            candidates.append(Path(env_value))
        candidates.extend(
            [
                Path.cwd() / "data-recipes",
                Path.cwd().parent / "data-recipes",
                Path(__file__).resolve().parents[4] / "data-recipes",
            ]
        )
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if (resolved / "opt_algos" / "benchmarks.py").is_file():
                return resolved
        raise FileNotFoundError(
            "could not find data-recipes; clone it next to llm-design-bench "
            "or set DATA_RECIPES_ROOT"
        )

    @cached_property
    def _logged_data(self) -> tuple[CandidateBatch, np.ndarray, np.ndarray]:
        # pandas pickles may execute code: use only the explicitly trusted
        # published logged dataset, never an untrusted uploaded pickle.
        runs = pd.read_pickle(self.root / "results" / "data_mixing_runs.pkl")
        mixtures: list[np.ndarray] = []
        model_scales: list[float] = []
        training_steps: list[float] = []
        utilities: list[float] = []
        source_row_ids: list[int] = []
        label_to_scale = {label: scale for scale, label in MODEL_SCALE_LABELS.items()}

        for source_position, (_, row) in enumerate(runs.iterrows()):
            history = row["history"]
            if history.empty or self.metric.history_column not in history:
                continue
            row_model_scale = label_to_scale[row["group"]]
            if (
                self.logged_model_scale is not None
                and row_model_scale != self.logged_model_scale
            ):
                continue
            final = history.iloc[-1]
            mixtures.append(np.asarray(row["token_probabilities"], dtype=float))
            model_scales.append(row_model_scale)
            training_steps.append(float(final["_step"]))
            utilities.append(self._to_utility(float(final[self.metric.history_column])))
            source_row_ids.append(source_position)

        if not mixtures:
            raise ValueError(
                "no logged data matched the selected metric and model-scale filter"
            )

        batch = CandidateBatch(
            mixtures=np.vstack(mixtures),
            model_scales=np.asarray(model_scales),
            training_steps=np.asarray(training_steps),
        )
        self.validate(batch)
        utility = np.asarray(utilities, dtype=np.float64)
        if not np.isfinite(utility).all():
            raise ValueError("logged utilities must be finite")
        return batch, utility, np.asarray(source_row_ids, dtype=np.int64)

    @property
    def logged_x(self) -> CandidateBatch:
        return self._logged_data[0]

    @property
    def logged_y(self) -> np.ndarray:
        return self._logged_data[1]

    @property
    def logged_row_ids(self) -> np.ndarray:
        """Original zero-based row positions, independent of DataFrame labels."""

        return self._logged_data[2].copy()

    def validate(self, batch: CandidateBatch) -> None:
        if batch.mixtures.ndim != 2 or batch.mixtures.shape[1] != self.mixture_dim:
            raise ValueError(f"mixtures must have shape (n, {self.mixture_dim})")
        count = len(batch)
        if batch.model_scales.shape != (count,) or batch.training_steps.shape != (
            count,
        ):
            raise ValueError("model_scales and training_steps must have shape (n,)")
        if not (
            np.isfinite(batch.mixtures).all()
            and np.isfinite(batch.model_scales).all()
            and np.isfinite(batch.training_steps).all()
        ):
            raise ValueError("candidate values must be finite")
        if (batch.mixtures < -self.simplex_tolerance).any():
            raise ValueError("mixture components must be non-negative")
        if not np.allclose(
            batch.mixtures.sum(axis=1),
            1.0,
            atol=self.simplex_tolerance,
            rtol=0.0,
        ):
            raise ValueError("mixture components must sum to 1.0")
        if not np.isin(batch.model_scales, self.allowed_model_scales).all():
            raise ValueError(
                f"model scales must be one of {self.allowed_model_scales.tolist()}"
            )
        steps = batch.training_steps
        if (
            (steps < self.min_training_steps).any()
            or (steps > self.max_training_steps).any()
            or not np.allclose(steps % self.training_step_increment, 0.0)
        ):
            raise ValueError(
                "training steps must be 100-step increments between "
                f"{self.min_training_steps} and {self.max_training_steps}"
            )

    def predict(self, batch: CandidateBatch) -> np.ndarray:
        self.validate(batch)
        benchmark = self._get_benchmark()
        values = [
            benchmark._raw_func_with_model_scale(
                training_steps / 100,
                model_scale / 10,
                mixture,
                with_exp=False,
            )
            for mixture, model_scale, training_steps in zip(
                batch.mixtures,
                batch.model_scales,
                batch.training_steps,
                strict=True,
            )
        ]
        return np.asarray([self._to_utility(float(value)) for value in values])

    def cost(self, batch: CandidateBatch) -> np.ndarray:
        self.validate(batch)
        flops = pd.read_csv(self.root / "opt_algos" / "flops_dataset.csv")
        costs: list[float] = []
        for model_scale, training_steps in zip(
            batch.model_scales,
            batch.training_steps,
            strict=True,
        ):
            rows = flops[flops["model_size"] == MODEL_SCALE_LABELS[int(model_scale)]]
            costs.append(float(np.interp(training_steps, rows["step"], rows["flops"])))
        return np.asarray(costs)

    def at_target_fidelity(self, mixtures: np.ndarray) -> CandidateBatch:
        return CandidateBatch.at_fidelity(
            mixtures,
            model_scale=self.target_model_scale,
            training_steps=self.target_training_steps,
        )

    def _to_utility(self, value: float) -> float:
        return value if self.metric.maximize else -value

    def _get_benchmark(self):
        # One verification per predict(batch), not per candidate. This also
        # rejects drift when an already-initialized simulator is reused.
        if self.checked_file_hashes:
            _verify_file_hashes(self.root, self.checked_file_hashes)
            expected_sources = {
                name for name in self.checked_file_hashes if name.endswith(".py")
            }
            actual_sources = {
                item.relative_to(self.root).as_posix()
                for item in (self.root / "opt_algos").rglob("*.py")
            }
            if actual_sources != expected_sources:
                raise ValueError("frozen upstream Python source inventory mismatch")
        if self.checked_checkpoint_hashes is not None:
            _verify_file_hashes(self.root, self.checked_checkpoint_hashes)
        if self._benchmark is None:
            with (
                _temporary_upstream_modules(self.root / "opt_algos"),
                _trusted_checkpoint_loading(self.root, self.checked_checkpoint_hashes),
            ):
                module = self._load_upstream_benchmarks()
                benchmark = module.DataModelBenchmark(
                    metric_index=self.metric_index,
                    device=self.device,
                )
            self._benchmark = benchmark
        return self._benchmark

    def _load_upstream_benchmarks(self) -> ModuleType:
        opt_algos = self.root / "opt_algos"
        # Never reuse the simulator from a different checkout in this process.
        source_id = hashlib.sha256(str(self.root).encode("utf-8")).hexdigest()[:16]
        module_name = f"_llm_design_bench_data_recipes_benchmarks_{source_id}"
        spec = importlib.util.spec_from_file_location(
            module_name, opt_algos / "benchmarks.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not load data-recipes benchmarks.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        with _temporary_sys_path(opt_algos):
            spec.loader.exec_module(module)
        return module


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path.remove(str(path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def python_source_sha256(path: Path) -> str:
    """Hash source text identically across Git CRLF/LF checkouts."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@contextmanager
def _temporary_upstream_modules(path: Path) -> Iterator[None]:
    """Keep bare upstream imports (notably data_model) checkout-local."""

    names = {item.stem for item in path.glob("*.py")}
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    with _temporary_sys_path(path):
        try:
            yield
        finally:
            for name in names:
                sys.modules.pop(name, None)
            sys.modules.update(previous)


def checked_relative_file(root: Path, path: str | Path) -> Path:
    """Resolve an existing file without permitting escape from the trusted root."""

    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"file must stay within data-recipes root: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _verify_file_hashes(root: Path, expected: Mapping[str, str]) -> None:
    for relative_path, digest in expected.items():
        path = checked_relative_file(root, relative_path)
        actual = (
            python_source_sha256(path) if path.suffix == ".py" else file_sha256(path)
        )
        if actual != digest:
            raise ValueError(f"frozen file hash mismatch: {relative_path}")


@contextmanager
def _trusted_checkpoint_loading(
    root: Path | None = None,
    expected: Mapping[str, str] | None = None,
) -> Iterator[None]:
    import torch

    original_load = torch.load
    accessed: set[str] = set()
    if expected is not None and not expected:
        raise ValueError("frozen oracle requires declared checkpoint files")

    def trusted_load(*args, **kwargs):
        if expected is not None:
            value = args[0] if args else kwargs.get("f")
            if root is None or not isinstance(value, (str, os.PathLike)):
                raise ValueError(
                    "frozen oracle may load only declared checkpoint paths"
                )
            path = checked_relative_file(root, value)
            relative = path.relative_to(root).as_posix()
            if relative not in expected:
                raise ValueError(f"undeclared oracle checkpoint: {relative}")
            _verify_file_hashes(root, {relative: expected[relative]})
            accessed.add(relative)
            # Load from the verified absolute path, not a relative path interpreted
            # against the process cwd. Keep a path (rather than a stream) so
            # torch.load's mmap support and all other arguments remain intact.
            if args:
                args = (path, *args[1:])
            else:
                kwargs["f"] = path
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        yield
        if expected is not None and accessed != set(expected):
            raise ValueError("oracle did not load every declared checkpoint")
    finally:
        torch.load = original_load
