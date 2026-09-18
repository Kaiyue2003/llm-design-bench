"""Frozen LLM-DM data, with separate optimizer-visible and evaluator artifacts.

Preparation reads the explicitly selected, trusted upstream pandas pickle. A
pickle is executable input, not a safe interchange format. The shareable visible
artifact uses ordinary NumPy arrays and is loaded with ``allow_pickle=False``.
Preparation never imports, constructs, or queries the simulator.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm_design_bench.evaluation.unified_report import BenchmarkTaskSpec, BenchmarkTrial
from llm_design_bench.problem import OfflineProblem, ProblemMetadata
from llm_design_bench.spaces import SimplexSpace
from llm_design_bench.tasks.data_recipes import (
    DOMAIN_ORDER,
    METRICS,
    MODEL_SCALE_LABELS,
    DataRecipesTask,
    checked_relative_file,
    file_sha256,
    python_source_sha256,
)

DATA_MANIFEST_SCHEMA_VERSION = 1
SPLIT_RULE = {
    "group_by": "model_scale",
    "min_percentile": 0.0,
    "max_percentile": 40.0,
    "quantile_method": "linear",
    "bounds": "inclusive",
    "tie_rule": "keep_all_equal_to_cutoff",
    "row_order": "source_position_ascending",
    "checkpoint_selection": "last_history_row",
    "fixed_1b": "subset_of_main_visible_without_resplitting",
}
_VISIBLE_KEYS = ("row_ids", "mixtures", "context", "utility")


def stratified_percentile_mask(
    utility: np.ndarray,
    model_scales: np.ndarray,
    *,
    min_percentile: float = 0.0,
    max_percentile: float = 40.0,
) -> np.ndarray:
    """Inclusive, linear percentiles per scale; ties may exceed 40% of rows."""

    if not (0.0 <= min_percentile <= max_percentile <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= min <= max <= 100")
    utility = np.asarray(utility, dtype=np.float64)
    scales = np.asarray(model_scales, dtype=np.float64)
    if utility.ndim != 1 or scales.shape != utility.shape or utility.size == 0:
        raise ValueError("utility and model_scales must be non-empty aligned vectors")
    if not np.isfinite(utility).all() or not np.isfinite(scales).all():
        raise ValueError("utility and model_scales must be finite")
    mask = np.zeros(len(utility), dtype=bool)
    for scale in np.unique(scales):
        group = scales == scale
        lower, upper = np.percentile(
            utility[group], [min_percentile, max_percentile], method="linear"
        )
        mask |= group & (utility >= lower) & (utility <= upper)
    if not mask.any():
        raise ValueError("stratified percentile split is empty")
    return mask


@dataclass(frozen=True)
class FrozenDataManifest:
    """Evaluator-owned bundle; give methods only an OfflineProblem or visible.npz."""

    manifest_id: str
    metadata: Mapping[str, Any]
    visible_row_ids: np.ndarray
    mixtures: np.ndarray
    context: np.ndarray
    utility: np.ndarray
    reference_row_ids: np.ndarray
    reference_mixtures: np.ndarray
    reference_context: np.ndarray
    reference_utility: np.ndarray

    def __post_init__(self) -> None:
        # Defensive copies prevent writes to caller-owned arrays from changing
        # the frozen data. Hash verification also catches later forced mutation.
        object.__setattr__(self, "metadata", json.loads(_canonical_json(self.metadata)))
        for name in _ARRAY_FIELDS:
            dtype = np.int64 if name.endswith("row_ids") else np.float64
            array = np.array(getattr(self, name), dtype=dtype, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


_ARRAY_FIELDS = (
    "visible_row_ids",
    "mixtures",
    "context",
    "utility",
    "reference_row_ids",
    "reference_mixtures",
    "reference_context",
    "reference_utility",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array_hash(array: np.ndarray) -> str:
    # Explicit little-endian storage makes identities independent of host endian.
    dtype = (
        np.dtype("<i8") if np.issubdtype(array.dtype, np.integer) else np.dtype("<f8")
    )
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(_canonical_json([dtype.str, list(array.shape)]).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _content_id(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    payload = {
        "metadata": metadata,
        "arrays": {k: _array_hash(v) for k, v in arrays.items()},
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    files = sorted((root / "opt_algos").rglob("*.py"))
    if not files:
        raise ValueError("data-recipes Python sources are missing")
    return {
        checked_relative_file(root, path)
        .relative_to(root)
        .as_posix(): python_source_sha256(path)
        for path in files
    }


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _verify_domain_order(root: Path) -> None:
    source = checked_relative_file(root, "opt_algos/benchmarks.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    benchmark = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DataModelBenchmark"
        ),
        None,
    )
    if benchmark is not None:
        for node in ast.walk(benchmark):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "feature_names"
                for target in node.targets
            ):
                continue
            try:
                names = ast.literal_eval(node.value)
                order = tuple(names[index] for index in range(5))
            except (ValueError, TypeError, KeyError, IndexError):
                continue
            if order == DOMAIN_ORDER:
                return
    raise ValueError(
        "upstream DataModelBenchmark does not declare the canonical domain order"
    )


def prepare_data_manifest(
    data_recipes_root: str | Path,
    oracle_checkpoint_paths: Sequence[str | Path],
    *,
    metric_index: int = 4,
) -> FrozenDataManifest:
    """Freeze trusted logged data without importing or evaluating the oracle.

    Checkpoints must be explicitly supplied files within the upstream checkout.
    For each ``checkpoints/*.pt`` file, the upstream loader's ``config.json`` and
    ``feature_mask.json`` sidecars are also required and hashed. No pickle or
    checkpoint is discovered or deserialized other than the fixed logged pickle.
    """

    if metric_index != 4:
        raise ValueError(
            "the agreed LLM-DM protocol requires metric_index=4 (StackExchange)"
        )
    task = DataRecipesTask(data_recipes_root, metric_index=metric_index)
    root = task.root
    _verify_domain_order(root)
    if not oracle_checkpoint_paths:
        raise ValueError("declare at least one trusted oracle checkpoint file")
    checkpoints: dict[str, str] = {}
    sidecars: dict[str, str] = {}
    for supplied in oracle_checkpoint_paths:
        path = checked_relative_file(root, supplied)
        relative = path.relative_to(root).as_posix()
        if path.parent.name != "checkpoints" or path.suffix != ".pt":
            raise ValueError(
                "upstream checkpoints must be explicit checkpoints/*.pt files"
            )
        if relative in checkpoints:
            raise ValueError(f"duplicate checkpoint declaration: {relative}")
        checkpoints[relative] = file_sha256(path)
        for filename in ("config.json", "feature_mask.json"):
            sidecar = checked_relative_file(root, path.parent.parent / filename)
            sidecars[sidecar.relative_to(root).as_posix()] = file_sha256(sidecar)
    data_path = checked_relative_file(root, "results/data_mixing_runs.pkl")
    data_digest = file_sha256(data_path)
    source_hashes = _source_hashes(root)
    full_x, full_y = task.logged_x, task.logged_y
    context = np.column_stack([full_x.model_scales, full_x.training_steps])
    mask = stratified_percentile_mask(full_y, full_x.model_scales)
    if not np.any(mask & (full_x.model_scales == 1000)):
        raise ValueError("the shared visible data must contain a 1B ablation subset")
    arrays = {
        "visible_row_ids": task.logged_row_ids[mask],
        "mixtures": full_x.mixtures[mask],
        "context": context[mask],
        "utility": full_y[mask],
        "reference_row_ids": task.logged_row_ids,
        "reference_mixtures": full_x.mixtures,
        "reference_context": context,
        "reference_utility": full_y,
    }
    metadata = {
        "schema_version": DATA_MANIFEST_SCHEMA_VERSION,
        "domain_order": list(DOMAIN_ORDER),
        "metric_index": metric_index,
        "objective": METRICS[metric_index].history_column,
        "utility_transform": "negative_loss",
        "target_context": [1000.0, 19500.0],
        "split": dict(SPLIT_RULE),
        "data_file": {
            "path": data_path.relative_to(root).as_posix(),
            "sha256": data_digest,
        },
        "upstream_revision": _git_revision(root),
        "source_files": source_hashes,
        "source_fingerprint": hashlib.sha256(
            _canonical_json(source_hashes).encode("utf-8")
        ).hexdigest(),
        "oracle_checkpoints": checkpoints,
        "oracle_sidecars": sidecars,
    }
    bundle = FrozenDataManifest(_content_id(metadata, arrays), metadata, **arrays)
    _validate_bundle(bundle)
    verify_data_manifest_sources(bundle, root)
    return bundle


def _validate_bundle(bundle: FrozenDataManifest) -> None:
    metadata = bundle.metadata
    if metadata.get("schema_version") != DATA_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported frozen data manifest schema")
    if (
        metadata.get("domain_order") != list(DOMAIN_ORDER)
        or metadata.get("metric_index") != 4
        or metadata.get("objective") != METRICS[4].history_column
        or metadata.get("utility_transform") != "negative_loss"
        or metadata.get("target_context") != [1000.0, 19500.0]
        or metadata.get("split") != SPLIT_RULE
    ):
        raise ValueError("manifest does not match the agreed LLM-DM protocol")
    arrays = {name: getattr(bundle, name) for name in _ARRAY_FIELDS}
    if _content_id(metadata, arrays) != bundle.manifest_id:
        raise ValueError("data manifest content hash mismatch")
    for prefix in ("", "reference_"):
        row_ids = bundle.visible_row_ids if not prefix else bundle.reference_row_ids
        mixtures = getattr(bundle, f"{prefix}mixtures")
        context = getattr(bundle, f"{prefix}context")
        utility = getattr(bundle, f"{prefix}utility")
        count = len(row_ids)
        if (
            row_ids.ndim != 1
            or count == 0
            or mixtures.shape != (count, 5)
            or context.shape != (count, 2)
            or utility.shape != (count,)
        ):
            raise ValueError("invalid frozen data array shape")
        if (row_ids < 0).any() or (np.diff(row_ids) <= 0).any():
            raise ValueError("source row IDs must be unique, nonnegative, and ordered")
        if not all(np.isfinite(value).all() for value in (mixtures, context, utility)):
            raise ValueError("frozen data arrays must be finite")
        if (mixtures < -1e-6).any() or not np.allclose(
            mixtures.sum(axis=1), 1.0, rtol=0, atol=1e-6
        ):
            raise ValueError("frozen mixtures must satisfy the simplex")
        if not np.isin(context[:, 0], list(MODEL_SCALE_LABELS)).all():
            raise ValueError("frozen model scale is invalid")
        steps = context[:, 1]
        if ((steps < 100) | (steps > 19600) | (steps % 100 != 0)).any():
            raise ValueError("frozen training steps are invalid")
    mask = stratified_percentile_mask(
        bundle.reference_utility, bundle.reference_context[:, 0]
    )
    if not np.array_equal(bundle.visible_row_ids, bundle.reference_row_ids[mask]):
        raise ValueError("visible row IDs do not match the stratified split")
    for name in ("mixtures", "context", "utility"):
        if not np.array_equal(
            getattr(bundle, name), getattr(bundle, f"reference_{name}")[mask]
        ):
            raise ValueError("visible arrays do not match the selected source rows")
    if not (bundle.context[:, 0] == 1000).any():
        raise ValueError("visible data is missing the fixed-1B ablation subset")


def verify_data_manifest_sources(
    bundle: FrozenDataManifest, data_recipes_root: str | Path
) -> None:
    """Reject data, source, sidecar, or checkpoint drift without reading pickles."""

    _validate_bundle(bundle)
    root = DataRecipesTask._resolve_root(data_recipes_root)
    metadata = bundle.metadata
    if _source_hashes(root) != metadata["source_files"]:
        raise ValueError("upstream Python source fingerprint mismatch")
    source_digest = hashlib.sha256(
        _canonical_json(metadata["source_files"]).encode()
    ).hexdigest()
    if source_digest != metadata["source_fingerprint"]:
        raise ValueError("invalid recorded source fingerprint")
    expected_files = {
        metadata["data_file"]["path"]: metadata["data_file"]["sha256"],
        **metadata["oracle_checkpoints"],
        **metadata["oracle_sidecars"],
    }
    if not metadata["oracle_checkpoints"]:
        raise ValueError("oracle checkpoint provenance is missing")
    for relative, digest in expected_files.items():
        if file_sha256(checked_relative_file(root, relative)) != digest:
            raise ValueError(f"frozen file hash mismatch: {relative}")


def save_data_manifest(bundle: FrozenDataManifest, path: str | Path) -> Path:
    """Write a new bundle directory; never overwrite a prior experiment bundle."""

    _validate_bundle(bundle)
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        directory / "visible.npz",
        row_ids=bundle.visible_row_ids,
        mixtures=bundle.mixtures,
        context=bundle.context,
        utility=bundle.utility,
    )
    np.savez_compressed(
        directory / "reference.npz",
        row_ids=bundle.reference_row_ids,
        mixtures=bundle.reference_mixtures,
        context=bundle.reference_context,
        utility=bundle.reference_utility,
    )
    envelope = {
        "manifest_id": bundle.manifest_id,
        "metadata": bundle.metadata,
        "artifacts": {
            name: file_sha256(directory / name)
            for name in ("visible.npz", "reference.npz")
        },
    }
    # Write the completion marker last: an interrupted directory cannot load.
    (directory / "manifest.json").write_text(
        json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return directory


def load_data_manifest(
    path: str | Path,
    *,
    data_recipes_root: str | Path,
) -> FrozenDataManifest:
    directory = Path(path).resolve()
    envelope = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(envelope) != {"manifest_id", "metadata", "artifacts"}:
        raise ValueError("invalid frozen manifest fields")
    if set(envelope["artifacts"]) != {"visible.npz", "reference.npz"}:
        raise ValueError("invalid frozen manifest artifacts")
    arrays: dict[str, np.ndarray] = {}
    for filename, prefix in (("visible.npz", ""), ("reference.npz", "reference_")):
        file_path = checked_relative_file(directory, filename)
        if file_sha256(file_path) != envelope["artifacts"][filename]:
            raise ValueError(f"frozen artifact file hash mismatch: {filename}")
        with np.load(file_path, allow_pickle=False) as archive:
            if set(archive.files) != set(_VISIBLE_KEYS):
                raise ValueError(f"invalid frozen array names in {filename}")
            for key in _VISIBLE_KEYS:
                array = archive[key]
                expected_dtype = (
                    np.dtype("int64") if key == "row_ids" else np.dtype("float64")
                )
                if array.dtype != expected_dtype:
                    raise ValueError(
                        f"invalid frozen array precision: {filename}/{key}"
                    )
                name = (
                    "visible_row_ids"
                    if not prefix and key == "row_ids"
                    else prefix + key
                )
                arrays[name] = array
    bundle = FrozenDataManifest(envelope["manifest_id"], envelope["metadata"], **arrays)
    verify_data_manifest_sources(bundle, data_recipes_root)
    return bundle


def make_frozen_data_recipes_task_spec(
    bundle: FrozenDataManifest,
    *,
    data_recipes_root: str | Path,
    logged_model_scale: float | None = None,
    device: str = "cpu",
) -> BenchmarkTaskSpec:
    """Build main/ablation from the same frozen visible rows, never resplitting."""

    if logged_model_scale not in (None, 1000):
        raise ValueError("the agreed settings are multi-scale and fixed-1B")
    verify_data_manifest_sources(bundle, data_recipes_root)
    metadata = bundle.metadata
    checked_files = {
        **metadata["source_files"],
        **metadata["oracle_sidecars"],
        metadata["data_file"]["path"]: metadata["data_file"]["sha256"],
    }
    task = DataRecipesTask(
        data_recipes_root,
        metric_index=4,
        device=device,
        checked_checkpoint_hashes=metadata["oracle_checkpoints"],
        checked_file_hashes=checked_files,
    )
    mask = np.ones(len(bundle.utility), dtype=bool)
    if logged_model_scale is not None:
        mask &= bundle.context[:, 0] == logged_model_scale
    task_id = "data_recipes_stack_exchange" + ("_1b" if logged_model_scale else "")
    setting = "multi_scale" if logged_model_scale is None else "fixed_scale"
    problem = OfflineProblem(
        train_designs=torch.tensor(bundle.mixtures[mask], dtype=torch.float64),
        train_context=torch.tensor(bundle.context[mask], dtype=torch.float64),
        train_utility=torch.tensor(bundle.utility[mask], dtype=torch.float64),
        target_context=torch.tensor([1000.0, 19500.0], dtype=torch.float64),
        design_space=SimplexSpace(5),
        metadata=ProblemMetadata(
            task_name=task_id,
            objective_name=METRICS[4].name,
            source="data-recipes",
            extra={
                "setting": setting,
                "logged_model_scale": logged_model_scale,
                "manifest_id": bundle.manifest_id,
                "domain_order": list(DOMAIN_ORDER),
                "visible_row_ids": bundle.visible_row_ids[mask].tolist(),
                "utility_transform": "negative_loss",
                "target_model_scale": 1000,
                "target_training_steps": 19500,
                "split_rule": dict(SPLIT_RULE),
            },
        ),
    )
    trial = BenchmarkTrial(
        evaluator_task=task,
        problem=problem,
        reference_utility=bundle.reference_utility,
        d_best_utility=float(bundle.utility[mask].max()),
    )

    def trial_factory(seed: int) -> BenchmarkTrial:
        del seed
        return trial

    return BenchmarkTaskSpec(
        task_id=task_id,
        display_name="LLM-DM" + (" (1B logged)" if logged_model_scale else ""),
        suite="data_mixture",
        category="data_mixture",
        category_display_name="Data Mixture",
        trial_factory=trial_factory,
        normalization_reference_id=f"llmdm_{bundle.manifest_id}_full_logged",
    )
