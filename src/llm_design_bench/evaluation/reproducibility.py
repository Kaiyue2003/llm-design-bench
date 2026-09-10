"""Content-addressed input and candidate artifacts for offline replay."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import time
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_digest():
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def environment_versions():
    names = ("torch", "numpy", "pandas", "scipy", "scikit-learn", "bayeso-benchmarks")
    return {name: importlib.metadata.version(name) for name in names}


def data_assets_manifest(root):
    if root is None:
        return {}
    root = Path(root)
    paths = [root / "results" / "data_mixing_runs.pkl"]
    paths += sorted((root / "opt_algos").rglob("*.py"))
    paths += sorted((root / "opt_algos" / "data_models" / "20250119_174726_j8mad2i5").rglob("*"))
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in paths if p.is_file() and "__pycache__" not in p.parts}


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("object arrays are not permitted in replay artifacts")
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_replace(source, destination):
    """Preserve the previous checkpoint while Windows readers release it."""
    for attempt in range(21):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 20:
                raise
            time.sleep(0.1)


def save_arrays(root, relative_path, **arrays):
    path = Path(root) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
    atomic_replace(temporary, path)
    return {"path": Path(relative_path).as_posix(), "sha256": sha256_file(path), "arrays_sha256": arrays_digest(arrays)}


def safe_name(name):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def save_run_artifacts(root, problem, result, utility, *, method_id, method_seed, dataset_seed):
    name = safe_name(problem.metadata.task_name)
    prefix = f"{name}__data-{dataset_seed if dataset_seed is not None else 'fixed'}"
    arrays = {"train_designs": problem.train_designs.cpu().numpy(),
              "train_context": problem.train_context.cpu().numpy(),
              "train_utility": problem.train_utility.cpu().numpy(),
              "target_context": problem.target_context.cpu().numpy()}
    if hasattr(problem.design_space, "bounds"):
        arrays["bounds"] = problem.design_space.bounds.cpu().numpy()
    dataset = save_arrays(root, f"datasets/{prefix}.npz", **arrays)
    candidates = save_arrays(root, f"candidates/{prefix}__{method_id}__seed-{method_seed}.npz",
                             candidates=result.candidates.detach().cpu().numpy(),
                             oracle_utility=utility, target_context=arrays["target_context"])
    return {"dataset": dataset, "candidates": candidates}


def verify_artifact(root, entry):
    root = Path(root).resolve()
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("artifact path escapes results directory")
    if not path.is_file() or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"missing or modified replay artifact: {entry['path']}")
    with np.load(path, allow_pickle=False) as arrays:
        if arrays_digest(dict(arrays)) != entry["arrays_sha256"]:
            raise ValueError(f"array hash mismatch: {entry['path']}")


def verify_run_artifacts(root, artifact_json):
    entries = json.loads(artifact_json) if isinstance(artifact_json, str) else artifact_json
    for entry in entries.values():
        verify_artifact(root, entry)
