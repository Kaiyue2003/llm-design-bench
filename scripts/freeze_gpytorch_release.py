"""Publish new immutable GP-backend inputs from a clean, tested code commit.

This command never trains, invokes the oracle, re-splits data, or overwrites a
release. The original trusted data bundle is verified and copied byte-for-byte.
Run only after committing the code, dependency pins, and launcher scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from pathlib import Path

from llm_design_bench.evaluation.data_manifest import load_data_manifest
from llm_design_bench.evaluation.llmdm_protocol import (
    freeze_method_plan,
    package_source_identity,
    save_method_plan,
)
from llm_design_bench.evaluation.run_artifacts import atomic_bytes, atomic_json

EXPERIMENT_ID = "llmdm_forward_gpytorch_v2"
GP_REQUIREMENTS = {"gpytorch": "1.15.2", "linear_operator": "0.6.1"}


def publish(root: Path, upstream: Path, output: Path) -> dict:
    root, upstream, output = root.resolve(), upstream.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite a frozen release: {output}")
    source = package_source_identity()
    if source["git_dirty"] is not False or not source["git_commit"]:
        raise RuntimeError("freeze requires a clean, committed package checkout")
    installed = {name: importlib.metadata.version(name) for name in GP_REQUIREMENTS}
    if installed != GP_REQUIREMENTS:
        raise RuntimeError(
            f"GP backend versions differ from the release pins: {installed}"
        )
    constraints_path = root / "configs/llmdm_gpytorch_runtime.txt"
    pins = [
        line.strip()
        for line in constraints_path.read_text("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if sorted(pins) != sorted(
        f"{name}=={version}" for name, version in GP_REQUIREMENTS.items()
    ):
        raise ValueError("runtime constraints differ from the declared GP backend pins")

    previous = root / "experiments/llmdm_forward_v1"
    old_release = json.loads((previous / "release.json").read_text("utf-8"))
    for relative, expected in old_release["artifact_sha256"].items():
        artifact = (previous / relative).resolve()
        if not artifact.is_relative_to(previous) or _sha256(artifact) != expected:
            raise ValueError(f"original frozen artifact changed: {relative}")
    bundle = load_data_manifest(previous / "data", data_recipes_root=upstream)
    if bundle.manifest_id != old_release["data_manifest_id"]:
        raise ValueError("original data identity differs from its release")
    methods_path = root / "configs/llmdm_methods.forward_v1.json"
    methods = json.loads(methods_path.read_text("utf-8"))
    plan = freeze_method_plan(bundle, methods, experiment_id=EXPERIMENT_ID)
    if plan["package_source"] != source:
        raise RuntimeError("package checkout changed while preparing the release")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "release"
        release = _write_release(
            staging,
            previous,
            methods_path,
            constraints_path,
            source,
            old_release,
            bundle,
            plan,
        )
        if output.exists():
            raise FileExistsError(f"release appeared while preparing it: {output}")
        staging.rename(output)
    return release


def _write_release(
    output,
    previous,
    methods_path,
    constraints_path,
    source,
    old_release,
    bundle,
    plan,
):
    """Build a complete release in private staging before publication."""
    output.mkdir(parents=True, exist_ok=False)
    atomic_bytes(
        output / ".gitattributes",
        b"# Preserve immutable published file hashes on Windows and Linux.\n"
        b"*.json -text whitespace=trailing-space,space-before-tab,cr-at-eol\n"
        b"*.txt -text\n"
        b"data/*.json -text whitespace=trailing-space,space-before-tab,cr-at-eol\n"
        b"data/*.npz -text\n",
    )
    shutil.copytree(previous / "data", output / "data")
    shutil.copyfile(methods_path, output / "methods.json")
    shutil.copyfile(constraints_path, output / "runtime-constraints.txt")
    save_method_plan(plan, output / "plan.json")
    relative_paths = (
        "plan.json",
        "methods.json",
        "runtime-constraints.txt",
        "data/manifest.json",
        "data/visible.npz",
        "data/reference.npz",
    )
    release = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": plan["shared_settings"]["protocol_id"],
        "code_commit": source["git_commit"],
        "package_source_sha256": source["sha256"],
        "upstream_commit": old_release["upstream_commit"],
        "data_manifest_id": bundle.manifest_id,
        "plan_id": plan["plan_id"],
        "gp_backend": {
            "implementation": "gpytorch_exact_gp",
            "packages": GP_REQUIREMENTS,
            "methods": ["ga_on_gp", "bo_qei"],
            "mean": "zero",
            "kernel": "scaled_ard_rbf",
            "posterior": "latent",
            "factorization": "dense_cholesky",
        },
        "counts": {
            **old_release["counts"],
            "usable_logged": len(bundle.reference_utility),
            "main_visible": len(bundle.utility),
            "fixed_1b_visible": int((bundle.context[:, 0] == 1000).sum()),
            "methods": len(plan["methods"]),
        },
        "artifact_sha256": {
            relative: _sha256(output / relative) for relative in relative_paths
        },
        "result_policy": "fresh_experiment_no_v1_result_reuse",
        "runtime_policy": "preserve_colab_torch_and_lock_first_runtime_contract",
    }
    atomic_json(output / "release.json", release)
    return release


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-recipes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = publish(
        Path(__file__).resolve().parents[1], args.data_recipes_root, args.output
    )
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
