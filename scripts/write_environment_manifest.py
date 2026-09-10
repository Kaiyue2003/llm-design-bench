from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from llm_design_bench.evaluation.reproducibility import code_digest
from llm_design_bench.optimizers import list_methods
from dataclasses import asdict


PACKAGES = (
    "llm-design-bench",
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "bayeso-benchmarks",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    lockfile = repository_root / "uv.lock"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "container_vcs_ref": os.environ.get("LLM_DESIGN_BENCH_VCS_REF"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "code_sha256": code_digest(),
        "method_registry": [asdict(method) for method in list_methods()],
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "determinism_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "uv_lock_sha256": (
            hashlib.sha256(lockfile.read_bytes()).hexdigest()
            if lockfile.is_file()
            else None
        ),
        "packages": {package: _version(package) for package in PACKAGES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
