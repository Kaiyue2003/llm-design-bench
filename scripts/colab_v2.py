"""Verified setup and finite pilot/review/formal orchestration for Colab v2.

Importing this stdlib-only module does not install packages, mount Drive or train.
The frozen BatchRunner remains responsible for job identities, seed coverage,
artifact verification, environment checks, backups and the no-automatic-retry rule.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

EXPERIMENT = "llmdm_forward_gpytorch_v2"
REQUIRED_ARTIFACTS = frozenset(
    {
        "plan.json",
        "data/manifest.json",
        "data/visible.npz",
        "data/reference.npz",
        "runtime-constraints.txt",
    }
)
EXPECTED_COUNTS = {
    "usable_logged": 454,
    "main_visible": 184,
    "fixed_1b_visible": 26,
    "methods": 19,
}
RUNTIME_PACKAGES = (
    "torch",
    "gpytorch",
    "linear_operator",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "plotly",
    "matplotlib",
    "pydantic",
    "typer",
    "mpmath",
    "joblib",
    "threadpoolctl",
    "bayeso-benchmarks",
    "seaborn",
)
MODULE_DISTRIBUTIONS = {
    "torch": "torch",
    "gpytorch": "gpytorch",
    "linear_operator": "linear_operator",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
}


def _hex(value: Any, length: int, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{label} must be a pinned lowercase {length}-digit hash")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("invalid release artifact path")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not relative
        or any(x in {"", ".", ".."} for x in relative.split("/"))
    ):
        raise ValueError("release artifact must be a normal relative path")
    path = root.joinpath(*pure.parts)
    if any(item.is_symlink() for item in (root, path, *path.parents)):
        raise ValueError("release paths must not contain symlinks")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("release artifact escapes its root")
    return path


def checkout_exact(url: str, destination: str | Path, revision: str) -> Path:
    """Create a pinned checkout or verify it, never reset an existing directory."""
    _hex(revision, 40, "revision")
    allowed = {
        "https://github.com/Kaiyue2003/llm-design-bench.git",
        "https://github.com/namkoong-lab/data-recipes.git",
    }
    if url not in allowed:
        raise ValueError("unexpected repository URL")
    destination = Path(destination)
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("checkout destination cannot contain symlinks")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                url,
                str(destination),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(destination),
                "-c",
                "core.autocrlf=false",
                "checkout",
                "--detach",
                revision,
            ],
            check=True,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain"], text=True
    ).strip()
    origin = subprocess.check_output(
        ["git", "-C", str(destination), "remote", "get-url", "origin"], text=True
    ).strip()
    if actual != revision or dirty or origin != url:
        raise RuntimeError(
            "checkout/version/origin mismatch; use a fresh runtime, do not overwrite"
        )
    return destination


def validate_release(
    assets: str | Path, *, code_commit: str, upstream_commit: str
) -> dict[str, Any]:
    """Validate release metadata and all required artifacts before imports/install."""
    _hex(code_commit, 40, "code commit")
    _hex(upstream_commit, 40, "upstream commit")
    assets = Path(assets)
    path = _safe_path(assets, "release.json")
    release = json.loads(path.read_text("utf-8"))
    if not isinstance(release, dict):
        raise TypeError("release must be an object")
    if release.get("experiment_id") != EXPERIMENT or release.get("schema_version") != 1:
        raise ValueError("not a supported GPyTorch v2 release")
    if (
        release.get("code_commit") != code_commit
        or release.get("upstream_commit") != upstream_commit
    ):
        raise ValueError("release disagrees with pinned checkout revisions")
    for key in ("data_manifest_id", "plan_id", "package_source_sha256"):
        _hex(release.get(key), 64, key)
    counts = release.get("counts")
    if not isinstance(counts, dict) or any(
        counts.get(k) != v for k, v in EXPECTED_COUNTS.items()
    ):
        raise ValueError("unexpected frozen dataset/method counts")
    artifacts = release.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise ValueError("release is missing required artifact hashes")
    for relative, expected in artifacts.items():
        _hex(expected, 64, "artifact SHA256")
        path = _safe_path(assets, relative)
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"release artifact mismatch: {relative}")
    constraints = read_runtime_constraints(assets / "runtime-constraints.txt")
    if not {"gpytorch", "linear_operator"}.issubset(constraints):
        raise ValueError("GPyTorch and linear_operator must both have frozen versions")
    return release


def read_runtime_constraints(path: str | Path) -> dict[str, str]:
    versions = {}
    for line in Path(path).read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9.+_-]*)", line
        )
        if match is None:
            raise ValueError(
                "runtime constraints must contain exact name==version pins only"
            )
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "_", name).lower()
        if name in versions:
            raise ValueError(f"duplicate runtime constraint: {name}")
        versions[name] = version
    if not versions:
        raise ValueError("empty runtime constraints")
    return versions


def _check_loaded_sources(repo: Path) -> None:
    expected = {
        "llm_design_bench": repo / "src/llm_design_bench/__init__.py",
        "colab_support": repo / "scripts/colab_support.py",
        "colab_batch": repo / "scripts/colab_batch.py",
        "colab_types": repo / "scripts/colab_types.py",
        "colab_verification": repo / "scripts/colab_verification.py",
        "colab_v2": repo / "scripts/colab_v2.py",
    }
    for name, path in expected.items():
        module = sys.modules.get(name)
        if (
            module is not None
            and Path(getattr(module, "__file__", "")).resolve() != path.resolve()
        ):
            raise RuntimeError(
                f"{name} is loaded from another checkout; start a fresh Python runtime"
            )


def _check_versions(versions: dict[str, str]) -> None:
    for name, expected in versions.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"frozen dependency mismatch: {name}=={expected}")


def _check_loaded_versions() -> None:
    for name, distribution in MODULE_DISTRIBUTIONS.items():
        module = sys.modules.get(name)
        if module is not None:
            loaded = getattr(module, "__version__", None)
            if loaded is None or str(loaded) != importlib.metadata.version(
                distribution
            ):
                raise RuntimeError(
                    f"loaded {name} differs from installed version; restart Python before continuing"
                )


def install_environment(repo: str | Path, assets: str | Path) -> None:
    """Install once with frozen GP dependencies, retaining Colab's CUDA Torch."""
    repo, assets = Path(repo).resolve(), Path(assets).resolve()
    _check_loaded_sources(repo)
    _check_loaded_versions()
    versions = read_runtime_constraints(assets / "runtime-constraints.txt")
    torch_before = importlib.metadata.version("torch")
    loaded_before = {
        name: importlib.metadata.version(distribution)
        for name, distribution in MODULE_DISTRIBUTIONS.items()
        if name in sys.modules
    }
    with tempfile.TemporaryDirectory(prefix="llmdm-v2-install-") as temporary:
        torch_pin = Path(temporary) / "torch.txt"
        torch_pin.write_text(f"torch=={torch_before}\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-c",
                str(torch_pin),
                "-c",
                str(assets / "runtime-constraints.txt"),
                "-e",
                str(repo),
            ],
            check=True,
        )
        check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            text=True,
            capture_output=True,
            check=False,
        )
        # Some Colab images omit IPython's Jedi dependency. Repair only this
        # exact known failure, not arbitrary conflicts or a failed installation.
        lines = [line.strip() for line in check.stdout.splitlines() if line.strip()]
        if (
            check.returncode
            and lines
            and all(
                re.fullmatch(
                    r"ipython [^ ]+ requires jedi, which is not installed\.",
                    line,
                    re.IGNORECASE,
                )
                for line in lines
            )
            and not check.stderr.strip()
        ):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-c",
                    str(torch_pin),
                    "jedi==0.19.2",
                ],
                check=True,
            )
            check = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                text=True,
                capture_output=True,
                check=False,
            )
        if check.returncode:
            raise RuntimeError(
                f"pip check failed; training has not started:\n{check.stdout}\n{check.stderr}"
            )
    if importlib.metadata.version("torch") != torch_before:
        raise RuntimeError(
            "PyTorch changed during installation; do not train in this runtime"
        )
    _check_versions(versions)
    changed = [
        name
        for name, previous in loaded_before.items()
        if importlib.metadata.version(MODULE_DISTRIBUTIONS[name]) != previous
    ]
    if changed:
        raise RuntimeError(
            f"loaded libraries changed: {changed}; restart Python and run all again"
        )
    _check_loaded_versions()
    # Editable installation .pth files are normally processed only at startup.
    # Explicitly add the already-verified checkout after installation succeeded.
    for path in (repo / "scripts", repo / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    importlib.invalidate_caches()
    os.environ.setdefault("MPLCONFIGDIR", str(repo.parent / "matplotlib-cache"))
    importlib.import_module("llm_design_bench")
    _check_loaded_sources(repo)


def runtime_identity() -> dict[str, Any]:
    _check_loaded_versions()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "select a GPU runtime; v2 does not silently switch neural methods to CPU"
        )
    if str(torch.__version__) != importlib.metadata.version("torch"):
        raise RuntimeError("loaded Torch differs from its installation; restart Python")
    return {
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "packages": {
            name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES
        },
    }


def prepare_workflow(
    repo: str | Path,
    assets: str | Path,
    upstream: str | Path,
    release: dict[str, Any],
    *,
    content_root: str | Path = "/content",
    drive_root: str | Path = "/content/drive",
):
    """Require mounted Drive, restore only this v2 plan, then construct the queue."""
    repo, assets, upstream = (
        Path(repo).resolve(),
        Path(assets).resolve(),
        Path(upstream).resolve(),
    )
    _check_loaded_sources(repo)
    verified_release = validate_release(
        assets,
        code_commit=release["code_commit"],
        upstream_commit=release["upstream_commit"],
    )
    if release != verified_release:
        raise ValueError("supplied release differs from its verified on-disk metadata")
    drive_root, content_root = Path(drive_root), Path(content_root)
    if not os.path.ismount(drive_root) or not (drive_root / "MyDrive").is_dir():
        raise RuntimeError(
            "Google Drive is not mounted; authorize mounting before any training"
        )
    for path in (content_root, drive_root / "MyDrive"):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("state/backup roots cannot contain symlinks")
    from colab_batch import BatchRunner
    from colab_support import restore_latest, snapshot_tree

    from llm_design_bench.evaluation.run_artifacts import atomic_json

    state = content_root / "llmdm_gpytorch_v2_state" / release["plan_id"]
    backups = drive_root / "MyDrive/llm_design_bench" / EXPERIMENT / release["plan_id"]
    for path in (state, backups):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("state/backup paths cannot contain symlinks")
    backups.mkdir(parents=True, exist_ok=True)
    if not state.exists() or not any(state.iterdir()):
        if list((backups / "snapshots").glob("snapshot-*.tar.gz")):
            print("Restored:", restore_latest(backups, state), flush=True)
        elif any(backups.iterdir()):
            raise RuntimeError(
                "backup records exist without a restorable snapshot; inspect before retrying"
            )
        else:
            state.mkdir(parents=True, exist_ok=True)

    constraints = read_runtime_constraints(assets / "runtime-constraints.txt")
    contract_path = state / "runtime-v2.json"

    def check_runtime():
        _check_versions(constraints)
        current = {"plan_id": release["plan_id"], "environment": runtime_identity()}
        if contract_path.exists():
            if json.loads(contract_path.read_text("utf-8")) != current:
                raise RuntimeError(
                    "v2 runtime changed (including GPyTorch/GPU); inspect, do not mix environments"
                )
        elif any(state.iterdir()):
            raise RuntimeError(
                "existing state lacks its v2 runtime contract; do not adopt old results"
            )
        else:
            atomic_json(contract_path, current)
            snapshot_tree(state, backups)

    class RuntimeCheckedBatch(BatchRunner):
        def _check_environment(self, run_id, device, *, write=False):
            check_runtime()
            return super()._check_environment(run_id, device, write=write)

    batch = RuntimeCheckedBatch(assets, upstream, state, backups, neural_device="cuda")
    if (
        batch.plan["experiment_id"] != EXPERIMENT
        or batch.plan["package_source"]["git_dirty"]
    ):
        raise ValueError("expected a clean new v2 frozen plan")
    if batch.plan["package_source"]["git_commit"] != release["code_commit"]:
        raise ValueError("frozen plan code commit differs from release")
    counts = (
        len(batch.bundle.reference_utility),
        len(batch.bundle.utility),
        int((batch.bundle.context[:, 0] == 1000).sum()),
        len(batch.plan["methods"]),
    )
    if counts != (454, 184, 26, 19):
        raise ValueError("frozen data/method counts changed")
    if batch.plan["shared_settings"]["pilot_seeds"] != [0] or batch.plan[
        "shared_settings"
    ]["formal_seeds"] != list(range(38, 46)):
        raise ValueError("frozen seeds disagree with the v2 experiment")
    check_runtime()
    return Workflow(batch, check_runtime)


class Workflow:
    """Run all pilots, request one scope-specific human review, then all seeds."""

    def __init__(self, batch, check_runtime: Callable[[], None]):
        self.batch = batch
        self.check_runtime = check_runtime

    def run_all(
        self,
        *,
        methods=None,
        settings=("multi_scale",),
        display_report=None,
        input_fn=None,
    ):
        methods = (
            tuple(entry["run_id"] for entry in self.batch.plan["methods"])
            if methods is None
            else tuple(methods)
        )
        settings = tuple(settings)
        # Validate the entire chosen scope before launching even the first pilot.
        self.check_runtime()
        preview = self.batch.preview(methods=methods, settings=settings, phase="pilot")
        if any(row["status"] == "blocked" for row in preview):
            raise RuntimeError(f"pilot queue blocked: {preview}")
        print(
            f"Full-budget pilots: {len(preview)}; formal queue: {len(methods) * len(settings) * 8} seeds.",
            flush=True,
        )
        pilot = self.batch.run(methods=methods, settings=settings, phase="pilot")
        reports = self.batch.pilot_report(methods=methods, settings=settings)
        required = {(setting, run_id) for setting in settings for run_id in methods}
        received = [(row["setting"], row["run_id"]) for row in reports]
        if (
            len(received) != len(required)
            or set(received) != required
            or any(row["status"] != "verified" for row in reports)
        ):
            raise RuntimeError(
                "pilot coverage/verification incomplete; formal training blocked"
            )
        if display_report is None:
            print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)
        else:
            display_report(reports)
        scope = {
            "plan_id": self.batch.plan["plan_id"],
            "reviewed_pilots": sorted(required),
            "formal_seeds": list(range(38, 46)),
        }
        digest = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()[
            :12
        ]
        token = f"RUN FORMAL {digest}"
        print(
            "Review every pilot's cost, memory and training diagnostics above. Finite values alone do not prove convergence.\n"
            "Do not choose budgets/seeds using pilot oracle scores. Confirm only the displayed scope: "
            f"{len(methods)} methods x {', '.join(settings)}.\n"
            f"Type exactly: {token}\nAny other answer leaves formal training unstarted.",
            flush=True,
        )
        answer = (input_fn or input)("Pilot review confirmation: ")
        if answer.strip() != token:
            print("Pilot results saved. Formal queue was not started.", flush=True)
            return {"pilot": pilot, "formal": [], "formal_started": False}
        self.check_runtime()
        from llm_design_bench.evaluation.run_artifacts import atomic_json

        atomic_json(
            self.batch.state / "reviews" / f"review-{uuid.uuid4().hex}.json",
            {
                **scope,
                "confirmation": token,
                "reviewed_at": datetime.now(UTC).isoformat(),
            },
        )
        formal = self.batch.run(
            methods=methods,
            settings=settings,
            phase="formal",
            reviewed_pilots=sorted(required),
        )
        return {"pilot": pilot, "formal": formal, "formal_started": True}

    def coverage(self, *, methods=None, settings=("multi_scale",)):
        """Read-only, artifact-verified formal coverage; no performance ranking."""
        self.check_runtime()
        rows = self.batch.preview(methods=methods, settings=settings, phase="formal")
        result = {}
        for row in rows:
            key = (row["setting"], row["run_id"])
            record = result.setdefault(
                key,
                {
                    "setting": key[0],
                    "run_id": key[1],
                    "verified_successes": 0,
                    "pending": 0,
                    "blocked": 0,
                },
            )
            record[
                {
                    "complete": "verified_successes",
                    "pending": "pending",
                    "blocked": "blocked",
                }[row["status"]]
            ] += 1
        return list(result.values())
