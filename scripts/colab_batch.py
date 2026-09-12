"""Finite, sequential queues around the frozen Colab single-job runner.

No training is performed by construction, preview, or pilot_report. A run stops
on its first failure; there are no automatic retries, budget changes, or pilot
approvals. The existing notebook's exact job identity and environment contract
are retained so verified completed seeds are reused without launching a child.

This release deliberately uses private fingerprint helpers from the *frozen*
package and journal helpers from its companion colab_support.py. Source and
release hashes are checked before execution; changing these helpers requires
compatibility tests. Only one notebook may write a STATE/BACKUPS pair.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import colab_support as support
import numpy as np
import torch
from colab_support import run_job

from llm_design_bench.evaluation.data_manifest import (
    load_data_manifest,
    make_frozen_data_recipes_task_spec,
)
from llm_design_bench.evaluation.llmdm_protocol import load_method_plan
from llm_design_bench.evaluation.run_artifacts import (
    _component,
    atomic_json,
    verify_successful_attempt,
)
from llm_design_bench.evaluation.seed_runner import (
    MethodSpec,
    SeedBenchmarkConfig,
    _base_row,
    _logical_config,
)
from llm_design_bench.optimizers.registry import get_method_metadata

CPU_METHODS = frozenset(
    {"best_logged", "random_search", "sobol", "bdi", "bo_qei", "ga_on_gp"}
)
TASK_IDS = {
    "multi_scale": "data_recipes_stack_exchange",
    "fixed_1b": "data_recipes_stack_exchange_1b",
}


def _environment(device: str) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not silently switch devices")
    packages = ("torch", "numpy", "pandas", "scipy", "scikit-learn", "plotly")
    return {
        "device": device,
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


class BatchRunner:
    """Run frozen run IDs in order; formal review is a list of (setting, run_id).

    ``methods=None`` selects all methods from the immutable plan. Settings
    default to the main experiment only. A formal queue always contains all
    eight frozen seeds; already-complete exact matches are verified and skipped.
    Preview and reports are read-only and never grant formal approval.
    """

    def __init__(
        self,
        assets: str | Path,
        data_recipes_root: str | Path,
        state: str | Path,
        backups: str | Path,
        neural_device: str = "cuda",
        allow_environment_change: bool = False,
    ) -> None:
        if neural_device not in {"cpu", "cuda"}:
            raise ValueError("neural_device must be cpu or cuda")
        self.assets = Path(assets).resolve()
        self.data_recipes_root = Path(data_recipes_root).resolve()
        self.state, self.backups = support._roots(state, backups)
        self.neural_device = neural_device
        self.allow_environment_change = allow_environment_change
        self._validate_release()

    def _validate_release(self) -> None:
        release = json.loads((self.assets / "release.json").read_text("utf-8"))
        for relative, expected in release["artifact_sha256"].items():
            path = (self.assets / relative).resolve()
            if (
                not path.is_relative_to(self.assets)
                or support._sha256(path) != expected
            ):
                raise ValueError(f"release artifact mismatch: {relative}")
        self.bundle = load_data_manifest(
            self.assets / "data", data_recipes_root=self.data_recipes_root
        )
        self.plan = load_method_plan(self.assets / "plan.json", self.bundle)
        if (
            self.bundle.manifest_id != release["data_manifest_id"]
            or self.plan["plan_id"] != release["plan_id"]
            or self.plan["package_source"]["sha256"] != release["package_source_sha256"]
        ):
            raise ValueError("release, plan, and dataset identities disagree")
        self._tasks: dict[str, Any] = {}
        self._journal_index = None

    def _jobs(self, methods, settings, phase) -> list[dict[str, Any]]:
        if phase not in {"pilot", "formal"}:
            raise ValueError("phase must be pilot or formal")
        entries = {entry["run_id"]: entry for entry in self.plan["methods"]}
        chosen = list(entries) if methods is None else list(methods)
        settings = list(settings)
        if (
            not chosen
            or len(set(chosen)) != len(chosen)
            or set(chosen) - entries.keys()
            or not settings
            or len(set(settings)) != len(settings)
            or set(settings) - TASK_IDS.keys()
        ):
            raise ValueError("select unique frozen run IDs and known settings")
        return [
            {
                "plan_id": self.plan["plan_id"],
                "run_id": run_id,
                "setting": setting,
                "phase": phase,
                "seed": seed,
                "device": "cpu"
                if entries[run_id]["method_id"] in CPU_METHODS
                else self.neural_device,
                "oracle_device": "cpu",
            }
            for setting in settings
            for run_id in chosen
            for seed in self.plan["shared_settings"][f"{phase}_seeds"]
        ]

    def _check_environment(self, run_id, device, *, write=False) -> None:
        current = _environment(device)
        path = self.state / f"environment-{run_id}.json"
        if path.exists():
            previous = json.loads(path.read_text("utf-8"))
            if previous["device"] != device:
                raise RuntimeError("device type differs from the original pilot")
            if previous != current and not self.allow_environment_change:
                raise RuntimeError(
                    "environment changed; review software/GPU comparability before "
                    "explicitly allowing the change (timings are not directly comparable)"
                )
        elif write:
            atomic_json(path, current)

    def _journal(self, job) -> dict[str, Any] | None:
        digest = hashlib.sha256(support._canonical(job)).hexdigest()
        journal = self.backups / "journal"
        # Index once per public operation. With one owner, journals can only
        # grow through our run_job calls; those new records are added below.
        # This avoids O(queue_size * completed_jobs) small reads from Drive for
        # preview/skip/report. run_job retains its own independent retry guard.
        if getattr(self, "_journal_index", None) is None:
            index = {}
            for path in sorted(journal.glob("*.intent.json")):
                intent = support._verified_json(path)
                intent_hash = hashlib.sha256(
                    support._canonical(intent["identity"])
                ).hexdigest()
                if intent_hash != intent["identity_sha256"]:
                    raise ValueError("journal identity hash mismatch")
                if intent.get("dispatch_id") != path.name.removesuffix(".intent.json"):
                    raise ValueError(
                        "intent dispatch differs from its journal filename"
                    )
                index[intent_hash] = {"intent": intent, "path": path}
            self._journal_index = index
        entry = self._journal_index.get(digest)
        if entry is None:
            return None
        if "completion" not in entry:
            path = entry["path"]
            completion = path.with_name(
                path.name.removesuffix(".intent.json") + ".completion.json"
            )
            if not completion.exists():
                raise RuntimeError("prior job was interrupted; inspect before retrying")
            entry["completion"] = support._verified_json(completion)
        saved = entry["completion"]
        if saved.get("identity_sha256") != digest:
            raise ValueError("completion belongs to a different identity")
        if saved.get("dispatch_id") != entry["intent"].get("dispatch_id"):
            raise ValueError("completion dispatch differs from its intent")
        if saved.get("status") != "success" or saved.get("returncode") != 0:
            raise RuntimeError(
                "prior dispatch was unsuccessful; inspect before retrying"
            )
        local = self.state / "_colab_jobs" / saved["dispatch_id"] / "completion.json"
        if not local.is_file() or json.loads(local.read_text("utf-8")) != {
            key: value for key, value in saved.items() if key != "snapshot"
        }:
            raise RuntimeError(
                "successful prior job lacks its local completion; restore the complete "
                "snapshot instead of silently rerunning after an older backup"
            )
        return saved

    def _remember_dispatch(self, result) -> None:
        """Verify only the newly published pair before advancing the local index."""
        journal = self.backups / "journal"
        path = journal / f"{result['dispatch_id']}.intent.json"
        intent = support._verified_json(path)
        saved = support._verified_json(
            journal / f"{result['dispatch_id']}.completion.json"
        )
        digest = hashlib.sha256(support._canonical(intent["identity"])).hexdigest()
        if digest != intent["identity_sha256"] or saved != result:
            raise ValueError("new dispatch journal does not match the runner result")
        if self._journal_index is not None:
            self._journal_index[digest] = {
                "intent": intent,
                "path": path,
                "completion": saved,
            }

    def _expected_logical(self, job) -> dict[str, Any]:
        setting = job["setting"]
        if setting not in self._tasks:
            self._tasks[setting] = make_frozen_data_recipes_task_spec(
                self.bundle,
                data_recipes_root=self.data_recipes_root,
                logged_model_scale=1000.0 if setting == "fixed_1b" else None,
                device="cpu",
            )
        task = self._tasks[setting]
        trial = task.trial_factory(
            job["seed"]
        )  # Builds data objects, not an oracle query.
        entry = next(x for x in self.plan["methods"] if x["run_id"] == job["run_id"])
        dtype = getattr(torch, entry["dtype"])
        spec = MethodSpec(entry["method_id"], entry["kwargs"], entry["run_id"], dtype)
        config = SeedBenchmarkConfig(
            experiment_id=f"{self.plan['experiment_id']}_{job['phase']}",
            phase=job["phase"],
            seeds=(job["seed"],),
            required_seeds=self.plan["shared_settings"][f"{job['phase']}_seeds"],
            task_id=task.task_id,
            dtype=dtype,
            device=job["device"],
            candidate_budget=128,
            dataset_seed=trial.dataset_seed,
            split_seed=trial.split_seed,
            normalization_reference_id=task.normalization_reference_id,
            package_commit=self.plan["package_source"]["git_commit"],
            provenance={
                "protocol_id": self.plan["shared_settings"]["protocol_id"],
                "plan_id": self.plan["plan_id"],
                "data_manifest_id": self.bundle.manifest_id,
                "package_source_sha256": self.plan["package_source"]["sha256"],
                "oracle_device": "cpu",
            },
        )
        row = _base_row(
            spec=spec,
            metadata=get_method_metadata(entry["method_id"]),
            problem=trial.problem,
            config=config,
            seed=job["seed"],
            reference_low=float(trial.reference_utility.min()),
            reference_high=float(trial.reference_utility.max()),
        )
        return _logical_config(row, trial.problem, trial.reference_utility)

    def _verified_row(self, job) -> dict[str, Any] | None:
        completion = self._journal(job)  # Check even if the CSV claims success.
        root = self.state / job["phase"]
        trial_dir = (
            root
            / "runs"
            / _component(f"{self.plan['experiment_id']}_{job['phase']}")
            / _component(TASK_IDS[job["setting"]])
            / _component(job["run_id"])
            / f"seed-{job['seed']}"
        )
        attempts = sorted(trial_dir.glob("attempt-[0-9][0-9][0-9][0-9]"))
        path = root / "method_seed_results.csv"
        records = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                required = {
                    "task_id",
                    "run_id",
                    "method_seed",
                    "phase",
                    "status",
                    "artifact_relative_dir",
                }
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError("result CSV lacks the frozen-protocol schema")
                records = [
                    row
                    for row in reader
                    if row["task_id"] == TASK_IDS[job["setting"]]
                    and row["run_id"] == job["run_id"]
                    and row["method_seed"] == str(job["seed"])
                ]
        if not records:
            if completion is not None or attempts:
                raise RuntimeError(
                    "prior attempt exists without a result row; inspect/restore"
                )
            return None
        if len(records) != 1 or records[0]["phase"] != job["phase"]:
            raise ValueError("duplicate result rows or mismatched phase")
        if records[0]["status"] != "success":
            raise RuntimeError("failed attempt requires inspection; no automatic retry")
        if completion is None or completion.get("status") != "success":
            raise RuntimeError(
                "successful artifact lacks its verified completion journal"
            )
        if not (self.state / f"environment-{job['run_id']}.json").is_file():
            raise RuntimeError(
                "successful result is missing its original environment contract"
            )
        relative = Path(records[0]["artifact_relative_dir"])
        artifact = (root / relative).resolve()
        if (
            relative.is_absolute()
            or not artifact.is_relative_to(root.resolve())
            or not attempts
            or artifact != attempts[-1].resolve()
        ):
            raise ValueError(
                "artifact must be the latest attempt inside its trial directory"
            )
        manifest, row = verify_successful_attempt(artifact)
        if manifest["logical_config"] != self._expected_logical(job):
            raise ValueError(
                "saved logical configuration differs from the frozen task/plan"
            )
        with np.load(artifact / "candidates.npz", allow_pickle=False) as archive:
            candidates = archive["candidates"]
            if (candidates < -1e-6).any() or not np.allclose(
                candidates.sum(axis=1), 1.0, rtol=0, atol=1e-6
            ):
                raise ValueError("saved candidates violate the simplex")
        for key in ("training_summary_json", "diagnostics_json"):
            if not _finite(json.loads(row[key])):
                raise ValueError(f"non-finite diagnostic: {key}")
        for key in ("method_seconds", "evaluation_seconds", "total_seconds"):
            if not isinstance(row.get(key), (float, int)) or not math.isfinite(
                row[key]
            ):
                raise ValueError(f"missing or non-finite runtime: {key}")
            if row[key] < 0:
                raise ValueError(f"negative runtime: {key}")
        peak = row.get("peak_gpu_memory_bytes")
        if (job["device"] == "cuda" or peak is not None) and (
            not isinstance(peak, (float, int)) or not math.isfinite(peak) or peak < 0
        ):
            raise ValueError("missing or invalid peak GPU allocation")
        return row

    def preview(self, methods=None, settings=("multi_scale",), phase="pilot"):
        """Read-only queue: pending, complete, or blocked; never starts a child."""
        self._validate_release()
        result = []
        for job in self._jobs(methods, settings, phase):
            try:
                self._check_environment(job["run_id"], job["device"])
                saved = self._verified_row(job)
                result.append({**job, "status": "complete" if saved else "pending"})
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                result.append({**job, "status": "blocked", "error": str(exc)})
        return result

    def pilot_report(self, methods=None, settings=("multi_scale",)):
        """Verify pilot artifacts and return cost/diagnostics, never oracle scores."""
        self._validate_release()
        reports = []
        for job in self._jobs(methods, settings, "pilot"):
            report = {"setting": job["setting"], "run_id": job["run_id"]}
            try:
                row = self._verified_row(job)
                report["status"] = "verified" if row else "missing"
                if row:
                    for key in (
                        "train_size",
                        "candidate_budget",
                        "device",
                        "dtype",
                        "method_seconds",
                        "evaluation_seconds",
                        "total_seconds",
                        "peak_gpu_memory_bytes",
                        "unique_candidate_count",
                    ):
                        report[key] = row.get(key)
                    report["training_summary"] = json.loads(
                        row["training_summary_json"]
                    )
                    report["diagnostics"] = json.loads(row["diagnostics_json"])
                    peak = row.get("peak_gpu_memory_bytes")
                    report["peak_gpu_memory_mib"] = (
                        peak / 2**20 if isinstance(peak, (float, int)) else None
                    )
                    report["artifact_checks"] = "passed"
                    # Optional null diagnostics remain visible: numerical
                    # validation is not a proof of training convergence.
                    report["finite_numeric_diagnostics"] = True
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                report.update(status="blocked", error=str(exc))
            reports.append(report)
        return reports

    def _command(self, job) -> list[str]:
        command = [
            sys.executable,
            "-u",
            "-m",
            "llm_design_bench.llmdm_cli",
            "run",
            "--data-recipes-root",
            str(self.data_recipes_root),
            "--data-bundle",
            str(self.assets / "data"),
            "--plan",
            str(self.assets / "plan.json"),
            "--phase",
            job["phase"],
            "--setting",
            job["setting"],
            "--run-id",
            job["run_id"],
            "--seed",
            str(job["seed"]),
            "--device",
            job["device"],
            "--oracle-device",
            "cpu",
            "--torch-threads",
            "1",
            "--results-dir",
            str(self.state / job["phase"]),
            "--resume",
        ]
        if job["phase"] == "formal":
            command += ["--pilot-results", str(self.state / "pilot")]
        return command

    def run(
        self, methods=None, settings=("multi_scale",), phase="pilot", reviewed_pilots=()
    ):
        """Run one finite queue, stopping on the first error, without retrying.

        Explicit human review and an independently verified seed-0 pilot are
        both required before any formal job. Approval is not inferred from a
        successful process or from pilot oracle scores.
        """
        preview = self.preview(methods, settings, phase)
        jobs = self._jobs(methods, settings, phase)
        blocked = [row for row in preview if row["status"] == "blocked"]
        if blocked:
            first = blocked[0]
            raise RuntimeError(f"queue blocked before dispatch: {first}")
        if phase == "formal":
            reviewed = set()
            for pair in reviewed_pilots:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError(
                        "reviewed_pilots must contain (setting, run_id) pairs"
                    )
                reviewed.add(tuple(pair))
            required = {(job["setting"], job["run_id"]) for job in jobs}
            if required - reviewed:
                raise RuntimeError(
                    f"human pilot review required: {sorted(required - reviewed)}"
                )
            for setting, run_id in sorted(required):
                pilot = self._jobs([run_id], [setting], "pilot")[0]
                if self._verified_row(pilot) is None:
                    raise RuntimeError(f"verified pilot missing: {setting}/{run_id}")
        results = []
        for index, job in enumerate(jobs, 1):
            label = f"{job['setting']}/{job['run_id']}/{phase}/seed-{job['seed']}"
            print(f"[{index}/{len(jobs)}] {label}", flush=True)
            try:
                self._check_environment(job["run_id"], job["device"])
                if self._verified_row(job) is not None:
                    print("  verified complete: skipped", flush=True)
                    results.append({**job, "status": "skipped"})
                    continue
                self._check_environment(job["run_id"], job["device"], write=True)
                completed = run_job(
                    self._command(job),
                    self.state,
                    self.backups,
                    job,
                    snapshot_interval=60,
                )
                self._remember_dispatch(completed)
                if self._verified_row(job) is None:
                    raise RuntimeError(
                        "child exited successfully without verified result artifacts"
                    )
                results.append({**job, "status": "completed"})
            except KeyboardInterrupt:
                print(
                    f"Queue interrupted at {label}; no following job was started.",
                    flush=True,
                )
                raise
            except Exception as exc:
                raise RuntimeError(f"Queue stopped at {label}: {exc}") from exc
        return results
