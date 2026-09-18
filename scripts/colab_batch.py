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

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, cast

import colab_support as support
import torch
from colab_support import run_job
from colab_verification import verified_result_row

from llm_design_bench.evaluation.data_manifest import (
    load_data_manifest,
    make_frozen_data_recipes_task_spec,
)
from llm_design_bench.evaluation.llmdm_protocol import load_method_plan
from llm_design_bench.evaluation.run_artifacts import (
    atomic_json,
)
from llm_design_bench.evaluation.seed_runner import (
    MethodSpec,
    SeedBenchmarkConfig,
    _base_row,
    _logical_config,
)
from llm_design_bench.optimizers.registry import get_method_metadata

if TYPE_CHECKING:
    from colab_types import (
        Device,
        DispatchCompletion,
        DispatchIntent,
        EnvironmentContract,
        JobIdentity,
        JournalEntry,
        Phase,
        PilotReport,
        QueueOutcome,
        QueuePreview,
        Setting,
    )

    from llm_design_bench.evaluation.plan_types import FrozenMethodPlan
    from llm_design_bench.evaluation.seed_types import SeedResultRow
    from llm_design_bench.evaluation.unified_report import BenchmarkTaskSpec

CPU_METHODS = frozenset(
    {"best_logged", "random_search", "sobol", "bdi", "bo_qei", "ga_on_gp"}
)
TASK_IDS: dict[Setting, str] = {
    "multi_scale": "data_recipes_stack_exchange",
    "fixed_1b": "data_recipes_stack_exchange_1b",
}


def _environment(device: Device) -> EnvironmentContract:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not silently switch devices")
    packages = ("torch", "numpy", "pandas", "scipy", "scikit-learn", "plotly")
    return {
        "device": device,
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": cast(Callable[[], int | None], torch.backends.cudnn.version)(),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


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
        neural_device: Device = "cuda",
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
        self.plan: FrozenMethodPlan = load_method_plan(
            self.assets / "plan.json", self.bundle
        )
        if (
            self.bundle.manifest_id != release["data_manifest_id"]
            or self.plan["plan_id"] != release["plan_id"]
            or self.plan["package_source"]["sha256"] != release["package_source_sha256"]
        ):
            raise ValueError("release, plan, and dataset identities disagree")
        self._tasks: dict[Setting, BenchmarkTaskSpec] = {}
        self._journal_index: dict[str, JournalEntry] | None = None

    def _jobs(
        self, methods: Iterable[str] | None, settings: Iterable[Setting], phase: Phase
    ) -> list[JobIdentity]:
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
            for seed in self._phase_seeds(phase)
        ]

    def _phase_seeds(self, phase: Phase) -> list[int]:
        shared = self.plan["shared_settings"]
        return shared["pilot_seeds"] if phase == "pilot" else shared["formal_seeds"]

    def _check_environment(
        self, run_id: str, device: Device, *, write: bool = False
    ) -> None:
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

    def _journal(self, job: JobIdentity) -> DispatchCompletion | None:
        digest = hashlib.sha256(support._canonical(job)).hexdigest()
        journal = self.backups / "journal"
        # Index once per public operation. With one owner, journals can only
        # grow through our run_job calls; those new records are added below.
        # This avoids O(queue_size * completed_jobs) small reads from Drive for
        # preview/skip/report. run_job retains its own independent retry guard.
        if getattr(self, "_journal_index", None) is None:
            index: dict[str, JournalEntry] = {}
            for path in sorted(journal.glob("*.intent.json")):
                intent = cast("DispatchIntent", support._verified_json(path))
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
        assert self._journal_index is not None
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
            entry["completion"] = cast(
                "DispatchCompletion", support._verified_json(completion)
            )
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

    def _remember_dispatch(self, result: DispatchCompletion) -> None:
        """Verify only the newly published pair before advancing the local index."""
        journal = self.backups / "journal"
        path = journal / f"{result['dispatch_id']}.intent.json"
        intent = cast("DispatchIntent", support._verified_json(path))
        saved = cast(
            "DispatchCompletion",
            support._verified_json(
                journal / f"{result['dispatch_id']}.completion.json"
            ),
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

    def _expected_logical(self, job: JobIdentity) -> dict[str, object]:
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
            required_seeds=tuple(self._phase_seeds(job["phase"])),
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

    def _verified_row(self, job: JobIdentity) -> SeedResultRow | None:
        # The journal must be checked even when the CSV claims success. Keep
        # this orchestration hook so callers can revalidate immediately before
        # dispatch; the extracted verifier itself never starts a child.
        completion = self._journal(job)
        return verified_result_row(
            state=self.state,
            experiment_id=self.plan["experiment_id"],
            task_id=TASK_IDS[job["setting"]],
            job=job,
            completion=completion,
            expected_logical=lambda: self._expected_logical(job),
        )

    def preview(
        self,
        methods: Iterable[str] | None = None,
        settings: Iterable[Setting] = ("multi_scale",),
        phase: Phase = "pilot",
    ) -> list[QueuePreview]:
        """Read-only queue: pending, complete, or blocked; never starts a child."""
        self._validate_release()
        result: list[QueuePreview] = []
        for job in self._jobs(methods, settings, phase):
            try:
                self._check_environment(job["run_id"], job["device"])
                saved = self._verified_row(job)
                result.append({**job, "status": "complete" if saved else "pending"})
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                result.append({**job, "status": "blocked", "error": str(exc)})
        return result

    def pilot_report(
        self,
        methods: Iterable[str] | None = None,
        settings: Iterable[Setting] = ("multi_scale",),
    ) -> list[PilotReport]:
        """Verify pilot artifacts and return cost/diagnostics, never oracle scores."""
        self._validate_release()
        reports: list[PilotReport] = []
        for job in self._jobs(methods, settings, "pilot"):
            report: PilotReport = {
                "setting": job["setting"],
                "run_id": job["run_id"],
                "status": "missing",
            }
            try:
                row = self._verified_row(job)
                report["status"] = "verified" if row else "missing"
                if row:
                    report["train_size"] = row.get("train_size")
                    report["candidate_budget"] = row.get("candidate_budget")
                    report["device"] = row.get("device")
                    report["dtype"] = row.get("dtype")
                    report["method_seconds"] = row.get("method_seconds")
                    report["evaluation_seconds"] = row.get("evaluation_seconds")
                    report["total_seconds"] = row.get("total_seconds")
                    report["peak_gpu_memory_bytes"] = row.get("peak_gpu_memory_bytes")
                    report["unique_candidate_count"] = row.get("unique_candidate_count")
                    report["training_summary"] = json.loads(
                        cast(str, row["training_summary_json"])
                    )
                    report["diagnostics"] = json.loads(
                        cast(str, row["diagnostics_json"])
                    )
                    peak = row.get("peak_gpu_memory_bytes")
                    report["peak_gpu_memory_mib"] = (
                        peak / 2**20 if isinstance(peak, (float, int)) else None
                    )
                    report["artifact_checks"] = "passed"
                    # Optional null diagnostics remain visible: numerical
                    # validation is not a proof of training convergence.
                    report["finite_numeric_diagnostics"] = True
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                report["status"] = "blocked"
                report["error"] = str(exc)
            reports.append(report)
        return reports

    def _command(self, job: JobIdentity) -> list[str]:
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
        self,
        methods: Iterable[str] | None = None,
        settings: Iterable[Setting] = ("multi_scale",),
        phase: Phase = "pilot",
        reviewed_pilots: Iterable[tuple[Setting, str]] = (),
    ) -> list[QueueOutcome]:
        """Run one finite queue, stopping on the first error, without retrying.

        Explicit human review and an independently verified seed-0 pilot are
        both required before any formal job. Approval is not inferred from a
        successful process or from pilot oracle scores.
        """
        # Preview and dispatch must see the same selection, including when a
        # caller supplies a one-shot iterable. None still selects all methods.
        methods = None if methods is None else tuple(methods)
        settings = tuple(settings)
        preview = self.preview(methods, settings, phase)
        jobs = self._jobs(methods, settings, phase)
        blocked = [row for row in preview if row["status"] == "blocked"]
        if blocked:
            first = blocked[0]
            raise RuntimeError(f"queue blocked before dispatch: {first}")
        if phase == "formal":
            reviewed: set[tuple[Setting, str]] = set()
            for pair in reviewed_pilots:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError(
                        "reviewed_pilots must contain (setting, run_id) pairs"
                    )
                reviewed.add((pair[0], pair[1]))
            required = {(job["setting"], job["run_id"]) for job in jobs}
            if required - reviewed:
                raise RuntimeError(
                    f"human pilot review required: {sorted(required - reviewed)}"
                )
            for setting, run_id in sorted(required):
                pilot = self._jobs([run_id], [setting], "pilot")[0]
                if self._verified_row(pilot) is None:
                    raise RuntimeError(f"verified pilot missing: {setting}/{run_id}")
        results: list[QueueOutcome] = []
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
