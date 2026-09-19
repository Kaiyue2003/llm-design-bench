"""Exercise the installed formal CLI with invented data, never real experiments.

All 27 reviewed methods receive tiny *engineering-test* budgets and the protocol's
K=128. The pilot covers every method; a one-seed Best Logged formal shard checks
the pilot gate, saved artifacts and resume. It must remain incomplete/unranked.
No published data, checkpoint, experiment plan or result is read or overwritten.
This script is a container/source test utility, not another benchmark entry point.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from llm_design_bench.evaluation.formal_methods import FORMAL_METHOD_IDS
from llm_design_bench.evaluation.run_artifacts import verify_successful_attempt
from llm_design_bench.tasks.data_recipes import DOMAIN_ORDER

_NEURAL = {"hidden_size": 8, "epochs": 1, "batch_size": 8, "particle_steps": 1}
_SURROGATE = {
    "hidden_size": 8,
    "surrogate_epochs": 1,
    "batch_size": 8,
    "solver_steps": 1,
}
_GENERATIVE = {"hidden_size": 8, "epochs": 1, "batch_size": 8, "steps": 1}
_DIFFUSION = {**_GENERATIVE, "diffusion_steps": 8}

# These are explicitly NOT proposed experimental budgets. Keep them small and
# separate from shipped experiment plans; defaults are expanded by the real CLI.
SMOKE_METHOD_KWARGS = {
    "best_logged": {},
    "random_search": {},
    "sobol": {},
    "offline_mlp": {**_NEURAL},
    "standard_ga": {**_SURROGATE},
    "coms": {**_NEURAL, "adversarial_steps": 1},
    "bdi": {"steps": 1},
    "ga_on_gp": {"gp_training_steps": 1, "solver_steps": 1},
    "bo_qei": {"gp_training_steps": 1, "acquisition_steps": 1, "mc_samples": 4},
    "cma_es": {
        "hidden_size": 8,
        "surrogate_epochs": 1,
        "batch_size": 8,
        "ensemble_size": 2,
        "generations": 1,
        "population_size": 4,
    },
    "reinforce": {
        "hidden_size": 8,
        "surrogate_epochs": 1,
        "batch_size": 8,
        "ensemble_size": 2,
        "iterations": 1,
        "reinforce_batch_size": 8,
    },
    "mc_dropout": {**_NEURAL, "mc_samples": 4},
    "roma": {**_SURROGATE, "weight_perturbation_steps": 1, "adaptation_steps": 1},
    "ict": {
        **_SURROGATE,
        "adaptation_steps": 1,
        "neighbor_samples": 4,
        "remember_count": 2,
        "surrogate_learning_rate": 0.01,
    },
    "tri_mentoring": {**_SURROGATE, "neighbor_samples": 4},
    "ltr": {
        **_SURROGATE,
        "list_length": 3,
        "lists_per_epoch": 3,
        "validation_lists": 2,
    },
    "match_opt": {
        "embedding_dim": 2,
        "surrogate_epochs": 1,
        "batch_size": 8,
        "solver_steps": 1,
    },
    "pgs": {
        **_SURROGATE,
        "rl_steps": 1,
        "cql_samples": 2,
        "trajectories_per_group": 2,
        "top_fraction": 1.0,
        "max_horizon": 3,
    },
    "cbas": {
        **_GENERATIVE,
        "population_size": 16,
        "adaptation_epochs": 1,
        "ensemble_size": 2,
    },
    "mins": {**_GENERATIVE, "target_grid_size": 2},
    "ddom": {**_DIFFUSION},
    "gabo": {
        **_GENERATIVE,
        "initial_points": 4,
        "acquisition_steps": 2,
        "acquisition_restarts": 2,
        "critic_steps": 2,
    },
    "gtg": {**_DIFFUSION, "trajectory_length": 3, "trajectory_count": 8},
    "rgd": {
        **_DIFFUSION,
        "refinement_rounds": 1,
        "refinement_candidates": 2,
        "likelihood_samples": 2,
        "likelihood_steps": 3,
    },
    "bonet": {**_GENERATIVE, "trajectory_length": 3, "trajectory_count": 8},
    "demo": {**_DIFFUSION, "editing_epochs": 1, "pseudo_samples": 8},
    "root": {**_GENERATIVE},
}


def smoke_methods(method_ids: Sequence[str] | None = None) -> list[dict]:
    if set(SMOKE_METHOD_KWARGS) != set(FORMAL_METHOD_IDS):
        raise ValueError(
            "Update the smoke configurations for the reviewed method roster"
        )
    selected = tuple(FORMAL_METHOD_IDS if method_ids is None else method_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select a nonempty, unique method list")
    if set(selected) - set(FORMAL_METHOD_IDS):
        raise ValueError("Smoke methods must belong to the non-SPADE formal roster")
    if "best_logged" not in selected:
        raise ValueError("The resume/gate smoke requires best_logged")
    return [
        {"method_id": name, "kwargs": dict(SMOKE_METHOD_KWARGS[name])}
        for name in selected
    ]


def _write_fixture(root: Path) -> Path:
    checkpoint = root / "opt_algos/data_models/smoke/checkpoints/checkpoint_latest.pt"
    checkpoint.parent.mkdir(parents=True)
    (root / "results").mkdir()
    torch.save({"invented_container_fixture": True}, checkpoint)
    for sidecar in ("config.json", "feature_mask.json"):
        (checkpoint.parent.parent / sidecar).write_text("{}\n", encoding="utf-8")
    source = (
        "from pathlib import Path\nimport torch\n"
        "class DataModelBenchmark:\n"
        "    def __init__(self, metric_index=4, device='cpu'):\n"
        f"        feature_names = {dict(enumerate(DOMAIN_ORDER))!r}\n"
        "        checkpoint = Path(__file__).parent / 'data_models/smoke/checkpoints/checkpoint_latest.pt'\n"
        "        assert torch.load(checkpoint) == {'invented_container_fixture': True}\n"
        "    def _raw_func_with_model_scale(self, z, m, x, with_exp=True):\n"
        "        assert z == 195 and m == 100 and with_exp is False\n"
        "        with (Path(__file__).parents[1] / 'oracle_calls.log').open('a') as stream:\n"
        "            stream.write('evaluate\\n')\n"
        "        return 1.0 + sum(float(v) * (i + 1) / 10 for i, v in enumerate(x))\n"
    )
    (root / "opt_algos/benchmarks.py").write_text(source, encoding="utf-8")
    rows = []
    for index in range(12):
        for group, offset in (("20M", 2.0), ("1B", 1.0)):
            mixture = np.array([1 + index, 2, 3, 4, 5], dtype=np.float64)
            mixture /= mixture.sum()
            rows.append(
                {
                    "group": group,
                    "token_probabilities": mixture,
                    "history": pd.DataFrame(
                        {
                            "_step": [19500],
                            "eval/RedPajamaStackExchange/CrossEntropyLoss": [
                                offset + index / 10
                            ],
                        }
                    ),
                }
            )
    pd.DataFrame(rows).to_pickle(root / "results/data_mixing_runs.pkl")
    return checkpoint


def _run_cli(prefix: Sequence[str], arguments: list[str], log: Path) -> None:
    print(f"[container smoke] {log.stem}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            [*prefix, *arguments],
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Smoke CLI failed ({completed.returncode}); see {log}\n{log.read_text(encoding='utf-8')[-6000:]}"
        )


def _verify_phase(
    directory: Path, expected_methods: set[str], seed: int
) -> pd.DataFrame:
    rows = pd.read_csv(directory / "method_seed_results.csv")
    if len(rows) != len(expected_methods) or set(rows["method_id"]) != expected_methods:
        raise AssertionError("Smoke method coverage mismatch")
    if set(rows["method_seed"]) != {seed} or set(rows["status"]) != {"success"}:
        raise AssertionError("Smoke seed coverage or success status mismatch")
    if rows.duplicated(["task_id", "run_id", "method_seed"]).any():
        raise AssertionError("Duplicate smoke rows")
    for _, row in rows.iterrows():
        relative = Path(row["artifact_relative_dir"])
        attempt = (directory / relative).resolve()
        if relative.is_absolute() or not attempt.is_relative_to(directory.resolve()):
            raise AssertionError("Artifact path escaped its result directory")
        manifest, saved = verify_successful_attempt(attempt)
        if (
            saved["status"] != "success"
            or manifest["logical_config"]["candidate_budget"] != 128
        ):
            raise AssertionError("Invalid smoke attempt")
        with np.load(attempt / "candidates.npz", allow_pickle=False) as candidates:
            values = candidates["candidates"]
            if (
                values.shape != (128, 5)
                or not np.isfinite(values).all()
                or (values < -1e-6).any()
            ):
                raise AssertionError("Invalid smoke candidate batch")
            np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-5)
            np.testing.assert_array_equal(
                candidates["target_context"], [1000.0, 19500.0]
            )
        with np.load(attempt / "evaluation.npz", allow_pickle=False) as evaluated:
            if (
                evaluated["utility"].shape != (128,)
                or not np.isfinite(evaluated["utility"]).all()
            ):
                raise AssertionError("Invalid smoke evaluation")
            np.testing.assert_array_equal(evaluated["raw_loss"], -evaluated["utility"])
    return rows


def run_smoke(
    output: Path,
    *,
    method_ids: Sequence[str] | None = None,
    command_prefix: Sequence[str] = ("llm-design-bench",),
) -> dict:
    methods = smoke_methods(method_ids)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "SMOKE_ONLY.json").write_text(
        json.dumps({"invented_data": True, "scientific_result": False}),
        encoding="utf-8",
    )
    upstream, bundle, plan = output / "fixture", output / "bundle", output / "plan.json"
    checkpoint = _write_fixture(upstream)
    methods_file = output / "smoke-methods.json"
    methods_file.write_text(json.dumps(methods), encoding="utf-8")
    _run_cli(
        command_prefix,
        [
            "prepare",
            "--data-recipes-root",
            str(upstream),
            "--oracle-checkpoint",
            str(checkpoint),
            "--output",
            str(bundle),
        ],
        output / "01-prepare.log",
    )
    _run_cli(
        command_prefix,
        [
            "freeze",
            "--data-recipes-root",
            str(upstream),
            "--data-bundle",
            str(bundle),
            "--methods-file",
            str(methods_file),
            "--experiment-id",
            "invented-container-smoke-only",
            "--output",
            str(plan),
        ],
        output / "02-freeze.log",
    )
    oracle_log = upstream / "oracle_calls.log"
    if oracle_log.exists():
        raise AssertionError("Preparation/freezing accessed the oracle")
    base = [
        "run",
        "--data-recipes-root",
        str(upstream),
        "--data-bundle",
        str(bundle),
        "--plan",
        str(plan),
        "--device",
        "cpu",
        "--oracle-device",
        "cpu",
        "--torch-threads",
        "1",
    ]
    pilot, formal = output / "pilot", output / "formal-shard"
    _run_cli(
        command_prefix,
        [*base, "--phase", "pilot", "--results-dir", str(pilot)],
        output / "03-pilot.log",
    )
    _verify_phase(pilot, {entry["method_id"] for entry in methods}, 0)
    formal_args = [
        *base,
        "--phase",
        "formal",
        "--results-dir",
        str(formal),
        "--pilot-results",
        str(pilot),
        "--run-id",
        "best_logged",
        "--seed",
        "38",
    ]
    _run_cli(command_prefix, formal_args, output / "04-formal-shard.log")
    before_rows = _verify_phase(formal, {"best_logged"}, 38)
    calls_before = oracle_log.read_bytes()
    _run_cli(command_prefix, [*formal_args, "--resume"], output / "05-resume.log")
    resumed = _verify_phase(formal, {"best_logged"}, 38)
    pd.testing.assert_frame_equal(before_rows, resumed)
    if oracle_log.read_bytes() != calls_before:
        raise AssertionError("Resume queried the fake oracle again")
    summary = pd.read_csv(formal / "method_seed_summary.csv")
    if (
        len(summary) != 1
        or summary["rank_eligible"].any()
        or summary["complete_seed_set"].any()
    ):
        raise AssertionError("A one-seed smoke shard must not be ranked as complete")
    if summary.iloc[0]["requested_runs"] != 8 or summary.iloc[0]["missing_runs"] != 7:
        raise AssertionError("Smoke shard lost the formal eight-seed requirement")
    result = {
        "status": "passed",
        "scientific_result": False,
        "invented_data": True,
        "pilot_methods": len(methods),
        "formal_shard_rows": 1,
        "formal_missing_seeds": 7,
        "formal_rank_eligible": False,
        "resume_reused_attempt": True,
        "candidate_budget": 128,
    }
    (output / "smoke_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Engineering smoke passed; NOT benchmark results: {output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    # Reserve a unique name without ever selecting an existing experiment folder.
    reserved = Path(tempfile.mkdtemp(prefix="invented-smoke-", dir=args.results_root))
    run_smoke(reserved / "run")


if __name__ == "__main__":
    main()
