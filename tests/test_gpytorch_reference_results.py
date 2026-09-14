"""Read-only CSV consistency checks for the archived formal GPyTorch run.

These tests inspect committed files only. They do not train methods, query the
oracle, or claim to validate the separately retained raw experiment artifacts.
"""

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reference_results/llmdm_forward_gpytorch_v2"
FROZEN = ROOT / "experiments/llmdm_forward_gpytorch_v2"
SEEDS = frozenset(range(38, 46))
CSV_HASHES = {
    "method_seed_results.csv": (
        "8a98934cd6da1b0e8612e6df96c4dd7f4111c482a6bc6efa604ce65ce7d1b3a0"
    ),
    "method_seed_summary.csv": (
        "96edb7a3f74bcc08e2b6bf5b4450dab0986626f28b7e9ec7cb1879e877752123"
    ),
}
CPU_METHODS = frozenset(
    {"best_logged", "random_search", "sobol", "bdi", "bo_qei", "ga_on_gp"}
)
METRICS = (
    "raw_max_utility",
    "raw_median_utility",
    "raw_mean_utility",
    "refnorm_max_score",
    "refnorm_median_score",
    "refnorm_mean_score",
    "raw_min_loss",
    "raw_median_loss",
    "raw_mean_loss",
    "method_seconds",
    "evaluation_seconds",
    "total_seconds",
    "unique_candidate_count",
    "unique_candidate_fraction",
    "candidate_diversity",
    "candidate_novelty",
    "mean_mixture_entropy",
    "mean_active_domain_count",
)
TABLE_HEADER = (
    "Rank",
    "Method",
    "Best refnorm score (mean +/- SD)",
    "Best loss (mean +/- SD)",
    "Method seconds (mean)",
)


def _read_csv(name):
    with (ARCHIVE / name).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
        rows = list(reader)
    assert rows
    assert all(None not in row and None not in row.values() for row in rows)
    return rows


def _number(row, key):
    value = float(row[key])
    assert math.isfinite(value), (row.get("run_id"), key, row[key])
    return value


@pytest.fixture(scope="module")
def archived():
    return {
        "rows": _read_csv("method_seed_results.csv"),
        "summary": _read_csv("method_seed_summary.csv"),
        "plan": json.loads((FROZEN / "plan.json").read_text("utf-8")),
        "release": json.loads((FROZEN / "release.json").read_text("utf-8")),
        "metadata": json.loads((ARCHIVE / "run_metadata.json").read_text("utf-8")),
        "readme": (ARCHIVE / "README.md").read_text("utf-8"),
    }


@pytest.mark.parametrize("filename", CSV_HASHES)
def test_reference_csv_bytes_and_metadata_are_unchanged(archived, filename):
    content = (ARCHIVE / filename).read_bytes()
    expected_rows = 152 if filename == "method_seed_results.csv" else 19
    recorded = archived["metadata"]["files"][filename]
    assert hashlib.sha256(content).hexdigest() == CSV_HASHES[filename]
    assert recorded["sha256"] == CSV_HASHES[filename]
    assert recorded["bytes"] == len(content)
    assert recorded["rows"] == expected_rows == len(_read_csv(filename))
    assert recorded["columns"] == len(_read_csv(filename)[0])


def test_reference_contains_exactly_nineteen_complete_formal_seed_sets(archived):
    rows = archived["rows"]
    summary = archived["summary"]
    methods = {entry["run_id"] for entry in archived["plan"]["methods"]}
    assert len(methods) == 19
    assert len(rows) == 152
    identities = [(row["run_id"], _number(row, "method_seed")) for row in rows]
    assert len(set(identities)) == len(identities)
    assert set(identities) == {(run_id, seed) for run_id in methods for seed in SEEDS}
    assert {row["status"] for row in rows} == {"success"}
    assert all(not row["error_type"] and not row["error_message"] for row in rows)
    assert len(summary) == 19
    assert {row["run_id"] for row in summary} == methods
    for row in summary:
        assert _number(row, "requested_runs") == 8
        assert _number(row, "attempted_runs") == 8
        assert _number(row, "successful_runs") == 8
        assert _number(row, "failed_runs") == 0
        assert _number(row, "missing_runs") == 0
        assert row["complete_seed_set"].lower() == "true"
        assert row["rank_eligible"].lower() == "true"


def test_reference_rows_match_the_frozen_main_experiment(archived):
    plan, release = archived["plan"], archived["release"]
    methods = {entry["run_id"]: entry for entry in plan["methods"]}
    expected_provenance = {
        "protocol_id": plan["shared_settings"]["protocol_id"],
        "plan_id": plan["plan_id"],
        "data_manifest_id": plan["data_manifest_id"],
        "package_source_sha256": plan["package_source"]["sha256"],
        "oracle_device": "cpu",
    }
    assert plan["plan_id"] == release["plan_id"]
    assert plan["data_manifest_id"] == release["data_manifest_id"]
    assert plan["shared_settings"]["objective"] == (
        "eval/RedPajamaStackExchange/CrossEntropyLoss"
    )
    for row in archived["rows"]:
        entry = methods[row["run_id"]]
        assert row["experiment_id"] == plan["experiment_id"] + "_formal"
        assert row["task_id"] == "data_recipes_stack_exchange"
        assert row["phase"] == "formal"
        assert row["method_id"] == entry["method_id"]
        assert row["objective_name"] == "stack_exchange_cross_entropy"
        assert json.loads(row["method_config_json"]) == entry["kwargs"]
        assert json.loads(row["requested_method_config_json"]) == entry["kwargs"]
        assert json.loads(row["provenance_json"]) == expected_provenance
        assert row["package_commit"] == plan["package_source"]["git_commit"]
        assert json.loads(row["required_seeds_json"]) == sorted(SEEDS)
        assert _number(row, "candidate_budget") == 128
        assert _number(row, "train_size") == 184
        assert json.loads(row["target_context_json"]) == [1000, 19500]
        assert row["dtype"] == "torch." + entry["dtype"]
        assert row["device"] == ("cpu" if entry["method_id"] in CPU_METHODS else "cuda")
        assert row["normalization_reference_id"] == (
            f"llmdm_{plan['data_manifest_id']}_full_logged"
        )
        problem = json.loads(row["problem_metadata_json"])
        assert problem["utility_transform"] == "negative_loss"
        assert problem["manifest_id"] == plan["data_manifest_id"]
        assert len(problem["visible_row_ids"]) == 184
    for row in archived["summary"]:
        entry = methods[row["run_id"]]
        assert row["experiment_id"] == plan["experiment_id"] + "_formal"
        assert row["task_id"] == "data_recipes_stack_exchange"
        assert row["phase"] == "formal"
        assert row["method_id"] == entry["method_id"]
        assert row["package_commit"] == plan["package_source"]["git_commit"]
        assert json.loads(row["method_config_json"]) == entry["kwargs"]
        assert json.loads(row["required_seeds_json"]) == sorted(SEEDS)


@pytest.mark.parametrize("metric", METRICS)
def test_reference_summary_recomputes_sample_sd_and_se(archived, metric):
    by_method = defaultdict(list)
    for row in archived["rows"]:
        by_method[row["run_id"]].append(_number(row, metric))
    for summary in archived["summary"]:
        values = by_method[summary["run_id"]]
        assert len(values) == _number(summary, f"{metric}_n") == 8
        mean, std = statistics.mean(values), statistics.stdev(values)
        assert _number(summary, f"{metric}_mean") == pytest.approx(
            mean, rel=1e-10, abs=1e-12
        )
        assert _number(summary, f"{metric}_std") == pytest.approx(
            std, rel=1e-10, abs=1e-12
        )
        assert _number(summary, f"{metric}_se") == pytest.approx(
            std / math.sqrt(8), rel=1e-10, abs=1e-12
        )


def test_reference_loss_direction_and_fixed_reference_are_consistent(archived):
    references = {
        (_number(row, "reference_min_utility"), _number(row, "reference_max_utility"))
        for row in archived["rows"]
    }
    assert len(references) == 1
    low, high = references.pop()
    assert high > low
    for row in archived["rows"]:
        for utility_stat, loss_stat in (
            ("max", "min"),
            ("median", "median"),
            ("mean", "mean"),
        ):
            utility = _number(row, f"raw_{utility_stat}_utility")
            loss = _number(row, f"raw_{loss_stat}_loss")
            assert loss == pytest.approx(-utility, rel=1e-10, abs=1e-12)
            assert _number(row, f"refnorm_{utility_stat}_score") == pytest.approx(
                (utility - low) / (high - low), rel=1e-10, abs=1e-12
            )


def test_reference_task_ranks_follow_mean_best_candidate_score(archived):
    scores = [_number(row, "refnorm_max_score_mean") for row in archived["summary"]]
    for row, score in zip(archived["summary"], scores, strict=True):
        better = sum(other > score for other in scores)
        tied = sum(other == score for other in scores)
        assert _number(row, "task_rank") == 1 + better + (tied - 1) / 2


def test_reference_readme_reports_all_methods_with_mean_plus_sample_sd(archived):
    lines = archived["readme"].splitlines()
    header = "| " + " | ".join(TABLE_HEADER) + " |"
    start = lines.index(header)
    table_rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        assert len(cells) == 5
        table_rows.append(cells)
    summary = sorted(archived["summary"], key=lambda row: _number(row, "task_rank"))
    assert len(table_rows) == len(summary) == 19
    for cells, row in zip(table_rows, summary, strict=True):
        rank = _number(row, "task_rank")
        assert rank.is_integer()
        seconds = _number(row, "method_seconds_mean")
        expected = (
            str(int(rank)),
            row["method_display_name"],
            (
                f"{_number(row, 'refnorm_max_score_mean'):.6f} +/- "
                f"{_number(row, 'refnorm_max_score_std'):.6f}"
            ),
            (
                f"{_number(row, 'raw_min_loss_mean'):.6f} +/- "
                f"{_number(row, 'raw_min_loss_std'):.6f}"
            ),
            f"{seconds:.2f}" if seconds >= 0.01 else f"{seconds:.3f}",
        )
        assert cells == expected
    labels = {row["run_id"]: row["method_display_name"] for row in summary}
    assert labels["bdi"] == "BDI adaptation"
    assert labels["ga_on_gp"] == "GA on GP adaptation"
    assert labels["bo_qei"] == "BO-qEI adaptation"


def test_reference_metadata_keeps_retry_history_and_audit_limits(archived):
    metadata = archived["metadata"]
    assert metadata["audit"]["scope"] == "csv_consistency_only"
    assert metadata["audit"]["raw_artifacts_verified"] is False
    expected = {}
    for row in archived["rows"]:
        reason = row["infrastructure_retry_reason"]
        if reason:
            expected[(row["run_id"], int(row["method_seed"]))] = {
                "run_id": row["run_id"],
                "method_seed": int(row["method_seed"]),
                "reason": reason,
                "artifact_relative_dir": row["artifact_relative_dir"],
            }
    assert set(expected) == {("roma", 41), ("roma", 45)}
    retries = metadata["infrastructure_retries"]
    assert len(retries) == len(expected) == 2
    observed = {(entry["run_id"], entry["method_seed"]): entry for entry in retries}
    assert observed.keys() == expected.keys()
    for key, record in expected.items():
        assert {field: observed[key][field] for field in record} == record


def test_reference_metadata_identifies_only_this_frozen_formal_main_run(archived):
    metadata, release, plan = (
        archived["metadata"],
        archived["release"],
        archived["plan"],
    )
    manifest = json.loads((FROZEN / "data/manifest.json").read_text("utf-8"))[
        "metadata"
    ]
    for key in (
        "protocol_id",
        "code_commit",
        "upstream_commit",
        "package_source_sha256",
        "plan_id",
        "data_manifest_id",
    ):
        assert metadata[key] == release[key]
    assert metadata["experiment_id"] == plan["experiment_id"] + "_formal"
    assert metadata["phase"] == "formal"
    assert metadata["setting"] == "multi_scale"
    assert metadata["task_id"] == "data_recipes_stack_exchange"
    coverage = metadata["coverage"]
    assert coverage["methods"] == len(archived["summary"]) == 19
    assert coverage["method_seeds"] == sorted(SEEDS)
    assert coverage["expected_final_results"] == len(archived["rows"]) == 152
    assert coverage["successful_final_results"] == 152
    assert coverage["failed_final_results"] == coverage["missing_final_results"] == 0
    assert coverage["pilot_results_included"] is False
    assert coverage["fixed_1b_results_included"] is False
    data = metadata["data"]
    for key in (
        "source_rows",
        "empty_histories",
        "usable_logged",
        "main_visible",
        "fixed_1b_visible",
    ):
        assert data[key] == release["counts"][key]
    for key in ("objective", "utility_transform", "domain_order", "target_context"):
        assert data[key] == manifest[key]
    for key, value in data["split"].items():
        assert value == manifest["split"][key]
    assert data["optimization_direction"] == "maximize"
    assert data["fixed_across_methods_and_seeds"] is True
    evaluation = metadata["evaluation"]
    assert evaluation["candidate_budget"] == plan["shared_settings"]["candidate_budget"]
    assert evaluation["primary_metric"] == plan["shared_settings"]["primary_score"]
    assert evaluation["table_uncertainty"] == "sample_standard_deviation"
    assert evaluation["standard_deviation_ddof"] == 1
    assert evaluation["real_llm_pretraining_performed"] is False
    assert metadata["raw_artifact_storage"]["included_in_this_directory"] is False
    for row in archived["rows"]:
        assert (
            row["normalization_reference_id"]
            == evaluation["normalization_reference_id"]
        )
        for key in ("reference_min_utility", "reference_max_utility"):
            assert _number(row, key) == evaluation[key]
    for key, relative in (
        ("release", "release.json"),
        ("plan", "plan.json"),
        ("data_manifest", "data/manifest.json"),
    ):
        assert (ARCHIVE / metadata["frozen_inputs"][key]).resolve() == (
            FROZEN / relative
        ).resolve()
    assert "CSV-level consistency verification only" in archived["readme"]
    assert metadata["code_commit"] in archived["readme"]
    assert metadata["plan_id"] in archived["readme"]


def test_reference_metadata_runtime_and_device_policy_match_every_seed(archived):
    metadata, plan = archived["metadata"], archived["plan"]
    runtime = metadata["recorded_runtime"]
    policy = metadata["execution_policy"]
    methods = {entry["method_id"] for entry in plan["methods"]}
    assert set(policy["cpu_methods"]) == CPU_METHODS
    assert set(policy["cuda_methods"]) == methods - CPU_METHODS
    assert (
        sorted(policy["float64_methods"]) == plan["shared_settings"]["float64_methods"]
    )
    assert (
        policy["other_methods_dtype"] == plan["shared_settings"]["other_methods_dtype"]
    )
    assert (
        policy["mixed_precision"] is plan["shared_settings"]["mixed_precision"] is False
    )
    assert policy["oracle_device"] == "cpu"
    for key, value in archived["release"]["gp_backend"].items():
        assert metadata["gp_backend"][key] == value
    for row in archived["rows"]:
        environment = json.loads(row["environment_json"])
        for key in (
            "python",
            "platform",
            "machine",
            "logical_cpu_count",
            "torch",
            "numpy",
            "cuda_runtime",
            "cudnn_version",
            "torch_threads",
            "deterministic_algorithms",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "float32_matmul_precision",
        ):
            assert environment[key] == runtime[key], (row["run_id"], key)
        assert environment["device"] == row["device"]
        assert environment["gpu"] == (
            runtime["gpu_for_cuda_methods"] if row["device"] == "cuda" else None
        )
        inventory = {
            name.lower().replace("-", "_"): version
            for name, version in environment["installed_packages"].items()
        }
        for name, version in metadata["gp_backend"]["packages"].items():
            assert inventory[name] == version
    for value in (
        runtime["python"].split()[0],
        runtime["torch"],
        runtime["gpu_for_cuda_methods"]["name"],
    ):
        assert value in archived["readme"]
