import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reference_results" / "publication"

METHODS = {"best_logged", "coms", "bdi"}
TASKS = {
    "data_recipes_stack_exchange",
    "ackley",
    "schaffer2",
    "sum_different_powers",
    "matyas",
    "power_sum",
    "rosenbrock",
    "michalewicz",
    "hartmann6",
    "shekel",
}
SEEDS = set(range(38, 46))
AUDITED_SHA256 = {
    "README.md": "333fb0aac2cc1d20d1c2fa69f4be7ffcfcd77892ab910d81ef717ee7acbd976f",
    "d_best_summary.csv": "8bf644237af47008d62dca05b9f754134bf32959ec147aee94f30b7dedf14157",
    "rank_summary.csv": "086e403bd234c1d2d5ed1c46647249d2d523af51e7f1c342ea727671f0e06682",
    "raw_runs.csv": "e33544a5ef76f8397d9fc44e4be392f79cb27c43ef3070cc7fea515acfc140b8",
    "run_metadata.json": "5cecf228a67ba59d1444983ac33d3bf03beb7b0ee0e32abb15846bc395776eb7",
    "seed_manifest.csv": "7018a179609c580f1a0198ce12c2f8c4374fc52ffe418f12d639d44083a40946",
    "seeded_benchmark_table.tex": "1b18618b33d810d0495b8fb726cdf2199b1bc8c8f48b14ac7a00f9a1779bee21",
    "task_summary.csv": "a0a79887bd444e0c8d110d5bd01fef918f4dde4eba65b7e4397351170e3c34ab",
}


def test_publication_v1_files_match_audited_fingerprints() -> None:
    for filename, expected in AUDITED_SHA256.items():
        digest = hashlib.sha256((RESULTS / filename).read_bytes()).hexdigest()
        assert digest == expected, f"{filename} changed without updating the v1 audit"


def test_publication_v1_raw_contract_is_frozen() -> None:
    raw = pd.read_csv(RESULTS / "raw_runs.csv")
    metadata = json.loads((RESULTS / "run_metadata.json").read_text(encoding="utf-8"))

    assert len(raw) == len(METHODS) * len(TASKS) * len(SEEDS) == 240
    assert set(raw["optimizer"]) == METHODS
    assert set(raw["task"]) == TASKS
    assert set(raw["seed"]) == SEEDS
    assert not raw.duplicated(["task", "seed", "optimizer"]).any()
    assert (raw["recommendation_count"] == 128).all()
    assert (raw["query_count"] == 0).all()
    assert (raw["cumulative_simulated_cost"] == 0.0).all()
    assert raw.notna().all().drop(labels=["task_seed", "best_objective"]).all()

    expected_pairs = pd.MultiIndex.from_product(
        [sorted(TASKS), sorted(SEEDS), sorted(METHODS)],
        names=["task", "seed", "optimizer"],
    )
    actual_pairs = pd.MultiIndex.from_frame(
        raw[["task", "seed", "optimizer"]].sort_values(["task", "seed", "optimizer"])
    )
    assert actual_pairs.equals(expected_pairs)

    assert metadata["methods"] == ["best_logged", "coms", "bdi"]
    assert metadata["seeds"] == sorted(SEEDS)
    assert metadata["trial_count"] == 8
    assert metadata["recommendations"] == 128
    assert metadata["train_min_percentile"] == 0.0
    assert metadata["train_max_percentile"] == 40.0
    assert metadata["uncertainty"] == "sample_standard_deviation_ddof_1"
    assert metadata["config_fingerprint"] == "37c4cda71c6478d0"


def test_publication_v1_normalization_and_aggregation_are_reproducible() -> None:
    raw = pd.read_csv(RESULTS / "raw_runs.csv")
    summary = pd.read_csv(RESULTS / "task_summary.csv")

    denominator = raw["reference_max_utility"] - raw["reference_min_utility"]
    for utility_column, score_column in (
        ("raw_max_utility", "refnorm_max_score"),
        ("raw_median_utility", "refnorm_median_score"),
        ("raw_mean_utility", "refnorm_mean_score"),
        ("d_best_utility", "refnorm_d_best_score"),
    ):
        expected = (raw[utility_column] - raw["reference_min_utility"]) / denominator
        np.testing.assert_allclose(raw[score_column], expected, rtol=1e-12, atol=1e-12)

    grouped = raw.groupby(["suite", "task", "optimizer"], sort=False)
    rebuilt = grouped.agg(
        mean_score=("refnorm_max_score", "mean"),
        std_score=("refnorm_max_score", lambda values: values.std(ddof=1)),
        mean_raw_max_utility=("raw_max_utility", "mean"),
        std_raw_max_utility=("raw_max_utility", lambda values: values.std(ddof=1)),
        trials=("seed", "count"),
    ).reset_index()
    columns = [
        "suite",
        "task",
        "optimizer",
        "mean_score",
        "std_score",
        "mean_raw_max_utility",
        "std_raw_max_utility",
        "trials",
    ]
    expected = summary[columns].sort_values(columns[:3]).reset_index(drop=True)
    actual = rebuilt[columns].sort_values(columns[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


def test_publication_v1_seed_manifest_and_data_mixture_contract() -> None:
    raw = pd.read_csv(RESULTS / "raw_runs.csv")
    manifest = pd.read_csv(RESULTS / "seed_manifest.csv")

    assert len(manifest) == len(TASKS) * len(SEEDS) == 80
    assert not manifest.duplicated(["task", "seed"]).any()
    assert set(manifest["task"]) == TASKS
    assert set(manifest["seed"]) == SEEDS

    llm = raw[raw["suite"] == "data_mixture"]
    assert len(llm) == len(METHODS) * len(SEEDS) == 24
    assert (llm["train_size"] == 182).all()
    assert (llm["train_min_percentile"] == 0.0).all()
    assert (llm["train_max_percentile"] == 40.0).all()
    assert (llm["raw_max_utility"] < 0.0).all()

    synthetic = raw[raw["suite"] == "synthetic"]
    assert len(synthetic) == 9 * len(METHODS) * len(SEEDS) == 216
    assert (synthetic["train_size"] == 256).all()

    reference_by_task_seed = raw.groupby(["task", "seed"])[
        [
            "reference_min_utility",
            "reference_max_utility",
            "d_best_utility",
            "refnorm_d_best_score",
        ]
    ].nunique()
    assert (reference_by_task_seed == 1).all().all()
