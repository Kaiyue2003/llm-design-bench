"""Verify saved artifacts and optionally compare two independent executions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_design_bench.evaluation.reproducibility import verify_run_artifacts


KEYS = ["suite", "task", "seed", "optimizer"]


def verify(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(directory / "raw_runs.csv")
    expected_tasks = set(metadata["synthetic_functions"])
    if metadata["include_data_mixture"]:
        expected_tasks.add("data_recipes_stack_exchange")
    expected = {(task, seed, method) for task in expected_tasks for seed in metadata["seeds"] for method in metadata["methods"]}
    actual = set(rows[["task", "seed", "optimizer"]].itertuples(index=False, name=None))
    if actual != expected or rows.duplicated(KEYS).any():
        raise ValueError("result rows are incomplete, duplicated, or unexpected")
    for encoded in rows["artifacts_json"]:
        verify_run_artifacts(directory, encoded)
    return metadata, rows.sort_values(KEYS).reset_index(drop=True)


def compare(first, second, *, atol=0.0, rtol=0.0):
    left_meta, left = verify(first)
    right_meta, right = verify(second)
    if left_meta["config_fingerprint"] != right_meta["config_fingerprint"]:
        raise ValueError("replay configurations/code/dependencies differ")
    if not left[KEYS].equals(right[KEYS]):
        raise ValueError("replay keys differ")
    for (_, lrow), (_, rrow) in zip(left.iterrows(), right.iterrows()):
        la, ra = json.loads(lrow["artifacts_json"]), json.loads(rrow["artifacts_json"])
        if la["dataset"]["arrays_sha256"] != ra["dataset"]["arrays_sha256"]:
            raise ValueError("replay input arrays differ")
        with np.load(Path(first) / la["candidates"]["path"], allow_pickle=False) as lvalues, np.load(Path(second) / ra["candidates"]["path"], allow_pickle=False) as rvalues:
            for name in lvalues.files:
                np.testing.assert_allclose(lvalues[name], rvalues[name], rtol=rtol, atol=atol, err_msg=f"{lrow['optimizer']} {name}")
    return {"verified_rows": len(left), "replay_rows": len(right), "atol": atol, "rtol": rtol, "status": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    if args.compare:
        report = compare(args.results, args.compare, atol=args.atol, rtol=args.rtol)
    else:
        _, rows = verify(args.results)
        report = {"verified_rows": len(rows), "status": "passed"}
    (args.results / "replay_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
