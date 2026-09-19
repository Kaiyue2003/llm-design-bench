from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from llm_design_bench.evaluation.data_manifest import (
    load_data_manifest,
    make_frozen_data_recipes_task_spec,
    prepare_data_manifest,
    save_data_manifest,
    stratified_percentile_mask,
)
from llm_design_bench.evaluation.task_specs import make_data_recipes_task_spec
from llm_design_bench.tasks.data_recipes import (
    DOMAIN_ORDER,
    DataRecipesTask,
    _trusted_checkpoint_loading,
    file_sha256,
)

CHECKPOINT = "opt_algos/data_models/test_run/checkpoints/checkpoint_latest.pt"


def make_fake_root(root: Path) -> Path:
    (root / "opt_algos").mkdir(parents=True)
    (root / "results").mkdir()
    source = (
        "class DataModelBenchmark:\n"
        "    def __init__(self, metric_index=4, device='cpu'):\n"
        f"        feature_names = {dict(enumerate(DOMAIN_ORDER))!r}\n"
        "        raise RuntimeError('oracle must not run during preparation')\n"
    )
    (root / "opt_algos/benchmarks.py").write_text(source, encoding="utf-8")
    checkpoint = root / CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"explicit trusted test checkpoint, not a real pickle")
    for filename in ("config.json", "feature_mask.json"):
        (checkpoint.parent.parent / filename).write_text("{}\n", encoding="utf-8")
    precise = np.float64("0.12345678912345678")
    mixture = [precise, 0.2, 0.2, 0.2, 0.4 - precise]
    rows = []
    for index in range(5):
        for group, loss in (("20M", 100.0 + index), ("1B", 1.0 + index)):
            rows.append(
                {
                    "group": group,
                    "token_probabilities": mixture,
                    "history": pd.DataFrame(
                        {
                            "_step": [100, 19000 + 100 * index],
                            "eval/RedPajamaStackExchange/CrossEntropyLoss": [
                                999.0,
                                loss,
                            ],
                        }
                    ),
                }
            )
    rows.insert(
        2, {"group": "20M", "token_probabilities": mixture, "history": pd.DataFrame()}
    )
    # Duplicate labels must never become row identities; position 2 is skipped.
    pd.DataFrame(rows, index=[7] * len(rows)).to_pickle(
        root / "results/data_mixing_runs.pkl"
    )
    return root


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    return make_fake_root(tmp_path / "data-recipes")


def test_stratification_keeps_low_utility_in_each_scale() -> None:
    utility = np.array([-100, -1, -101, -2, -102, -3, -103, -4, -104, -5])
    scales = np.tile([20, 1000], 5)
    mask = stratified_percentile_mask(utility, scales)
    assert np.flatnonzero(mask).tolist() == [6, 7, 8, 9]
    assert set(scales[mask]) == {20, 1000}


def test_split_is_inclusive_linear_and_keeps_all_cutoff_ties() -> None:
    # At p40 the linear cutoff is exactly -3. All three equal values are kept.
    assert stratified_percentile_mask(
        np.array([-3, -3, -3, -1, 0]), np.ones(5)
    ).tolist() == [True, True, True, False, False]
    # Adjacent values must not leak in through np.isclose.
    assert stratified_percentile_mask(
        np.array([0, 1e-12]), np.ones(2), max_percentile=0
    ).tolist() == [True, False]
    assert stratified_percentile_mask(
        np.array([-2.0]), np.array([1000.0])
    ).tolist() == [True]


@pytest.mark.parametrize("low,high", [(-1, 40), (40, 0), (0, 101), (0, float("nan"))])
def test_split_rejects_invalid_percentiles(low: float, high: float) -> None:
    with pytest.raises(ValueError, match="percentiles"):
        stratified_percentile_mask(
            np.ones(2), np.ones(2), min_percentile=low, max_percentile=high
        )


def test_source_row_ids_are_positions_and_last_checkpoint_is_selected(
    source_root: Path,
) -> None:
    task = DataRecipesTask(source_root)
    assert task.logged_row_ids.tolist() == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    assert task.logged_y.tolist() == [-100, -1, -101, -2, -102, -3, -103, -4, -104, -5]
    assert task.logged_x.training_steps.tolist() == [
        19000,
        19000,
        19100,
        19100,
        19200,
        19200,
        19300,
        19300,
        19400,
        19400,
    ]
    filtered = DataRecipesTask(source_root, logged_model_scale=1000)
    assert filtered.logged_row_ids.tolist() == [1, 4, 6, 8, 10]
    task.logged_row_ids[0] = 99
    assert task.logged_row_ids[0] == 0


def test_prepare_never_constructs_or_queries_oracle(
    source_root: Path, monkeypatch
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("oracle access during preparation")

    monkeypatch.setattr(DataRecipesTask, "_get_benchmark", forbidden)
    monkeypatch.setattr(DataRecipesTask, "_load_upstream_benchmarks", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    bundle = prepare_data_manifest(source_root, [CHECKPOINT])
    assert bundle.visible_row_ids.tolist() == [7, 8, 9, 10]
    assert bundle.utility.tolist() == [-103, -4, -104, -5]
    assert bundle.metadata["domain_order"] == list(DOMAIN_ORDER)
    assert bundle.metadata["utility_transform"] == "negative_loss"
    assert len(bundle.metadata["oracle_sidecars"]) == 2


def test_bundle_roundtrip_preserves_float64_and_separates_hidden_data(
    source_root: Path, tmp_path: Path
) -> None:
    bundle = prepare_data_manifest(source_root, [CHECKPOINT])
    folder = save_data_manifest(bundle, tmp_path / "bundle")
    loaded = load_data_manifest(folder, data_recipes_root=source_root)
    assert loaded.manifest_id == bundle.manifest_id
    for name in ("mixtures", "context", "utility", "reference_utility"):
        assert getattr(loaded, name).dtype == np.float64
        np.testing.assert_array_equal(getattr(loaded, name), getattr(bundle, name))
    precise = bundle.mixtures[0, 0]
    assert precise != np.float64(np.float32(precise))
    with np.load(folder / "visible.npz", allow_pickle=False) as visible:
        assert set(visible.files) == {"row_ids", "mixtures", "context", "utility"}
        assert len(visible["row_ids"]) == 4
    assert not loaded.mixtures.flags.writeable
    with pytest.raises(FileExistsError):
        save_data_manifest(bundle, folder)


def test_fixed_1b_is_exact_main_subset_with_same_reference(source_root: Path) -> None:
    bundle = prepare_data_manifest(source_root, [CHECKPOINT])
    main_spec = make_frozen_data_recipes_task_spec(
        bundle, data_recipes_root=source_root
    )
    fixed_spec = make_frozen_data_recipes_task_spec(
        bundle,
        data_recipes_root=source_root,
        logged_model_scale=1000,
    )
    main, fixed = main_spec.trial_factory(38), fixed_spec.trial_factory(45)
    mask = main.problem.train_context[:, 0] == 1000
    assert main.problem.train_designs.dtype == torch.float64
    assert torch.equal(fixed.problem.train_designs, main.problem.train_designs[mask])
    assert torch.equal(fixed.problem.train_context, main.problem.train_context[mask])
    assert torch.equal(fixed.problem.train_utility, main.problem.train_utility[mask])
    assert fixed.problem.metadata.extra["visible_row_ids"] == [8, 10]
    assert main.problem.target_context.tolist() == [1000.0, 19500.0]
    np.testing.assert_array_equal(main.reference_utility, fixed.reference_utility)
    assert main_spec.normalization_reference_id == fixed_spec.normalization_reference_id
    assert main_spec.trial_factory(0) is main
    encoded_metadata = json.dumps(main.problem.metadata.extra)
    for forbidden in (
        "reference_utility",
        "reference_min",
        "reference_max",
        "oracle_checkpoints",
        "data_file",
        "source_files",
    ):
        assert forbidden not in encoded_metadata
    assert main.problem.train_designs[0, 0].item() == bundle.mixtures[0, 0]


def test_default_task_spec_uses_stratification_and_exact_precision(
    source_root: Path,
) -> None:
    main = make_data_recipes_task_spec(data_recipes_root=source_root).trial_factory(0)
    fixed = make_data_recipes_task_spec(
        data_recipes_root=source_root,
        logged_model_scale=1000,
    ).trial_factory(0)
    assert main.problem.metadata.extra["visible_row_ids"] == [7, 8, 9, 10]
    assert fixed.problem.metadata.extra["visible_row_ids"] == [8, 10]
    assert main.problem.train_utility.tolist() == [-103, -4, -104, -5]
    assert main.problem.train_designs.dtype == torch.float64
    precise = np.float64("0.12345678912345678")
    assert main.problem.train_designs[0, 0].item() == precise


@pytest.mark.parametrize(
    "relative",
    [
        "results/data_mixing_runs.pkl",
        "opt_algos/benchmarks.py",
        CHECKPOINT,
        "opt_algos/data_models/test_run/config.json",
        "opt_algos/data_models/test_run/feature_mask.json",
    ],
)
def test_changed_source_data_checkpoint_or_sidecar_rejected(
    source_root: Path,
    tmp_path: Path,
    relative: str,
) -> None:
    folder = save_data_manifest(
        prepare_data_manifest(source_root, [CHECKPOINT]), tmp_path / "bundle"
    )
    path = source_root / relative
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError, match="mismatch"):
        load_data_manifest(folder, data_recipes_root=source_root)


def test_added_source_rejected(source_root: Path, tmp_path: Path) -> None:
    folder = save_data_manifest(
        prepare_data_manifest(source_root, [CHECKPOINT]), tmp_path / "bundle"
    )
    (source_root / "opt_algos/extra.py").write_text("extra = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source fingerprint"):
        load_data_manifest(folder, data_recipes_root=source_root)


def test_source_hash_is_portable_across_crlf_lf(
    source_root: Path, tmp_path: Path
) -> None:
    folder = save_data_manifest(
        prepare_data_manifest(source_root, [CHECKPOINT]), tmp_path / "bundle"
    )
    source = source_root / "opt_algos/benchmarks.py"
    source.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
    first = load_data_manifest(folder, data_recipes_root=source_root)
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    assert (
        load_data_manifest(folder, data_recipes_root=source_root).manifest_id
        == first.manifest_id
    )


@pytest.mark.parametrize("filename", ["visible.npz", "reference.npz"])
def test_changed_artifact_rejected(
    source_root: Path, tmp_path: Path, filename: str
) -> None:
    folder = save_data_manifest(
        prepare_data_manifest(source_root, [CHECKPOINT]), tmp_path / "bundle"
    )
    (folder / filename).write_bytes(b"bad archive")
    with pytest.raises(ValueError, match="artifact file hash"):
        load_data_manifest(folder, data_recipes_root=source_root)


def test_manifest_and_in_memory_content_tampering_rejected(
    source_root: Path, tmp_path: Path
) -> None:
    bundle = prepare_data_manifest(source_root, [CHECKPOINT])
    with pytest.raises(ValueError, match="content hash"):
        save_data_manifest(
            replace(bundle, utility=bundle.utility - 1), tmp_path / "bad"
        )
    folder = save_data_manifest(bundle, tmp_path / "bundle")
    manifest_path = folder / "manifest.json"
    envelope = json.loads(manifest_path.read_text())
    envelope["metadata"]["upstream_revision"] = "tampered"
    manifest_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        load_data_manifest(folder, data_recipes_root=source_root)


def test_low_precision_artifact_rejected_even_if_file_hash_updated(
    source_root: Path, tmp_path: Path
) -> None:
    folder = save_data_manifest(
        prepare_data_manifest(source_root, [CHECKPOINT]), tmp_path / "bundle"
    )
    visible = folder / "visible.npz"
    with np.load(visible, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["mixtures"] = arrays["mixtures"].astype(np.float32)
    np.savez_compressed(visible, **arrays)
    manifest_path = folder / "manifest.json"
    envelope = json.loads(manifest_path.read_text())
    envelope["artifacts"]["visible.npz"] = file_sha256(visible)
    manifest_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="precision"):
        load_data_manifest(folder, data_recipes_root=source_root)


def test_unknown_domain_order_rejected_without_import(source_root: Path) -> None:
    source = source_root / "opt_algos/benchmarks.py"
    source.write_text(
        source.read_text().replace("RedPajamaWikipedia", "incorrect"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="domain order"):
        prepare_data_manifest(source_root, [CHECKPOINT])


def test_checkpoint_declarations_are_explicit_and_root_bounded(
    source_root: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="declare"):
        prepare_data_manifest(source_root, [])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_data_manifest(source_root, [CHECKPOINT, CHECKPOINT])
    outside = tmp_path / "external.pt"
    outside.write_bytes(b"test")
    with pytest.raises(ValueError, match="within"):
        prepare_data_manifest(source_root, [outside])
    with pytest.raises(FileNotFoundError):
        prepare_data_manifest(source_root, ["does_not_exist.pt"])
    with pytest.raises(ValueError, match="metric_index=4"):
        prepare_data_manifest(source_root, [CHECKPOINT], metric_index=0)


def test_checkpoint_loader_enforces_actual_paths_and_hashes(
    source_root: Path, monkeypatch
) -> None:
    checkpoint = source_root / CHECKPOINT
    expected = {CHECKPOINT: file_sha256(checkpoint)}
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return {"loaded": True}

    monkeypatch.setattr(torch, "load", load)
    with _trusted_checkpoint_loading(source_root, expected):
        assert torch.load(checkpoint) == {"loaded": True}
    assert calls[0][1]["weights_only"] is False
    assert torch.load is load
    other = checkpoint.with_name("other.pt")
    other.write_bytes(b"not-declared")
    with (
        pytest.raises(ValueError, match="undeclared"),
        _trusted_checkpoint_loading(source_root, expected),
    ):
        torch.load(other)
    with (
        pytest.raises(ValueError, match="every declared"),
        _trusted_checkpoint_loading(source_root, expected),
    ):
        pass
    checkpoint.write_bytes(b"changed")
    with (
        pytest.raises(ValueError, match="hash mismatch"),
        _trusted_checkpoint_loading(source_root, expected),
    ):
        torch.load(checkpoint)
    assert torch.load is load
    assert len(calls) == 1


def test_cached_oracle_rejects_later_file_drift(source_root: Path, monkeypatch) -> None:
    bundle = prepare_data_manifest(source_root, [CHECKPOINT])
    task = (
        make_frozen_data_recipes_task_spec(bundle, data_recipes_root=source_root)
        .trial_factory(0)
        .evaluator_task
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})

    def construct(**kwargs):
        torch.load(source_root / CHECKPOINT)
        return SimpleNamespace(_raw_func_with_model_scale=lambda *args, **kwargs: 1.5)

    monkeypatch.setattr(
        task,
        "_load_upstream_benchmarks",
        lambda: SimpleNamespace(DataModelBenchmark=construct),
    )
    candidates = task.at_target_fidelity(np.ones((1, 5)) / 5)
    assert task.predict(candidates).tolist() == [-1.5]
    (source_root / CHECKPOINT).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        task.predict(candidates)


def test_upstream_module_imports_are_isolated_between_roots(
    tmp_path: Path, monkeypatch
) -> None:
    previous_module = ModuleType("data_model")
    previous_module.MARKER = 99
    monkeypatch.setitem(sys.modules, "data_model", previous_module)
    tasks = []
    for value in (1, 2):
        root = make_fake_root(tmp_path / f"root{value}")
        (root / "opt_algos/data_model.py").write_text(
            f"MARKER = {value}\n", encoding="utf-8"
        )
        source = (
            "import data_model\nfrom pathlib import Path\nimport torch\n"
            "class DataModelBenchmark:\n"
            "    def __init__(self, metric_index=4, device='cpu'):\n"
            f"        feature_names = {dict(enumerate(DOMAIN_ORDER))!r}\n"
            "        torch.load(Path(__file__).parent / 'data_models/test_run/checkpoints/checkpoint_latest.pt')\n"
            "    def _raw_func_with_model_scale(self, *args, **kwargs):\n"
            "        return data_model.MARKER\n"
        )
        (root / "opt_algos/benchmarks.py").write_text(source, encoding="utf-8")
        torch.save({}, root / CHECKPOINT)
        bundle = prepare_data_manifest(root, [CHECKPOINT])
        task = (
            make_frozen_data_recipes_task_spec(bundle, data_recipes_root=root)
            .trial_factory(0)
            .evaluator_task
        )
        tasks.append(task)
    for value, task in enumerate(tasks, start=1):
        assert task.predict(task.at_target_fidelity(np.ones((1, 5)) / 5)).tolist() == [
            -value
        ]
        assert sys.modules["data_model"] is previous_module


def test_explicit_invalid_root_does_not_fall_back(
    source_root: Path, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATA_RECIPES_ROOT", str(source_root))
    with pytest.raises(FileNotFoundError, match="explicit"):
        DataRecipesTask(tmp_path / "missing")
