"""Exercise the Colab batch entry points without network, Drive, or training."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/colab_batch_cell.py"
NOTEBOOK_PATH = ROOT / "notebooks/LLMDM_Colab_Batch.ipynb"
NOTEBOOK = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
CELLS = {cell["id"]: cell for cell in NOTEBOOK["cells"]}
PLAN = json.loads(
    (ROOT / "experiments/llmdm_forward_v1/plan.json").read_text(encoding="utf-8")
)


def _source(cell_id):
    return "".join(CELLS[cell_id]["source"])


def _execute(cell_id, namespace):
    exec(compile(_source(cell_id), f"{NOTEBOOK_PATH}:{cell_id}", "exec"), namespace)  # noqa: S102


@pytest.fixture
def loader():
    spec = importlib.util.spec_from_file_location("batch_bootstrap_fixture", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_is_clean_and_every_code_cell_compiles():
    assert len(CELLS) == len(NOTEBOOK["cells"])
    assert NOTEBOOK["nbformat"] == 4
    for cell in CELLS.values():
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cell["id"])
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), str(NOTEBOOK_PATH), "exec")
    assert ast.dump(ast.parse(_source("batch-load"))) == ast.dump(
        ast.parse(EXAMPLE.read_text(encoding="utf-8"))
    )


def test_helper_revision_and_hashes_are_pinned_to_current_launcher(loader):
    assert re.fullmatch(r"[0-9a-f]{40}", loader.BATCH_REVISION)
    assert loader.BATCH_REVISION == "5f9bd5dc208d53f6f22684b76645168977ffd1b0"
    for name in ("BATCH_SHA256", "SUPPORT_SHA256"):
        assert re.fullmatch(r"[0-9a-f]{64}", getattr(loader, name))
    for name, path in (
        ("BATCH_SHA256", ROOT / "scripts/colab_batch.py"),
        ("SUPPORT_SHA256", ROOT / "scripts/colab_support.py"),
    ):
        assert (
            getattr(loader, name)
            == hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        )


def test_importing_bootstrap_never_downloads_or_trains(monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        pytest.fail("importing the bootstrap must not download or execute a job")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    spec = importlib.util.spec_from_file_location("only_definitions", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.load_batch_runner)
    assert not hasattr(module, "batch")
    calls = [
        node
        for node in ast.walk(ast.parse(EXAMPLE.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
    ]
    assert not any(
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"run", "fit", "predict"}
        for node in calls
    )


@pytest.fixture
def network(loader, monkeypatch):
    source = b"# harmless, commit-pinned fixture\n"
    monkeypatch.setattr(loader, "BATCH_SHA256", hashlib.sha256(source).hexdigest())
    calls = []

    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(source)

    monkeypatch.setattr(loader.urllib.request, "urlopen", fetch)
    return source, calls


def test_fetch_checks_hash_and_reuses_verified_cache(loader, network, tmp_path):
    source, calls = network
    cache = tmp_path / "cache"
    path = loader._fetch_launcher(cache)
    assert path == cache / loader.BATCH_REVISION / "colab_batch.py"
    assert path.read_bytes() == source
    assert loader._fetch_launcher(cache) == path
    assert len(calls) == 1
    assert calls[0] == (
        (
            "https://raw.githubusercontent.com/Kaiyue2003/llm-design-bench/"
            f"{loader.BATCH_REVISION}/scripts/colab_batch.py"
        ),
        30,
    )


def test_download_hash_failure_writes_nothing(loader, network, monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "BATCH_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="downloaded.*hash mismatch"):
        loader._fetch_launcher(tmp_path / "cache")
    assert not list(tmp_path.iterdir())


def test_oversized_download_writes_nothing(loader, network, monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "MAX_SCRIPT_BYTES", 8)
    with pytest.raises(ValueError, match="size limit"):
        loader._fetch_launcher(tmp_path / "cache")
    assert not list(tmp_path.iterdir())


def test_corrupt_cache_is_not_overwritten_or_redownloaded(loader, network, tmp_path):
    _, calls = network
    path = loader._fetch_launcher(tmp_path / "cache")
    path.write_bytes(b"corrupt existing cache")
    with pytest.raises(ValueError, match="cached.*hash mismatch"):
        loader._fetch_launcher(tmp_path / "cache")
    assert path.read_bytes() == b"corrupt existing cache"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "name,value", [("BATCH_REVISION", "main"), ("BATCH_SHA256", "invalid")]
)
def test_unpinned_launcher_is_rejected_before_io(
    loader, network, monkeypatch, tmp_path, name, value
):
    monkeypatch.setattr(loader, name, value)
    with pytest.raises(ValueError, match="pinned"):
        loader._fetch_launcher(tmp_path / "cache")
    assert not network[1] and not list(tmp_path.iterdir())


def test_wrong_original_helper_hash_stops_before_download(loader, network, tmp_path):
    repo = tmp_path / "repo"
    helper = repo / "scripts/colab_support.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"wrong helper")
    with pytest.raises(ValueError, match="single-job helper changed"):
        loader.load_batch_runner(repo, "assets", "upstream", "state", "backups")
    assert not network[1]


def test_load_constructs_only_checked_helper_without_training(
    loader, monkeypatch, tmp_path
):
    repo = tmp_path / "repo"
    helper = repo / "scripts/colab_support.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"# matching support fixture\n")
    monkeypatch.setattr(
        loader, "SUPPORT_SHA256", hashlib.sha256(helper.read_bytes()).hexdigest()
    )
    monkeypatch.setitem(
        sys.modules, "colab_support", SimpleNamespace(__file__=str(helper))
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    source = (
        b"class BatchRunner:\n"
        b"    def __init__(self, **kwargs): self.kwargs = kwargs\n"
        b"    def run(self): raise AssertionError('must not train')\n"
    )
    monkeypatch.setattr(loader, "BATCH_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(
        loader.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(source)
    )
    module_name = "llmdm_colab_batch_" + loader.BATCH_REVISION
    monkeypatch.setitem(sys.modules, module_name, None)
    result = loader.load_batch_runner(
        repo, "assets", "upstream", "state", "backups", neural_device="cpu"
    )
    assert result.kwargs == {
        "assets": "assets",
        "data_recipes_root": "upstream",
        "state": "state",
        "backups": "backups",
        "neural_device": "cpu",
        "allow_environment_change": False,
    }
    assert not (tmp_path / "state").exists()


class FakeFrame:
    def __init__(self, rows):
        self.rows = rows

    def __contains__(self, name):
        return bool(self.rows) and name in self.rows[0]

    def __getitem__(self, columns):
        return FakeFrame([{key: row[key] for key in columns} for row in self.rows])


@pytest.fixture
def notebook_namespace(tmp_path):
    calls = {"load": [], "preview": [], "run": [], "report": [], "display": []}
    fake = SimpleNamespace(plan=PLAN)

    def preview(**kwargs):
        calls["preview"].append(kwargs)
        return [
            {
                "setting": "multi_scale",
                "run_id": "coms",
                "seed": 0,
                "device": "cuda",
                "status": "pending",
            }
        ]

    def run(**kwargs):
        calls["run"].append(kwargs)
        return [{"status": "completed"}]

    def report(**kwargs):
        calls["report"].append(kwargs)
        return [{"status": "verified"}]

    def load(*args, **kwargs):
        calls["load"].append((args, kwargs))
        return fake

    fake.preview, fake.run, fake.pilot_report = preview, run, report
    namespace = {
        "load_batch_runner": load,
        "batch": fake,
        "pd": SimpleNamespace(DataFrame=FakeFrame),
        "display": calls["display"].append,
        **{
            name: tmp_path / name.lower()
            for name in ("REPO", "ASSETS", "UPSTREAM", "STATE", "BACKUPS")
        },
    }
    return namespace, calls


def test_default_config_and_run_only_preview(notebook_namespace):
    namespace, calls = notebook_namespace
    _execute("batch-config", namespace)
    assert namespace["RUN_BATCH"] is False
    assert namespace["PHASE"] == "pilot" and namespace["METHODS"] is None
    _execute("batch-run", namespace)
    assert calls["preview"] == [
        {"methods": None, "settings": ("multi_scale",), "phase": "pilot"}
    ]
    assert not calls["run"]


def test_pilot_dispatches_one_queue_without_granting_formal_review(notebook_namespace):
    namespace, calls = notebook_namespace
    _execute("batch-config", namespace)
    namespace.update(RUN_BATCH=True, METHODS=["coms", "bdi"])
    _execute("batch-run", namespace)
    assert calls["run"] == [
        {
            "methods": ["coms", "bdi"],
            "settings": ("multi_scale",),
            "phase": "pilot",
            "reviewed_pilots": [],
        }
    ]


def test_formal_requires_explicit_confirmation_before_any_dispatch(notebook_namespace):
    namespace, calls = notebook_namespace
    _execute("batch-config", namespace)
    namespace.update(RUN_BATCH=True, PHASE="formal")
    with pytest.raises(RuntimeError, match="pilot"):
        _execute("batch-run", namespace)
    assert not calls["run"]


@pytest.mark.parametrize("methods", [None, ["coms", "bdi"]])
def test_formal_review_pairs_exactly_match_selection(notebook_namespace, methods):
    namespace, calls = notebook_namespace
    _execute("batch-config", namespace)
    settings = ("multi_scale", "fixed_1b")
    namespace.update(
        RUN_BATCH=True,
        PHASE="formal",
        CONFIRM_BATCH_PILOTS_REVIEWED=True,
        METHODS=methods,
        SETTINGS=settings,
    )
    _execute("batch-run", namespace)
    selected = (
        methods
        if methods is not None
        else [entry["run_id"] for entry in PLAN["methods"]]
    )
    assert len(calls["run"]) == 1
    assert calls["run"][0] == {
        "methods": methods,
        "settings": settings,
        "phase": "formal",
        "reviewed_pilots": [
            (setting, method) for setting in settings for method in selected
        ],
    }


def test_report_reads_existing_results_without_dispatch(notebook_namespace):
    namespace, calls = notebook_namespace
    _execute("batch-config", namespace)
    _execute("batch-report", namespace)
    assert calls["report"] == [{"methods": None, "settings": ("multi_scale",)}]
    assert not calls["run"]
    assert not namespace["STATE"].exists()
