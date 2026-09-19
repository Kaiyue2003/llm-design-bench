"""Frozen checkpoint checks must bind the file actually passed to torch.load."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch

from llm_design_bench.tasks.data_recipes import (
    _trusted_checkpoint_loading,
    file_sha256,
)


@pytest.fixture
def checkpoint_tree(tmp_path):
    trusted = tmp_path / "trusted"
    other = tmp_path / "other-cwd"
    relative = Path("checkpoints") / "inert.pt"
    for root, marker in ((trusted, 1), (other, 9)):
        path = root / relative
        path.parent.mkdir(parents=True)
        # Inert tensor dictionaries only: no executable pickle payloads.
        torch.save({"marker": torch.tensor(marker)}, path)
    expected = {relative.as_posix(): file_sha256(trusted / relative)}
    return trusted, other, relative, expected


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
@pytest.mark.parametrize("mmap", [False, True])
def test_loads_verified_file_from_another_cwd(
    checkpoint_tree, monkeypatch, keyword, absolute, mmap
):
    trusted, other, relative, expected = checkpoint_tree
    monkeypatch.chdir(other)
    original_load = torch.load
    path = trusted / relative if absolute else relative.as_posix()
    assert expected[relative.as_posix()] != file_sha256(other / relative)

    with _trusted_checkpoint_loading(trusted, expected):
        options = {"map_location": "cpu", "mmap": mmap, "weights_only": True}
        value = (
            torch.load(f=path, **options) if keyword else torch.load(path, **options)
        )

    assert value["marker"].item() == 1
    assert value["marker"].device.type == "cpu"
    assert torch.load is original_load


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize("weights_only", ["omitted", True, False, None])
def test_forwards_all_other_load_arguments_unchanged(
    checkpoint_tree, monkeypatch, keyword, weights_only
):
    trusted, other, relative, expected = checkpoint_tree
    monkeypatch.chdir(other)
    calls = []
    result = object()
    map_location = object()
    pickle_module = object()

    def recording_load(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(torch, "load", recording_load)
    options = {"mmap": True, "encoding": "latin1"}
    if weights_only != "omitted":
        options["weights_only"] = weights_only
    original_options = options.copy()
    with _trusted_checkpoint_loading(trusted, expected):
        if keyword:
            actual = torch.load(
                f=relative,
                map_location=map_location,
                pickle_module=pickle_module,
                **options,
            )
        else:
            actual = torch.load(relative, map_location, pickle_module, **options)

    assert actual is result
    assert options == original_options
    assert torch.load is recording_load
    assert len(calls) == 1
    args, forwarded = calls[0]
    checked = (trusted / relative).resolve()
    if keyword:
        assert args == ()
        assert forwarded.pop("f") == checked
        assert forwarded.pop("map_location") is map_location
        assert forwarded.pop("pickle_module") is pickle_module
    else:
        assert args == (checked, map_location, pickle_module)
    assert forwarded == {
        "mmap": True,
        "encoding": "latin1",
        "weights_only": False if weights_only == "omitted" else weights_only,
    }


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize("file_object", [False, True], ids=["path", "file-object"])
def test_nonfrozen_loader_does_not_rewrite_its_input(monkeypatch, keyword, file_object):
    supplied = io.BytesIO(b"not deserialized") if file_object else "relative.pt"
    calls = []

    def recording_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "unchanged"

    monkeypatch.setattr(torch, "load", recording_load)
    with _trusted_checkpoint_loading():
        actual = torch.load(f=supplied) if keyword else torch.load(supplied)
    assert actual == "unchanged"
    assert torch.load is recording_load
    args, options = calls[0]
    if keyword:
        assert args == ()
        assert options.pop("f") is supplied
    else:
        assert args[0] is supplied
    assert options == {"weights_only": False}


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
@pytest.mark.parametrize(
    "failure,match",
    [
        ("hash", "hash mismatch"),
        ("undeclared", "undeclared oracle checkpoint"),
        ("outside", "must stay within"),
        ("relative-escape", "must stay within"),
        ("file-object", "only declared checkpoint paths"),
    ],
)
def test_invalid_file_is_rejected_before_the_underlying_load(
    checkpoint_tree, monkeypatch, keyword, failure, match
):
    trusted, other, relative, expected = checkpoint_tree
    monkeypatch.chdir(other)
    supplied = relative
    if failure == "hash":
        (trusted / relative).write_bytes(b"modified after freezing")
    elif failure == "undeclared":
        supplied = relative.with_name("other.pt")
        (trusted / supplied).write_bytes(b"undeclared, never deserialized")
    elif failure == "outside":
        supplied = other / relative
    elif failure == "relative-escape":
        supplied = Path("..") / other.name / relative
    elif failure == "file-object":
        supplied = io.BytesIO(b"not deserialized")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid input must not reach the checkpoint deserializer")

    monkeypatch.setattr(torch, "load", forbidden)
    with (
        pytest.raises(ValueError, match=match),
        _trusted_checkpoint_loading(trusted, expected),
    ):
        if keyword:
            torch.load(f=supplied)
        else:
            torch.load(supplied)
    assert torch.load is forbidden


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_deserializer_exceptions_restore_original_torch_load(
    checkpoint_tree, monkeypatch, error_type
):
    trusted, _, relative, expected = checkpoint_tree
    error = error_type("fixed loader failure")

    def failing_load(*args, **kwargs):
        raise error

    monkeypatch.setattr(torch, "load", failing_load)
    with (
        pytest.raises(error_type) as caught,
        _trusted_checkpoint_loading(trusted, expected),
    ):
        torch.load(relative)
    assert caught.value is error
    assert torch.load is failing_load


def test_duplicate_file_arguments_keep_native_argument_error(checkpoint_tree):
    trusted, _, relative, expected = checkpoint_tree
    original_load = torch.load
    with (
        pytest.raises(TypeError, match="multiple values"),
        _trusted_checkpoint_loading(trusted, expected),
    ):
        torch.load(relative, f=trusted / relative, weights_only=True)
    assert torch.load is original_load
