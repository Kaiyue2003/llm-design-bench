from pathlib import Path

import pytest

from llm_design_bench.evaluation.reproducibility import atomic_replace


@pytest.mark.parametrize("release_lock", [True, False])
def test_checkpoint_replacement_handles_reader_locks(tmp_path, monkeypatch, release_lock):
    source, destination = tmp_path / "checkpoint.tmp", tmp_path / "checkpoint.csv"
    source.write_text("new")
    destination.write_text("old")
    original_replace = Path.replace
    attempts = []

    def locked_replace(path, target):
        attempts.append(1)
        if not release_lock or len(attempts) == 1:
            assert destination.read_text() == "old"
            raise PermissionError("Windows reader holds destination")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr("llm_design_bench.evaluation.reproducibility.time.sleep", lambda _: None)
    if release_lock:
        atomic_replace(source, destination)
        assert len(attempts) == 2
        assert destination.read_text() == "new"
        assert not source.exists()
    else:
        with pytest.raises(PermissionError):
            atomic_replace(source, destination)
        assert len(attempts) == 21
        assert source.read_text() == "new"
        assert destination.read_text() == "old"
