"""Packaging and identity checks; actual Linux containers run in separate CI."""

import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _ci_smoke_script() -> str:
    workflow = (ROOT / ".github/workflows/container.yml").read_text(encoding="utf-8")
    smoke_step = workflow.split(
        "      - name: Exercise all 27 methods on invented data and verify resume\n", 1
    )[1].split("      - name:", 1)[0]
    return smoke_step.split("        run: |\n", 1)[1]


def _compose_environment() -> dict[str, str]:
    """Read the actual flat shared environment, without requiring a YAML package."""
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    shared = compose.split("\nx-benchmark:", 1)[1].split("\nservices:", 1)[0]
    environment = shared.split("  environment:\n", 1)[1]
    values = {}
    for name, value in re.findall(
        r"^    ([A-Z][A-Z0-9_]*):[ \t]*(.*)$", environment, re.MULTILINE
    ):
        tokens = shlex.split(value, comments=True)
        assert len(tokens) == 1, (
            f"Expected one literal Compose environment value: {name}"
        )
        values[name] = tokens[0]
    return values


def test_container_is_pinned_and_uses_only_the_explicit_formal_cli() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    entrypoint = (ROOT / "docker/entrypoint.sh").read_text(encoding="utf-8")
    assert "python:3.11.15-slim-bookworm" in dockerfile
    assert "ghcr.io/astral-sh/uv:0.11.8" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert 'CMD ["--help"]' in dockerfile
    assert 'exec llm-design-bench "$@"' in entrypoint
    assert "mkdir" not in entrypoint
    assert "write_environment_manifest" not in entrypoint
    assert "reproduce_publication" not in dockerfile
    assert "RUN_RESULTS_DIR" not in dockerfile
    for removed in ("reproduce_publication.sh", "reproduce_all_methods_smoke.sh"):
        assert not (ROOT / "docker" / removed).exists()


def test_compose_persists_results_and_keeps_input_assets_read_only() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "  benchmark:" in compose
    assert 'command: ["--help"]' in compose
    assert "${LLMDM_CONTAINER_USER:?" in compose
    assert "user:" in compose
    assert "${RESULTS_DIR:-./results/docker}" in compose
    assert "${ASSETS_DIR:-./assets}" in compose
    assert "target: /results" in compose
    assert "target: /assets" in compose
    assert "read_only: true" in compose
    assert compose.count("create_host_path: false") == 2
    assert compose.count("<<: *benchmark") == 2
    environment = _compose_environment()
    for name, value in {"HOME": "/tmp", "USER": "runner", "LOGNAME": "runner"}.items():
        assert environment[name] == value
    assert "  smoke:" in compose
    assert 'entrypoint: ["python", "/app/scripts/container_smoke.py"]' in compose
    assert "publication:" not in compose
    assert "all-methods-smoke:" not in compose
    assert "gpus:" not in compose
    assert "driver: nvidia" not in compose


def test_wrappers_forward_arguments_to_benchmark_without_implicit_training() -> None:
    shell = (ROOT / "scripts/reproduce_docker.sh").read_text(encoding="utf-8")
    powershell = (ROOT / "scripts/reproduce_docker.ps1").read_text(encoding="utf-8")
    assert '--project-directory "$project_dir"' in shell
    assert "set -- --help" in shell
    assert '"--project-directory", $projectDirectory' in powershell
    assert '@("--help")' in powershell
    for script in (shell, powershell):
        assert "--smoke" in script
        assert "--build" in script
        assert "benchmark" in script
    assert "publication" not in shell + powershell


def test_ci_checks_installed_entrypoints_and_actual_container_runtime() -> None:
    workflow = (ROOT / ".github/workflows/container.yml").read_text(encoding="utf-8")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "docker/login-action@v3" in workflow
    assert "password: ${{ secrets.GITHUB_TOKEN }}" in workflow
    assert "docker/build-push-action@v6" in workflow
    assert "load: ${{ github.event_name == 'pull_request' }}" in workflow
    assert "Check the unified default CLI without training" in workflow
    assert "sh scripts/reproduce_docker.sh --smoke" in workflow
    assert "docker run --rm" in workflow
    for entrypoint in (
        "llm-design-bench",
        "llm-design-bench-llmdm",
        "llm-design-bench-report",
    ):
        assert f"{entrypoint} --help" in ci
    assert "llm-design-bench methods" in ci
    assert "pip install dist/*.whl" in ci
    assert 'cd "$RUNNER_TEMP"' in ci
    assert "-I -m llm_design_bench methods" in ci
    assert "runs-on: windows-latest" in ci
    assert "python -m pytest --noconftest tests/test_docker_wrappers.py -q" in ci
    for removed in (
        "llm-design-bench-publication",
        "llm-design-bench-offline",
        "llm-design-bench-synthetic",
    ):
        assert removed not in ci + workflow


def test_ci_smoke_results_are_owned_by_the_artifact_uploader() -> None:
    script = _ci_smoke_script()
    workflow = (ROOT / ".github/workflows/container.yml").read_text(encoding="utf-8")
    # Do not create a separate Docker path that bypasses local Compose safeguards.
    assert "sh scripts/reproduce_docker.sh --smoke" in script
    assert "docker run" not in script
    assert "LLM_DESIGN_BENCH_IMAGE:" in workflow
    assert "RESULTS_DIR: ${{ runner.temp }}/container-smoke" in workflow
    assert "ASSETS_DIR: ${{ runner.temp }}/container-assets" in workflow
    assert (
        'LLMDM_CONTAINER_USER="$(id -u):$(id -g)" docker compose config --quiet'
        in workflow
    )
    assert "path: ${{ runner.temp }}/container-smoke" in workflow
    assert "path.stat().st_uid == os.getuid()" in workflow
    assert 'with path.open("rb") as stream:' in workflow
    assert "while stream.read(1024 * 1024):" in workflow


@pytest.mark.parametrize("provide_username", [False, True])
def test_compose_identity_supports_torch_without_a_passwd_entry(
    tmp_path, provide_username
) -> None:
    # Both local and CI runs inherit this actual shared Compose environment.
    container_env = _compose_environment()
    env = os.environ.copy()
    for name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        env.pop(name, None)
    env.update(container_env)
    if not provide_username:
        for name in ("LOGNAME", "USER", "LNAME", "USERNAME"):
            env.pop(name, None)
    # Use a writable host equivalent of /tmp, including on Windows.
    for name in ("HOME", "TMPDIR", "TMP", "TEMP"):
        env[name] = str(tmp_path)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "inductor")
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    probe = textwrap.dedent(
        """
        import getpass
        import os
        import sys
        import types
        from unittest.mock import patch

        def missing_user(uid):
            raise KeyError(f"getpwuid(): uid not found: {uid}")

        passwd = types.ModuleType("pwd")
        passwd.getpwuid = missing_user
        with patch.dict(sys.modules, {"pwd": passwd}), patch.object(
            os, "getuid", lambda: 1001, create=True
        ):
            assert getpass.getuser() == "runner"
            import torch

            parameter = torch.tensor([1.0], requires_grad=True)
            optimizer = torch.optim.Adam([parameter], lr=0.01)
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()
            assert 0.0 < parameter.item() < 1.0
            print("username resolved; Adam step passed")
        """
    )
    # An interrupted torch import must not contaminate the pytest process.
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if provide_username:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "username resolved; Adam step passed" in completed.stdout
    else:
        assert completed.returncode != 0
        assert "getpwuid(): uid not found: 1001" in completed.stderr
