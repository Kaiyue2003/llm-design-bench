"""Packaging and identity checks; actual Linux containers run in separate CI."""

import os
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
    assert "${RESULTS_DIR:-./results/docker}:/results" in compose
    assert "${ASSETS_DIR:-./assets}:/assets:ro" in compose
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
    assert 'run --rm benchmark "$@"' in shell
    assert "set -- --help" in shell
    assert "--project-directory $projectDirectory" in powershell
    assert "run --rm benchmark @cliArguments" in powershell
    assert '@("--help")' in powershell
    assert "publication" not in shell + powershell


def test_ci_checks_installed_entrypoints_and_actual_container_runtime() -> None:
    workflow = (ROOT / ".github/workflows/container.yml").read_text(encoding="utf-8")
    ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "docker/login-action@v3" in workflow
    assert "password: ${{ secrets.GITHUB_TOKEN }}" in workflow
    assert "docker/build-push-action@v6" in workflow
    assert "load: ${{ github.event_name == 'pull_request' }}" in workflow
    assert "Check the unified default CLI without training" in workflow
    assert "/app/scripts/container_smoke.py --results-root /results" in workflow
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
    for removed in (
        "llm-design-bench-publication",
        "llm-design-bench-offline",
        "llm-design-bench-synthetic",
    ):
        assert removed not in ci + workflow


def test_ci_smoke_results_are_owned_by_the_artifact_uploader() -> None:
    script = _ci_smoke_script()

    # Docker otherwise creates a missing bind source as root before switching UID.
    create_output = 'mkdir -p "$RUNNER_TEMP/container-smoke"'
    assert script.index(create_output) < script.index("docker run --rm")
    assert '--user "$(id -u):$(id -g)"' in script
    assert "--env HOME=/tmp" in script
    assert '--volume "$RUNNER_TEMP/container-smoke:/results"' in script
    assert "--entrypoint python" in script
    assert "/app/scripts/container_smoke.py --results-root /results" in script


@pytest.mark.parametrize("provide_username", [False, True])
def test_ci_smoke_identity_supports_torch_without_a_passwd_entry(
    tmp_path, provide_username
) -> None:
    # Read the actual Docker environment options, not a duplicate test config.
    tokens = shlex.split(_ci_smoke_script().replace("\\\n", ""), comments=True)
    container_env = dict(
        tokens[index + 1].split("=", 1)
        for index, token in enumerate(tokens)
        if token == "--env"
    )
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
