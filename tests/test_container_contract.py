"""Static packaging checks; executing a Linux container is a separate CI job."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
