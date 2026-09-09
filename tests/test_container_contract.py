from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_container_reproduction_files_are_pinned_and_persistent() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    reproduce = (ROOT / "docker" / "reproduce_publication.sh").read_text(
        encoding="utf-8"
    )
    workflow = (
        ROOT / ".github" / "workflows" / "container.yml"
    ).read_text(encoding="utf-8")

    assert "python:3.11.15-slim-bookworm" in dockerfile
    assert "ghcr.io/astral-sh/uv:0.11.8" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "results/docker" in compose
    assert ":/results" in compose
    assert "37269969a0957448d51622e0c083977bc5d260e8" in reproduce
    assert "--seed 38 --seed 39 --seed 40 --seed 41" in reproduce
    assert "docker/login-action@v3" in workflow
    assert "password: ${{ secrets.GITHUB_TOKEN }}" in workflow
    assert "docker/build-push-action@v6" in workflow
