# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.11.15-slim-bookworm
FROM ${PYTHON_IMAGE} AS runtime

ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="llm-design-bench" \
      org.opencontainers.image.description="Reproducible PyTorch offline black-box optimization benchmarks" \
      org.opencontainers.image.source="https://github.com/Kaiyue2003/llm-design-bench" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LLM_DESIGN_BENCH_VCS_REF="${VCS_REF}" \
    RESULTS_ROOT=/results

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/
WORKDIR /app
COPY . /app
RUN uv sync --frozen --no-dev \
    && chmod +x /app/docker/*.sh

VOLUME ["/results"]
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["--help"]
