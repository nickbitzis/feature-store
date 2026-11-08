# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv and app code.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 featstore
WORKDIR /app
COPY --from=build --chown=featstore:featstore /app /app
# The offline store, online snapshot, model, and registry are written under /app/data at runtime.
RUN mkdir -p /app/data && chown featstore:featstore /app/data
USER featstore
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8858 FEATSTORE_DATA_DIR=/app/data \
    FEATSTORE_CARDS=50000 FEATSTORE_DAYS=60
EXPOSE 8858
# With --bootstrap and an empty volume the service generates data, builds both stores, and trains first
# (about a minute at the default 50,000 cards), so the health check allows a long start period.
HEALTHCHECK --interval=15s --timeout=3s --start-period=240s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8858/healthz', timeout=2)"
CMD ["featstore", "serve", "--bootstrap"]
