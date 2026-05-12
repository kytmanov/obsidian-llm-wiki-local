# syntax=docker/dockerfile:1

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /src
COPY . .

# Install the package into a dedicated prefix
RUN uv pip install --system --prefix /opt/olw .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# Git is required for auto-commit functionality
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Mark every directory as safe so bind-mounted vaults (owned by host UID) work
RUN git config --global safe.directory '*'

# Copy installed environment from builder
COPY --from=builder /opt/olw /opt/olw

ENV PATH="/opt/olw/bin:${PATH}" \
    PYTHONPATH="/opt/olw/lib/python3.11/site-packages"

ENTRYPOINT ["olw"]
CMD ["--help"]
