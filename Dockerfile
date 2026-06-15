# syntax=docker/dockerfile:1

# ==========================================
# 1. Base stage
# ==========================================
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ==========================================
# 2. Builder stage
# ==========================================
FROM base AS builder

WORKDIR /build

COPY requirements.txt .

# Install dependencies into /install prefix to avoid permission issues
# Using --prefix ensures all files are in one directory for easy copying
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir /install && \
    pip install --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1+cpu torchvision==0.15.2+cpu -r requirements.txt

# ==========================================
# 3. Final execution stage
# ==========================================
FROM base AS inference

WORKDIR /app

# Copy the installed packages from builder to /usr/local
# This makes them globally accessible to any user
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create and switch to non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# Uvicorn is now in /usr/local/bin, which is in the default PATH
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
