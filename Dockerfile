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

# Install heavy dependencies first to leverage caching
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir /install && \
    pip install --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1+cpu torchvision==0.15.2+cpu

# Install other dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

# ==========================================
# 3. Final execution stage
# ==========================================
FROM base AS inference

WORKDIR /app

# Copy the installed packages from builder to /usr/local
COPY --from=builder /install /usr/local

# Copy application files individually to maximize cache efficiency
COPY model_weights.pth .
COPY model.py app.py ./
COPY templates/ ./templates/

# Create and switch to non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# Uvicorn is now in /usr/local/bin, which is in the default PATH
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
