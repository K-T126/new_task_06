# syntax=docker/dockerfile:1

# ==========================================
# 1. 共有のベース設定
# ==========================================
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ==========================================
# 2. 学習用ステージ（GPU対応、重量級）
# ==========================================
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS training

WORKDIR /app

# 依存関係のインストール（キャッシュマウントを利用）
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .
CMD ["python", "train.py"]


# ==========================================
# 3. 推論用ビルダーステージ（依存関係の構築）
# ==========================================
FROM base AS builder

WORKDIR /build

# ビルドに必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 非特権ユーザーを作成（builder内でも同じUID/GIDで作成）
RUN useradd -m appuser
USER appuser

# 依存関係のインストール
# --extra-index-url を使用して、PyTorch CPU版と通常のPyPIパッケージを両立
COPY requirements.txt .
RUN --mount=type=cache,target=/home/appuser/.cache/pip,uid=1000 \
    pip install --user --upgrade pip && \
    pip install --user --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    -r requirements.txt


# ==========================================
# 4. 推論用実行ステージ（軽量・高速）
# ==========================================
FROM base AS inference

WORKDIR /app

# builderからappuserのインストール済みパッケージをコピー
COPY --from=builder /home/appuser/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# アプリケーションコードのコピー
COPY . .

# 非特権ユーザーの設定
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
