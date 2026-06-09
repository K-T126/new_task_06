# PyTorchがプリインストールされた軽量なランタイムイメージを使用
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 環境変数の設定
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係のインストール
# PyTorch以外のライブラリのみをインストールするようにしてビルドを高速化
COPY requirements.txt /app/
RUN pip install --no-cache-dir fastapi uvicorn python-multipart jinja2 pytorch-lightning

# プロジェクトファイルのコピー
# .dockerignoreにより、不要なデータやキャッシュは除外される
COPY . /app/

# ポートの公開
EXPOSE 8080

# アプリケーションの実行
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
