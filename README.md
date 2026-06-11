# MNIST ResNet18 FastAPI サービス

このプロジェクトは、PyTorch Lightning (ResNet18) と FastAPI を使用して、MNIST 数字分類サービスを実装したものです。

## 構成
- `model.py`: MNIST 用に調整された ResNet18 モデルの定義。
- `train.py`: 60/20/20 のデータ分割を用いたトレーニングスクリプト。
- `app.py`: 予測を提供するための FastAPI アプリケーション。
- `templates/index.html`: 数字を記述するための Web インターフェース。
- `Dockerfile`: デプロイ用のコンテナ設定。

## 実行方法

### 1. トレーニング (ローカル)
モデルをトレーニングし、`model_weights.pth` を生成するには：
```bash
pip install -r requirements.txt
python train.py
```

### 2. FastAPI アプリの実行 (ローカル)
```bash
python app.py
```
ブラウザで `http://localhost:8080` を開きます。

### 3. Docker
イメージのビルド：
```bash
docker build -t mnist-service .
```
コンテナの実行：
```bash
docker run -p 8080:8080 mnist-service
```

### 4. GCP (Cloud Run) へのデプロイ
1. Google Artifact Registry にイメージをビルドしてプッシュします：
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT_ID]/mnist-service
   ```
2. Cloud Run にデプロイします：
   ```bash
   gcloud run deploy mnist-service --image gcr.io/[PROJECT_ID]/mnist-service --platform managed --region us-central1 --allow-unauthenticated
   ```
