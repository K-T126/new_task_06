# MNIST ResNet18 Flask Service

This project implements an MNIST digit classification service using PyTorch Lightning (ResNet18) and Flask.

## Structure
- `model.py`: ResNet18 model definition adapted for MNIST.
- `train.py`: Training script with 60/20/20 data split.
- `app.py`: Flask application for serving predictions.
- `templates/index.html`: Web interface for drawing digits.
- `Dockerfile`: Container configuration for deployment.

## How to Run

### 1. Training (Local)
To train the model and generate `model_weights.pth`:
```bash
pip install -r requirements.txt
python train.py
```

### 2. Running the Flask App (Local)
```bash
python app.py
```
Open `http://localhost:8080` in your browser.

### 3. Docker
Build the image:
```bash
docker build -t mnist-service .
```
Run the container:
```bash
docker run -p 8080:8080 mnist-service
```

### 4. Deploying to GCP (Cloud Run)
1. Build and push the image to Google Artifact Registry:
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT_ID]/mnist-service
   ```
2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy mnist-service --image gcr.io/[PROJECT_ID]/mnist-service --platform managed --region us-central1 --allow-unauthenticated
   ```
