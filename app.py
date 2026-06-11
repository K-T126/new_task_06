import io
import os
import torch
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import transforms
from model import MNISTResNet18

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTResNet18()
weights_path = 'model_weights.pth'
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded model weights from {weights_path}")
else:
    print(f"Warning: {weights_path} not found. Please train the model first.")
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={}
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('L')
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Debug: Print tensor stats to console
    print(f"Tensor stats - Mean: {tensor.mean():.4f}, Max: {tensor.max():.4f}, Min: {tensor.min():.4f}")
    
    with torch.no_grad():
        outputs = model(tensor)
        print(f"Logits: {outputs}")
        prediction = torch.argmax(outputs, dim=1).item()
        
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
