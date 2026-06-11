import io
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
try:
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
except FileNotFoundError:
    print("Warning: model_weights.pth not found. Please train the model first.")
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
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
    image = Image.open(io.BytesIO(img_bytes))
    
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        prediction = torch.argmax(outputs, dim=1).item()
        
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
