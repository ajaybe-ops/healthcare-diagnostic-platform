# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import Response as OpenAPIResponse
import uvicorn
import numpy as np
import io
from PIL import Image
import torch
from torchvision import transforms
from datetime import datetime

# -----------------------------
# Device detection
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load models
# -----------------------------
try:
    arrhythmia_model = torch.load("arhythmia_model.pth", map_location=DEVICE)
    arrhythmia_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load Arrhythmia model: {str(e)}")

try:
    pneumonia_model = torch.load("pneumonia_model.pth", map_location=DEVICE)
    pneumonia_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load Pneumonia model: {str(e)}")

# -----------------------------
# FastAPI App Initialization
# -----------------------------
app = FastAPI(
    title="Health AI Diagnostic API",
    description="Unified API for Arrhythmia and Pneumonia prediction with confidence scores and combined predictions",
    version="1.1.0"
)

# CORS middleware for frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utility Functions
# -----------------------------
def predict_arrhythmia(signal: np.ndarray):
    """Predict Arrhythmia from 1D ECG signal"""
    if signal.ndim != 1:
        raise ValueError("ECG signal must be a 1D array")
    input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = arrhythmia_model(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        prediction_class = int(np.argmax(probabilities))
    return {"prediction": prediction_class, "confidence": float(probabilities[prediction_class])}

def predict_pneumonia(image_bytes: bytes):
    """Predict Pneumonia from X-ray image"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Invalid image file")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = pneumonia_model(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        prediction_class = int(np.argmax(probabilities))
    return {"prediction": prediction_class, "confidence": float(probabilities[prediction_class])}

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
async def root():
    return {
        "message": "Welcome to Health AI Diagnostic API. Use /predict for combined predictions or individual endpoints."
    }

@app.post("/predict")
async def predict_endpoint(
    ecg_file: UploadFile = File(..., description="Upload ECG CSV file for Arrhythmia prediction"),
    xray_file: UploadFile = File(..., description="Upload X-ray image file for Pneumonia prediction")
):
    """
    Unified prediction endpoint for both Arrhythmia and Pneumonia
    """
    try:
        # Process ECG
        ecg_content = await ecg_file.read()
        ecg_signal = np.loadtxt(io.BytesIO(ecg_content), delimiter=",")
        arrhythmia_result = predict_arrhythmia(ecg_signal)

        # Process X-ray
        xray_content = await xray_file.read()
        pneumonia_result = predict_pneumonia(xray_content)

        # Combined response
        response = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "device": str(DEVICE),
            "models": {
                "arrhythmia_model": "v1.0",
                "pneumonia_model": "v1.0"
            },
            "results": {
                "arrhythmia": arrhythmia_result,
                "pneumonia": pneumonia_result
            }
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/arhythmia")
async def predict_arrhythmia_endpoint(signal_file: UploadFile = File(...)):
    try:
        content = await signal_file.read()
        signal = np.loadtxt(io.BytesIO(content), delimiter=",")
        result = predict_arrhythmia(signal)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Arrhythmia prediction failed: {str(e)}")

@app.post("/predict/pneumonia")
async def predict_pneumonia_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_pneumonia(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Pneumonia prediction failed: {str(e)}")

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
