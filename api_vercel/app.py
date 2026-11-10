import os
import requests
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="VisionGuard Prediction API", version="1.0.0")

MODEL_URL = os.getenv(
    "DAGSHUB_MODEL_URL",
    "https://dagshub.com/api/v1/repos/syedhasannawaz05/23i0703_A2_Ops/storage/raw/s3/23i0703_A2_Ops/models/transfer_best.keras"
)
MODEL_LOCAL_PATH = "/tmp/model.keras"
THRESHOLD = float(os.getenv("THRESHOLD", "0.15"))
IMG_SIZE = (224, 224)

_model = None

def get_model():
    """Lazily download and cache the trained Keras model to /tmp."""
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_LOCAL_PATH) or os.path.getsize(MODEL_LOCAL_PATH) == 0:
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()

        temp_path = MODEL_LOCAL_PATH + ".tmp"
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(temp_path, MODEL_LOCAL_PATH)
        print(f"Model downloaded and saved to {MODEL_LOCAL_PATH}.")

    print("Loading Keras model into memory...")
    _model = tf.keras.models.load_model(MODEL_LOCAL_PATH)
    print("Model loaded successfully.")
    return _model

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image bytes to (1, 224, 224, 3) with MobileNetV2 scaling."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.get("/")
@app.get("/app.py")
@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "VisionGuard API",
        "threshold": THRESHOLD,
        "model_cached": os.path.exists(MODEL_LOCAL_PATH)
    }

@app.get("/api/predict")
@app.get("/predict")
def predict_get_help():
    return {
        "status": "ok",
        "message": "VisionGuard Prediction API is running. Submit a POST request with an image file to /api/predict to get a prediction."
    }

@app.post("/api/predict")
@app.post("/predict")
@app.post("/app.py")
@app.post("/")
async def predict_image(
    request: Request,
    file: UploadFile = File(None),
    image: UploadFile = File(None)
):
    try:
        # Support multipart form-data under 'file' or 'image', or raw body bytes
        target_file = file or image
        if target_file is not None:
            raw_bytes = await target_file.read()
        else:
            raw_bytes = await request.body()

        if not raw_bytes:
            raise HTTPException(status_code=400, detail="No image file or binary content provided.")

        # Validate that the bytes represent an image
        try:
            Image.open(BytesIO(raw_bytes)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Corrupt or invalid image payload.")

        model = get_model()
        processed = preprocess_image(raw_bytes)
        prob = float(model.predict(processed, verbose=0)[0, 0])

        label = "AI-Generated" if prob >= THRESHOLD else "Real"
        confidence = prob if prob >= THRESHOLD else (1.0 - prob)

        return JSONResponse(
            content={
                "label": label,
                "confidence": round(float(confidence), 4),
                "raw_probability": round(float(prob), 4)
            }
        )
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(exc)}"}
        )
