import os
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from .data_pipeline import preprocess_single

MODEL_PATH = "models/transfer_best.keras"
THRESHOLD = 0.15  # The threshold we found in evaluation!

_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def predict(raw_bytes):
    model = get_model()
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    arr = preprocess_single(np.asarray(img))
    arr = np.expand_dims(arr, axis=0)
    
    prob = float(model.predict(arr, verbose=0)[0, 0])
    label = "AI-Generated" if prob >= THRESHOLD else "Real"
    conf = prob if prob >= THRESHOLD else 1.0 - prob
    
    return label, conf, prob