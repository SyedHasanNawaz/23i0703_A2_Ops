import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/api/predict")

def get_api_endpoint() -> str:
    """Normalize the API URL to guarantee targeting the prediction endpoint."""
    url = os.getenv("API_URL", API_URL).strip().rstrip("/")
    if not url.endswith("/predict"):
        url = f"{url}/api/predict"
    return url

def predict(raw_bytes: bytes):
    """
    Send raw image bytes to the Vercel FastAPI prediction endpoint
    and return (label, confidence, raw_probability).
    """
    endpoint = get_api_endpoint()
    files = {"image": ("image.jpg", raw_bytes, "image/jpeg")}

    try:
        response = requests.post(endpoint, files=files, timeout=45)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Error calling prediction API at {endpoint}: {exc}") from exc

    label = data.get("label", "Real")
    confidence = float(data.get("confidence", 0.0))
    # Use raw_probability if provided, or derive it from confidence and label
    if "raw_probability" in data:
        raw_prob = float(data["raw_probability"])
    else:
        raw_prob = confidence if label == "AI-Generated" else (1.0 - confidence)

    return label, confidence, raw_prob
