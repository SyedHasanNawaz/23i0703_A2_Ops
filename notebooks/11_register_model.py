import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()

model_name = "VisionGuard"
run_id = "ef4fbcd91cb84b6581a560d4e101bcf2" # Paste your exact Run ID from the DagsHub page here

print(f"Registering model from run {run_id}...")

try:
    client.create_registered_model(model_name)
except Exception as e:
    print("Model already exists, continuing...")

mv = client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model",
    run_id=run_id
)
print(f"Created model version {mv.version}")

client.set_registered_model_alias(model_name, "champion", mv.version)
print("Set alias 'champion' to this version!")