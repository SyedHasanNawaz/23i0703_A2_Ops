import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import subprocess
from dotenv import load_dotenv
from tensorflow.keras import layers, callbacks
from app.data_pipeline import make_ds, IMG_SIZE

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("VisionGuard_Transfer")

tf.random.set_seed(42)
np.random.seed(42)

class MLflowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items():
            mlflow.log_metric(k, v, step=epoch)

def build_model():
    base = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    inputs = tf.keras.Input((*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model, base

train_ds = make_ds("data/train.csv", training=True)
val_ds = make_ds("data/val.csv")

with mlflow.start_run(run_name="manual_logging_run"):
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("lr", 1e-3)
    
    git_hash = subprocess.getoutput("git rev-parse HEAD")
    mlflow.set_tag("git_commit", git_hash)
    mlflow.set_tag("seed", 42)
    
    model, base = build_model()
    
    cb = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
        callbacks.ModelCheckpoint("models/mlflow_best.keras", save_best_only=True, monitor="val_loss"),
        MLflowCallback()
    ]
    
    print("--- STAGE 1: Training Head ---")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb)
    
    mlflow.tensorflow.log_model(model, "model")
    mlflow.log_artifact("requirements.txt")
    
    print("Finished and logged to DagsHub!")