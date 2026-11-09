import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from app.data_pipeline import make_ds, IMG_SIZE

tf.random.set_seed(42)
np.random.seed(42)


def build_baseline():
    m = tf.keras.Sequential([
        layers.Input((*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return m


model = build_baseline()
model.summary()

train_ds = make_ds("data/train.csv", training=True)
val_ds = make_ds("data/val.csv")

cb = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    callbacks.ModelCheckpoint("models/baseline_best.keras", save_best_only=True, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
]

hist = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=cb)

best = int(np.argmin(hist.history["val_loss"])) + 1
print(f"best epoch: {best}  val_loss={min(hist.history['val_loss']):.4f}")
np.save("artifacts/baseline_history.npy", hist.history, allow_pickle=True)
