import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from app.data_pipeline import make_ds, IMG_SIZE

tf.random.set_seed(42)
np.random.seed(42)

base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
)
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

train_ds = make_ds("data/train.csv", training=True)
val_ds = make_ds("data/val.csv")

cb = [
    callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
    callbacks.ModelCheckpoint("models/transfer_best.keras", save_best_only=True, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
]

print("--- STAGE 1: Training Head Only ---")
hist1 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb)

print("\n--- STAGE 2: Fine-Tuning Last 20 Layers ---")
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

hist2 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cb)

print(f"\nbest val_loss: {min(hist2.history['val_loss']):.4f}")
print(f"best val_acc : {max(hist2.history['val_accuracy']):.4f}")
np.save("artifacts/transfer_hist1.npy", hist1.history, allow_pickle=True)
np.save("artifacts/transfer_hist2.npy", hist2.history, allow_pickle=True)