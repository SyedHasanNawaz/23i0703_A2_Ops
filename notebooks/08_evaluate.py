import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from app.data_pipeline import make_ds
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt

model = load_model("models/transfer_best.keras")

val_ds = make_ds("data/val.csv")
test_ds = make_ds("data/test.csv")

val_probs = model.predict(val_ds).ravel()
val_y = np.concatenate([y for _, y in val_ds])

grid = np.linspace(0.05, 0.95, 19)
f1s = [f1_score(val_y, (val_probs >= t).astype(int), zero_division=0) for t in grid]
best_thresh = grid[int(np.argmax(f1s))]
print(f"Chosen Threshold: {best_thresh:.2f}")

test_probs = model.predict(test_ds).ravel()
test_y = np.concatenate([y for _, y in test_ds])
pred = (test_probs >= best_thresh).astype(int)

print("\n--- TEST METRICS ---")
print(f"Accuracy : {accuracy_score(test_y, pred):.4f}")
print(f"Precision: {precision_score(test_y, pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(test_y, pred, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(test_y, pred, zero_division=0):.4f}")
print(f"ROC-AUC  : {roc_auc_score(test_y, test_probs):.4f}")

cm = confusion_matrix(test_y, pred)
print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(cm, cmap='Blues')
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='red', fontsize=16)
plt.savefig("artifacts/fig_confusion_matrix.png", dpi=120, bbox_inches='tight')
plt.close()

fpr, tpr, _ = roc_curve(test_y, test_probs)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(test_y, test_probs):.3f}")
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
plt.savefig("artifacts/fig_roc_curve.png", dpi=120, bbox_inches='tight')
plt.close()
print("\nSaved artifacts/fig_confusion_matrix.png and artifacts/fig_roc_curve.png")