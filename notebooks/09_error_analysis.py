import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from app.data_pipeline import make_ds
from PIL import Image
import matplotlib.pyplot as plt

model = load_model("models/transfer_best.keras")
test_ds = make_ds("data/test.csv")

test_probs = model.predict(test_ds).ravel()
test_y = np.concatenate([y for _, y in test_ds])
preds = (test_probs >= 0.15).astype(int)

test_df = pd.read_csv("data/test.csv")
test_df["prob"] = test_probs
test_df["pred"] = preds
test_df["true"] = test_y

wrong = test_df[test_df["true"] != test_df["pred"]].head(10)

fig, ax = plt.subplots(2, 5, figsize=(20, 8))
for i, (_, row) in enumerate(wrong.iterrows()):
    r, c = divmod(i, 5)
    img = Image.open(row["path"]).convert("RGB")
    ax[r, c].imshow(img)
    
    true_label = "AI" if row["true"] == 1 else "Real"
    pred_label = "AI" if row["pred"] == 1 else "Real"
    conf = row["prob"] if row["pred"] == 1 else 1 - row["prob"]
    
    title = f"True: {true_label}\nPred: {pred_label} ({conf:.2f})"
    ax[r, c].set_title(title, fontsize=12, color="red" if row["true"] != row["pred"] else "black")
    ax[r, c].axis("off")

plt.suptitle("10 Misclassified Test Images", fontsize=16)
plt.tight_layout()
plt.savefig("artifacts/fig_errors.png", dpi=120, bbox_inches='tight')
plt.close()

print("Saved artifacts/fig_errors.png")