import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


df = pd.read_csv("artifacts/dataset_audit.csv")
classes = sorted(df["cls"].unique())
print("classes:", classes)


fig, ax = plt.subplots(2, 6, figsize=(18, 6))

for i, c in enumerate(classes):
    s = df[df["cls"] == c].sample(6, random_state=42)
    for j, (_, r) in enumerate(s.iterrows()):
        try:
            im = Image.open(r["path"]).convert("RGB")
            ax[i, j].imshow(im)
        except Exception:
            pass
        ax[i, j].set_title(f"{c}\n{r['width']}x{r['height']}", fontsize=8)
        ax[i, j].axis("off")

plt.suptitle("sample images per class", fontsize=13)
plt.tight_layout()
plt.savefig("artifacts/fig_sample_grid.png", dpi=120, bbox_inches="tight")
plt.close()
print("saved sample grid")


fig, ax = plt.subplots(figsize=(8, 4))
for c in classes:
    ax.hist(df[df["cls"] == c]["width"], bins=40, alpha=0.6, label=c)
ax.set_xlabel("width (px)")
ax.set_ylabel("count")
ax.set_title("width distribution per class")
ax.legend()
plt.tight_layout()
plt.savefig("artifacts/fig_width_hist.png", dpi=120, bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(figsize=(8, 4))
for c in classes:
    ax.hist(df[df["cls"] == c]["height"], bins=40, alpha=0.6, label=c)
ax.set_xlabel("height (px)")
ax.set_ylabel("count")
ax.set_title("height distribution per class")
ax.legend()
plt.tight_layout()
plt.savefig("artifacts/fig_height_hist.png", dpi=120, bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(figsize=(8, 4))
for c in classes:
    ax.hist(df[df["cls"] == c]["size"] / 1024, bins=40, alpha=0.6, label=c)
ax.set_xlabel("file size (KB)")
ax.set_ylabel("count")
ax.set_title("file size distribution per class")
ax.legend()
plt.tight_layout()
plt.savefig("artifacts/fig_size_hist.png", dpi=120, bbox_inches="tight")
plt.close()

print("all figures saved")
