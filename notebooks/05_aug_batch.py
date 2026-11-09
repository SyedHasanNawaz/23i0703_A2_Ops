import matplotlib.pyplot as plt
from app.data_pipeline import make_ds

ds = make_ds("data/train.csv", training=True)

for imgs, lbls in ds.take(1):
    fig, ax = plt.subplots(2, 8, figsize=(20, 5))
    for i in range(16):
        r, c = divmod(i, 8)
        img = imgs[i].numpy().astype("uint8")
        ax[r, c].imshow(img)
        ax[r, c].set_title(f"label={int(lbls[i])}", fontsize=9)
        ax[r, c].axis("off")
    plt.suptitle("one augmented training batch", fontsize=14)
    plt.tight_layout()
    plt.savefig("artifacts/fig_aug_batch.png", dpi=120, bbox_inches="tight")
    plt.close()

print("saved artifacts/fig_aug_batch.png")
