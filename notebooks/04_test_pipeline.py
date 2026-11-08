from app.data_pipeline import make_ds

train_ds = make_ds("data/train.csv", training=True)
val_ds = make_ds("data/val.csv")

print("train batches:", len(train_ds))
print("val batches  :", len(val_ds))

for imgs, lbls in train_ds.take(1):
    print("image batch shape:", imgs.shape)
    print("label batch shape:", lbls.shape)
    print("pixel range:", float(imgs.numpy().min()), "to", float(imgs.numpy().max()))
