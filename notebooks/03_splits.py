import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("artifacts/dataset_audit.csv")
df["label"] = (df["cls"] == "AiArtData").astype(int)

train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

for name, part in [("train", train), ("val", val), ("test", test)]:
    out = part[["path", "label"]].copy()
    out.to_csv(f"data/{name}.csv", index=False)
    print(f"{name}: {len(out)} rows  labels={out['label'].value_counts().to_dict()}")

print()
print("saved data/train.csv data/val.csv data/test.csv")
