import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd

ROOT = Path("data")
OUT_CSV = Path("artifacts/dataset_audit.csv")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

rows = []
seen_hashes = {}
duplicates = 0
corrupt = 0

for cls_dir in sorted(ROOT.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls_name = cls_dir.name
    for img_path in cls_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in EXTS:
            continue
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                w, h = im.size
                mode = im.mode
                fmt = im.format
            size = img_path.stat().st_size
            h256 = hashlib.sha256(img_path.read_bytes()).hexdigest()
            if h256 in seen_hashes:
                duplicates += 1
                continue
            seen_hashes[h256] = str(img_path)
            rows.append(dict(
                path=str(img_path), cls=cls_name,
                size=size, width=w, height=h,
                mode=mode, format=fmt, hash=h256,
            ))
        except Exception as e:
            corrupt += 1
            print("CORRUPT:", img_path, "->", e)

df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print("=" * 60)
print("Class counts:")
print(df["cls"].value_counts().to_string())
print()
print(f"Total valid images : {len(df)}")
print(f"Corrupt files      : {corrupt}")
print(f"Exact duplicates   : {duplicates}")
print()
for col in ["width", "height", "size"]:
    print(f"{col:>7}  min={df[col].min():>10}  median={int(df[col].median()):>10}  max={df[col].max():>10}")
print()
print("Formats:", df["format"].value_counts().to_dict())
print("Modes  :", df["mode"].value_counts().to_dict())
print("=" * 60)
print(f"Saved to {OUT_CSV}")
