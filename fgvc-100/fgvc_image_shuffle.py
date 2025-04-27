import os
import pandas as pd
import shutil

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_IMAGE_ROOT = os.path.join(BASE_DIR, "datasets", "fgvc_aircraft_cls")
RESHUFFLE_BASE = os.path.join(BASE_DIR, "datasets", "fgvc_raw")
OUTPUT_ROOT = os.path.join(BASE_DIR, "datasets", "fgvc_aircraft_cls_reshuffled")

# Load the reshuffled CSVs
csv_splits = {
    "train": pd.read_csv(os.path.join(RESHUFFLE_BASE, "train.csv")),
    "val": pd.read_csv(os.path.join(RESHUFFLE_BASE, "val.csv")),
    "test": pd.read_csv(os.path.join(RESHUFFLE_BASE, "test.csv")),
}

# Build map of all images
source_images = {}
for split in ["train", "val", "test"]:
    split_dir = os.path.join(RAW_IMAGE_ROOT, split)
    if not os.path.isdir(split_dir):
        continue
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                source_images[fname] = os.path.join(class_dir, fname)

# Copy files into new structure
copied_counts = {"train": 0, "val": 0, "test": 0}
for split_name, df in csv_splits.items():
    for _, row in df.iterrows():
        fname = row['filename']
        cls = row['class']
        src_path = source_images.get(fname)
        if not src_path or not os.path.exists(src_path):
            print(f"Missing: {fname}")
            continue

        dest_dir = os.path.join(OUTPUT_ROOT, split_name, cls)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, os.path.join(dest_dir, fname))
        copied_counts[split_name] += 1

# Report
print("Image redistribution complete.")
for split in copied_counts:
    print(f"{split.capitalize()} set: {copied_counts[split]} images")
