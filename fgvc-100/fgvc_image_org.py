import os
import shutil
import pandas as pd

"""
This script reorganizes the FGVC Aircraft dataset for YOLOv8 classification mode.
It was run on 4.5.2025 and should not need to be run again unless the dataset structure changes.
"""

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "datasets", "fgvc_raw")  # correct directory name!
IMAGE_SRC = os.path.join(RAW_DIR, "fgvc-aircraft-2013b", "fgvc-aircraft-2013b", "data", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "fgvc_aircraft_cls")
splits = ["train", "val", "test"]

# --- REORGANIZE DATA ---
for split in splits:
    csv_path = os.path.join(RAW_DIR, f"{split}.csv")
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        filename = row['filename']
        label = row['Classes'].replace('/', '-')  # make safe for folder names

        target_dir = os.path.join(OUTPUT_DIR, split, label)
        os.makedirs(target_dir, exist_ok=True)

        src_img_path = os.path.join(IMAGE_SRC, filename)
        dst_img_path = os.path.join(target_dir, filename)

        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"[WARN] Missing file: {src_img_path}")

print(" Dataset reorganized for YOLOv8 classification mode.")
