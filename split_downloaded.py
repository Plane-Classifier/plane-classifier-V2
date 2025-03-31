import os
import shutil
import random
from collections import defaultdict
from glob import glob

# Configuration
input_dir = "downloaded_planes"
output_base = "dataset_subclass"
splits = ["train", "val", "test"]
ratios = [0.7, 0.2, 0.1]

# Ensure output directories exist
for split in splits:
    os.makedirs(os.path.join(output_base, "images", split), exist_ok=True)

# Gather all images grouped by subclass
subclass_groups = defaultdict(list)

image_paths = glob(f"{input_dir}/**/*.jpg", recursive=True)
for path in image_paths:
    filename = os.path.basename(path)
    subclass = filename.split("_")[0]  # e.g., A319, 737-4, etc.
    subclass_groups[subclass].append(path)

# Shuffle and distribute evenly
for subclass, images in subclass_groups.items():
    random.shuffle(images)
    n = len(images)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    split_map = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split, paths in split_map.items():
        for path in paths:
            filename = os.path.basename(path)
            dst_path = os.path.join(output_base, "images", split, filename)
            shutil.copy(path, dst_path)

print("✅ Downloaded planes split into train/val/test by subclass.")