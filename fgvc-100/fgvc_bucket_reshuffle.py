import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

"""This script reshuffles the buckets from the default 0.33/0.33/0.33 split to 0.7/0.2/0.1 for train/val/test
"""

# --- CONFIGURATION ---
# Split ratios - Enter new ratios. Make sure they sum to 1.0
train_split = 0.7
val_split = 0.2 
test_split = 0.1

try:
    total = train_split + val_split + test_split
    if not abs(total - 1.0) < 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")
except ValueError as e:
    raise ValueError(f"Invalid split configuration: {e}")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "fgvc_aircraft_cls")
SPLITS = {'train': train_split, 'val': val_split, 'test': test_split}
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "fgvc_raw")

all_data = []

# Gather all labeled image paths
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(DATA_DIR, split)
    if not os.path.isdir(split_dir):
        continue
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                all_data.append({'filename': filename, 'class': class_name})

# Shuffle and load into DataFrame
random.shuffle(all_data)
df = pd.DataFrame(all_data)

# Stratified split
train_val, test = train_test_split(df, test_size=SPLITS['test'], stratify=df['class'], random_state=42)
train, val = train_test_split(train_val, test_size=SPLITS['val'] / (SPLITS['train'] + SPLITS['val']), stratify=train_val['class'], random_state=42)

# Save CSVs
os.makedirs(OUTPUT_DIR, exist_ok=True)
train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("CSV splits generated successfully:")
print(f"Train: {len(train)} entries")
print(f"Validation: {len(val)} entries")
print(f"Test: {len(test)} entries")
print("Dataset reshuffled and saved to:", OUTPUT_DIR)