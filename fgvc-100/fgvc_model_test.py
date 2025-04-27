import os
# Avoid OpenMP crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO



# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_yolo_files", "yolo11x-702010-50epoch-size256", "best.pt")
TEST_DIR = os.path.join(BASE_DIR, "common_test_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")
LOG_PATH = os.path.join(BASE_DIR, "output_log.txt")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "trained_yolo_files", "yolo11x-702010-50epoch-size256", "ground_truth.csv")  # Must contain 'filename','label'

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

# --- GROUND TRUTH ---
if os.path.exists(GROUND_TRUTH_FILE):
    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
    ground_truth.set_index("filename", inplace=True)
else:
    ground_truth = pd.DataFrame()

# --- CLASSIFICATION ---
predictions = []
total_images = sum(len(files) for _, _, files in os.walk(TEST_DIR))
image_count = 0

with open(LOG_PATH, "w") as log_file:
    for root, _, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                results = model.predict(source=image_path, conf=0.25, verbose=False)

                probs = results[0].probs
                if probs is not None:
                    top_class_id = int(probs.top1)
                    class_name = model.names[top_class_id]
                    confidence = probs.data[top_class_id].item()

                    # Overlay and save output
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    output_path = os.path.join(OUTPUT_DIR, f"output_{file}")
                    cv2.imwrite(output_path, img)

                    log_file.write(f"{file}: Predicted {class_name} ({confidence:.2f})\n")
                    predictions.append((file, class_name))

# --- EVALUATION ---
pred_df = pd.DataFrame(predictions, columns=["filename", "predicted"])
if not ground_truth.empty:
    eval_df = pred_df.join(ground_truth, on="filename")
    eval_df["correct"] = eval_df["predicted"] == eval_df["label"]
    accuracy = eval_df["correct"].mean()
    print(f"Accuracy: {accuracy:.2%}")
else:
    print("Ground truth file not found. Evaluation skipped.")
    eval_df = pred_df

# --- VISUAL SUMMARY ---
class_counts = Counter(eval_df["predicted"])
plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=90)
plt.title("Predicted Class Distribution")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "prediction_distribution.png"))
plt.close()

print("Classification complete. Results logged and visualization saved.")
