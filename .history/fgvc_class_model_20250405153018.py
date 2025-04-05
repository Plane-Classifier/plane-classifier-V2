from ultralytics import YOLO
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "fgvc_aircraft_cls")
MODEL_TYPE = "yolov8n-cls.pt"  # You can change to yolov8s-cls.pt, yolov8m-cls.pt, etc.
EPOCHS = 50
IMAGE_SIZE = 224

# --- TRAINING ---
model = YOLO(MODEL_TYPE)

results = model.train(
    data=DATA_DIR,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE
)

print("Training complete.")
print("Results saved to:", results.save_dir)
