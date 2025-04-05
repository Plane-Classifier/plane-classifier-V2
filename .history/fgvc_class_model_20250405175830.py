from ultralytics import YOLO
import os
import torch

# CUDA CHECK
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available. Training will run on CPU.")

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "fgvc_aircraft_cls")
MODEL_TYPE = "yolov8m-cls.pt"  # Medium YOLOv8 classification model
EPOCHS = 50
IMAGE_SIZE = 224

# TRAINING
model = YOLO(MODEL_TYPE)

results = model.train(
    data=DATA_DIR,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE
)

print("\nTraining complete.")
print("Results saved to:", results.save_dir)
