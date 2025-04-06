import os
# Avoid OpenMP crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from ultralytics import YOLO

# ------------------- CONFIG -------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "fgvc_aircraft_cls")

MODEL_TYPE = "yolo8m-cls.pt"
# MODEL_TYPE = "yolov8x-cls.pt"
EPOCHS = 50
IMAGE_SIZE = 256
BATCH_SIZE = 96
NUM_WORKERS = 2
USE_AMP = True
USE_CACHE = True

RUN_NAME = "aircraft_run"
SAVE_MODEL = True
SAVE_PERIOD = 1
PATIENCE = 10

# ------------------- MAIN -------------------

if __name__ == "__main__":

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Training will run on CPU.")

    model = YOLO(MODEL_TYPE)

    results = model.train(
        #resume = True,
        #data=DATA_DIR,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=NUM_WORKERS,
        amp=USE_AMP,
        cache=USE_CACHE,
        save=SAVE_MODEL,
        save_period=SAVE_PERIOD,
        patience=PATIENCE,
        project="runs/classify_yolo11m_50epoch_img256",
        name=RUN_NAME,
        exist_ok=True
    )

    print("\nTraining complete.")
    print("Results saved to:", results.save_dir)
