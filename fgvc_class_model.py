import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import threading
import time
import torch
from datetime import timedelta
from ultralytics import YOLO

# ------------------- CONFIGURATION -------------------

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "fgvc_aircraft_cls")

# Training parameters
MODEL_TYPE = "yolov8m-cls.pt"
EPOCHS = 30
IMAGE_SIZE = 192
BATCH_SIZE = 96
NUM_WORKERS = 2
USE_AMP = True
USE_CACHE = True

# Pause behavior
PAUSE_KEY = b'p'
PAUSE_TIMEOUT_SECONDS = 60

# ------------------- GPU Monitoring -------------------

def print_gpu_status():
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode()
        print("----- GPU STATUS -----")
        print(output.split("Processes:")[0].strip())
    except Exception as e:
        print("Could not retrieve GPU usage:", e)

# ------------------- Pause Mechanism -------------------

pause_flag = False
last_toggle_time = time.time()

def check_for_pause():
    global pause_flag, last_toggle_time
    try:
        import msvcrt
        print(f"\nPress '{PAUSE_KEY.decode()}' to pause/resume training. Auto-resumes after {PAUSE_TIMEOUT_SECONDS} seconds.\n")
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == PAUSE_KEY:
                    pause_flag = not pause_flag
                    last_toggle_time = time.time()
                    status = "paused" if pause_flag else "resumed"
                    print(f"\nTraining {status}.\n")

            if pause_flag and (time.time() - last_toggle_time > PAUSE_TIMEOUT_SECONDS):
                pause_flag = False
                print(f"\nNo input for {PAUSE_TIMEOUT_SECONDS} seconds. Auto-resuming training.\n")

            time.sleep(0.5)
    except ImportError:
        print("Pause feature is only supported on Windows with msvcrt.")

# ------------------- MAIN EXECUTION -------------------

if __name__ == "__main__":
    # Start pause monitoring thread
    pause_thread = threading.Thread(target=check_for_pause, daemon=True)
    pause_thread.start()

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("CUDA not available. Training will run on CPU.")

    model = YOLO(MODEL_TYPE)
    epoch_times = []

    for epoch_chunk in range(EPOCHS):
        while pause_flag:
            print("Paused... waiting.")
            time.sleep(2)

        print(f"\nStarting epoch {epoch_chunk + 1} of {EPOCHS}")
        print_gpu_status()

        epoch_start = time.time()

        results = model.train(
            data=DATA_DIR,
            epochs=1,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            workers=NUM_WORKERS,
            amp=USE_AMP,
            cache=USE_CACHE,
            verbose=False
        )

        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = EPOCHS - (epoch_chunk + 1)
        eta = timedelta(seconds=int(avg_epoch_time * remaining_epochs))

        print(f"Epoch {epoch_chunk + 1} duration: {timedelta(seconds=int(epoch_duration))}")
        print(f"Estimated time remaining: {eta}")

    print("\nTraining complete.")
    print("Results saved to:", results.save_dir)
