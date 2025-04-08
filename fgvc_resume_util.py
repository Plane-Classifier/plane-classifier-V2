import os
# Avoid OpenMP crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

"""
This script will resume a crashed training run of a yolov8 model.
Point it at the latest file in the weights directory of that run.
It will then resume training from that point, using the same parameters as the original run.
Key variables:
RUN_DIR = os.path.join(BASE_DIR, "path", "to", "run", "directory") -- This is the metadata directory for the run
CHECKPOINT_PATH = os.path.join(RUN_DIR, "weights", "last.pt") -- This is the path to the last checkpoint file
FINAL_EPOCHS = 50 -- This is the final number of epochs you want to train for. Do not try to update for epochs remaining.
    This can also be set to add additional epochs after the run is completed.
"""

# --- CONFIGURATION ---

# Path to the run directory that contains weights/last.pt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(BASE_DIR, "runs", "classify_yolom_50epoch_img256", "aircraft_run")
CHECKPOINT_PATH = os.path.join(RUN_DIR, "weights", "last.pt")

# Final target epoch count
FINAL_EPOCHS = 50

# --- MAIN ---

if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")


    print(f"Resuming training from: {CHECKPOINT_PATH}")
    print(f"Targeting {FINAL_EPOCHS} total epochs.")

    # Load the model and resume
    model = YOLO(CHECKPOINT_PATH)
    results = model.train(
        resume=True,
        epochs=FINAL_EPOCHS
    )

    print("\nTraining resumed and completed.")
    print("Results saved to:", results.save_dir)
