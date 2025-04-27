import os
import cv2
import glob
from ultralytics import YOLO
from tqdm import tqdm

# Load pretrained COCO model (knows what a generic airplane is)
model = YOLO("yolov8x.pt")  # class 5 = airplane

# Directories
input_img_dir = "downloaded_planes"
output_label_dir = "yolo_annotations"
os.makedirs(output_label_dir, exist_ok=True)

# Collect all image paths
image_paths = glob.glob(f"{input_img_dir}/**/*.jpg", recursive=True)

for img_path in tqdm(image_paths, desc="Annotating"):
    filename = os.path.splitext(os.path.basename(img_path))[0]
    subclass = filename.split("_")[0]  # e.g., 737-3, A319
    label_path = os.path.join(output_label_dir, filename + ".txt")

    results = model(img_path, conf=0.3)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        continue

    # Get all boxes that are class 5 (airplane)
    plane_indices = (boxes.cls.cpu().numpy() == 5)
    plane_boxes = boxes.xywhn[plane_indices]

    if len(plane_boxes) == 0:
        continue

    # Take top detection (highest confidence)
    top_box = plane_boxes[0].cpu().numpy()
    x, y, w, h = top_box[:4]

    with open(label_path, "w") as f:
        f.write(f"{subclass} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")