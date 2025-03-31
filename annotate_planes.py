from ultralytics import YOLO
import os
import cv2
import glob
from collections import defaultdict

# Load trained YOLOv8 model
model = YOLO("yolov8.pt")

# Input and output paths
image_dir = "downloaded_planes"
label_dir = "yolo_annotations"
vis_dir = "yolo_visuals"
os.makedirs(label_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# Track how many visualizations per class
visual_count = defaultdict(int)
MAX_VISUALS_PER_CLASS = 5

# Find all images
image_paths = glob.glob(f"{image_dir}/**/*.jpg", recursive=True)

for path in image_paths:
    filename = os.path.basename(path)
    subclass = filename.split("_")[0]

    results = model(path, conf=0.3)

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print(f"No detection: {filename}")
            continue

        # Get top box
        top_idx = boxes.conf.argmax().item()
        conf = boxes.conf[top_idx].item()
        if conf < 0.5:
            print(f"Low confidence ({conf:.2f}) - skipping {filename}")
            continue

        xywhn = boxes.xywhn[top_idx].cpu().numpy()
        x, y, w, h = xywhn[:4]

        # Save YOLO label file
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{subclass} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # Save visual with bounding box if under limit
        if visual_count[subclass] < MAX_VISUALS_PER_CLASS:
            img = cv2.imread(path)
            img_h, img_w = img.shape[:2]

            # Convert normalized coords to pixel coords
            x1 = int((x - w / 2) * img_w)
            y1 = int((y - h / 2) * img_h)
            x2 = int((x + w / 2) * img_w)
            y2 = int((y + h / 2) * img_h)

            # Draw box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, subclass, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            out_path = os.path.join(vis_dir, filename)
            cv2.imwrite(out_path, img)
            visual_count[subclass] += 1

        print(f"Labeled (top, conf {conf:.2f}): {filename}")