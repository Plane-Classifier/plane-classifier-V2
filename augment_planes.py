import os
import cv2
import glob
import albumentations as A
from tqdm import tqdm

input_img_base = "dataset_subclass/images"
input_label_base = "yolo_annotations"
output_img_base = "dataset_subclass/images"
output_label_base = "dataset_subclass/labels"

splits = ["train", "val", "test"]

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.2, rotate_limit=15, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

AUG_PER_IMAGE = 5  # Number of augmentations per image

for split in splits:
    img_dir = os.path.join(input_img_base, split)
    out_img_dir = os.path.join(output_img_base, split)
    out_lbl_dir = os.path.join(output_label_base, split)

    os.makedirs(out_lbl_dir, exist_ok=True)

    image_paths = glob.glob(f"{img_dir}/*.jpg")

    for img_path in tqdm(image_paths, desc=f"Augmenting {split}"):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(input_label_base, filename + ".txt")

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        bboxes = []
        class_labels = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            bboxes.append([float(x), float(y), float(w), float(h)])
            class_labels.append(cls)

        for i in range(AUG_PER_IMAGE):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception:
                continue

            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            aug_filename = f"{filename}_aug{i}.jpg"
            aug_labelname = f"{filename}_aug{i}.txt"

            cv2.imwrite(os.path.join(out_img_dir, aug_filename), aug_img)

            with open(os.path.join(out_lbl_dir, aug_labelname), "w") as f:
                for bbox, label in zip(aug_bboxes, aug_labels):
                    x, y, w, h = bbox
                    f.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")