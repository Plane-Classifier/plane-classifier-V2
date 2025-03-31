import os
import shutil
from collections import defaultdict
from class_mapping import SUBCLASS_TO_CLASS

source_img_dir = "augmented/images"
source_lbl_dir = "augmented/labels"

subclass_dir = "dataset_subclass"
class_dir = "dataset_class"

for root in [subclass_dir, class_dir]:
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels"), exist_ok=True)

subclass_names = set()
class_names = set()

# Rewrite .txt files for each dataset
for file in os.listdir(source_lbl_dir):
    img_src = os.path.join(source_img_dir, file.replace(".txt", ".jpg"))
    lbl_src = os.path.join(source_lbl_dir, file)

    with open(lbl_src, "r") as f:
        lines = f.readlines()

    subclass_lines = []
    class_lines = []

    for line in lines:
        label, x, y, w, h = line.strip().split()
        subclass = label
        class_name = SUBCLASS_TO_CLASS.get(subclass)

        subclass_names.add(subclass)
        class_names.add(class_name)

        subclass_lines.append(f"{subclass} {x} {y} {w} {h}\\n")
        class_lines.append(f"{class_name} {x} {y} {w} {h}\\n")

    # Copy image and write new label files
    shutil.copy(img_src, os.path.join(subclass_dir, "images"))
    shutil.copy(img_src, os.path.join(class_dir, "images"))

    with open(os.path.join(subclass_dir, "labels", file), "w") as f:
        f.writelines(subclass_lines)

    with open(os.path.join(class_dir, "labels", file), "w") as f:
        f.writelines(class_lines)

# Save data.yaml files
def save_yaml(name_list, dataset_path, out_file):
    name_list = sorted(list(name_list))
    with open(out_file, "w") as f:
        f.write(f"path: {dataset_path}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("names:\n")
        for i, name in enumerate(name_list):
            f.write(f"  {i}: {name}\n")

save_yaml(subclass_names, os.path.abspath(subclass_dir), "data_subclass.yaml")
save_yaml(class_names, os.path.abspath(class_dir), "data_class.yaml")