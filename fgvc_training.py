import kagglehub
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
custom_path = os.path.join(BASE_DIR, "dataset")

path = kagglehub.dataset_download("seryouxblaster764/fgvc-aircraft", target_dir=custom_path)

