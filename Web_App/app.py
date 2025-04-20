import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load your YOLOv8 classification model
st.set_page_config(page_title="Plane Classifier",
                   page_icon="✈️", layout="centered")


@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Make sure 'best.pt' is in the same folder
    return model


model = load_model()

# Class names (update this if you have your own)
# model.names is a dictionary: {0: 'Boeing 747', 1: 'F-16', ...}
CLASS_NAMES = model.names

# Define image classification function


def classify_image(image):
    results = model.predict(image)
    top1_id = int(results[0].probs.top1)
    confidence = float(results[0].probs.top1conf)
    label = CLASS_NAMES[top1_id]
    return label, confidence


# Streamlit UI
# st.set_page_config(page_title="Plane Classifier",
#                    page_icon="✈️", layout="centered")
st.title("✈️ Plane Classifier")
st.write("Upload an image of a plane and see what model it is!")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, confidence = classify_image(image)

    st.markdown(f"### ✈️ Prediction: **{label}**")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
