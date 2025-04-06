import streamlit as st
from PIL import Image
from model import load_model, predict

st.title("🧠 Image Recognition AI")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if "model" not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        label = predict(st.session_state.model, image)
        st.success(f"Prediction: **{label}**")
