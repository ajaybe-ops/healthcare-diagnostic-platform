import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/pneumonia_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("AI Pneumonia Detection")
st.write("Upload a chest X-ray image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img.reshape(1, 224, 224, 1)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error("Pneumonia Detected")
    else:
        st.success("Normal")
