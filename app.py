import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
MODEL_PATH = "models/pneumonia_model.h5"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

# ---------------- PAGE UI ----------------
st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 AI Pneumonia Detection")
st.caption("Chest X-ray based diagnostic support tool")
st.markdown("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")          # ✅ FIXED (3 channels)
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0) # (1,224,224,3)
    return image

# ---------------- UI INPUT ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)

    with st.spinner("🧠 Analyzing image..."):
        img = Image.open(uploaded_file)
        processed_img = preprocess_image(img)

        prediction = model.predict(processed_img)[0][0]
        st.caption(f"🔎 Raw model output: {prediction:.6f}")


    st.markdown("---")

    confidence = prediction if prediction > CONFIDENCE_THRESHOLD else 1 - prediction
    confidence = round(confidence * 100, 2)

    if prediction > CONFIDENCE_THRESHOLD:
        st.error(f"🚨 **Pneumonia Detected**\n\nConfidence: **{confidence}%**")
    else:
        st.success(f"✅ **Normal Chest X-ray**\n\nConfidence: **{confidence}%**")

    st.info(
        "⚠️ **Disclaimer:** This tool is for educational purposes only and is not a medical diagnosis."
    )
