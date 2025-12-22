import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "models/pneumonia_model.h5"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="AI Pneumonia Detection",
    page_icon="🩺",
    layout="centered"
)

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# HEADER
# =========================
st.markdown(
    """
    <h1 style='text-align:center;'>🩺 AI Pneumonia Detection</h1>
    <p style='text-align:center; color:gray;'>
    Chest X-ray Screening Tool (Research & Educational Use)
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# INSTRUCTIONS
# =========================
st.markdown(
    """
    ### 📌 How to use
    1. Upload a **chest X-ray image**
    2. Click **Analyze X-ray**
    3. View prediction & confidence

    ⚠️ This system **does not replace a doctor**.
    """
)

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# IMAGE PROCESSING FUNCTION
# =========================
def preprocess_image(image: Image.Image):
    image = image.convert("L")
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = image.reshape(1, 224, 224, 1)
    return image

# =========================
# MAIN LOGIC
# =========================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.markdown("### 🖼️ Uploaded X-ray")
        st.image(image, use_column_width=True)

        if st.button("🔍 Analyze X-ray"):
            with st.spinner("Analyzing image using AI model..."):
                time.sleep(1.5)

                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)[0][0]

                pneumonia_prob = float(prediction)
                normal_prob = 1 - pneumonia_prob

            st.markdown("---")
            st.markdown("## 🧪 Analysis Result")

            # =========================
            # RESULTS DISPLAY
            # =========================
            if pneumonia_prob >= 0.5:
                st.error("⚠️ **Pneumonia Likely Detected**")
            else:
                st.success("✅ **Normal (No Pneumonia Detected)**")

            st.markdown("### 📊 Confidence Scores")

            st.progress(min(pneumonia_prob, 1.0))
            st.write(f"**Pneumonia Probability:** {pneumonia_prob*100:.2f}%")
            st.write(f"**Normal Probability:** {normal_prob*100:.2f}%")

            # =========================
            # EXPLANATION
            # =========================
            st.markdown(
                """
                ### 🧠 What does this mean?
                - The AI analyzed texture patterns in the X-ray
                - Higher pneumonia probability suggests abnormal lung opacity
                - Results depend on image quality and training data
                """
            )

            # =========================
            # MEDICAL DISCLAIMER
            # =========================
            st.markdown("---")
            st.warning(
                """
                ⚠️ **Medical Disclaimer**

                This AI tool is for **research and educational purposes only**.
                It is **not a medical device** and must **not** be used as a final diagnosis.
                Always consult a **qualified medical professional**.
                """
            )

    except Exception as e:
        st.error("❌ Error processing the image.")
        st.exception(e)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    """
    <p style='text-align:center; font-size:12px; color:gray;'>
    Built with ❤️ using Streamlit & Deep Learning<br>
    © 2026 AI Healthcare Research Project
    </p>
    """,
    unsafe_allow_html=True
)
