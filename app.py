import os
import numpy as np
import streamlit as st
import tensorflow as tf
import torch

# ===== PATHS =====
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

ARRHYTHMIA_WEIGHTS = os.path.join(MODEL_DIR, "baseline_cnn_weights.h5")
PNEUMONIA_MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.pth")

# ===== STREAMLIT UI =====
st.title("AI Healthcare Diagnostic Platform")
st.subheader("Arrhythmia & Pneumonia Detection")

# ===== LOAD MODELS (cached) =====
@st.cache_resource
def load_models():
    arrhythmia_model = build_arrhythmia_cnn((360, 1), 5)
    arrhythmia_model.load_weights(ARRHYTHMIA_WEIGHTS)

    pneumonia_model = PneumoniaCNN(2)
    pneumonia_model.load_state_dict(
        torch.load(PNEUMONIA_MODEL_PATH, map_location="cpu")
    )
    pneumonia_model.eval()

    return arrhythmia_model, pneumonia_model

arrhythmia_model, pneumonia_model = load_models()
st.success("Models loaded successfully")
