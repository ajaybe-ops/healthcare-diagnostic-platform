# ===== Imports =====
import os
import numpy as np
import streamlit as st
import tensorflow as tf
import torch

# ===== Model builders FIRST =====
def build_arrhythmia_cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

class PneumoniaCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc = torch.nn.Linear(32 * 64 * 64, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ===== Paths =====
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

ARRHYTHMIA_WEIGHTS = os.path.join(MODEL_DIR, "baseline_cnn_weights.h5")
PNEUMONIA_MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.pth")

# ===== Load models (AFTER definitions) =====
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

# ===== Streamlit UI =====
st.title("AI Healthcare Diagnostic Platform")

arrhythmia_model, pneumonia_model = load_models()
st.success("Models loaded successfully")
