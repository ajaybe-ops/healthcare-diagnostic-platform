import os
import sys
import json
import numpy as np

# TensorFlow / Keras
import tensorflow as tf

# PyTorch
import torch

# ====== SETTINGS ======
BASE_DIR = "C:/Users/Lab/OneDrive/Desktop/Arrhythmia model/processed/"

# Arrhythmia (TensorFlow)
ARRHYTHMIA_WEIGHTS = os.path.join(BASE_DIR, "baseline_cnn_weights.h5")
INPUT_SHAPE = (360, 1)
NUM_CLASSES_ARRHYTHMIA = 5  # Replace with your actual number

# Pneumonia (PyTorch)
PNEUMONIA_MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model.pth")
NUM_CLASSES_PNEUMONIA = 2  # Replace with your actual number

# ====== 1. Build Arrhythmia CNN ======
def build_arrhythmia_cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# Load Arrhythmia model
arrhythmia_model = build_arrhythmia_cnn(INPUT_SHAPE, NUM_CLASSES_ARRHYTHMIA)
arrhythmia_model.load_weights(ARRHYTHMIA_WEIGHTS)
print("✅ Arrhythmia model loaded successfully!")

# ====== 2. Load Pneumonia PyTorch Model ======
# Define your pneumonia model architecture here
class PneumoniaCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc = torch.nn.Linear(32*64*64, num_classes)  # adjust for your input size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

pneumonia_model = PneumoniaCNN(NUM_CLASSES_PNEUMONIA)
pneumonia_model.load_state_dict(torch.load(PNEUMONIA_MODEL_PATH, map_location=torch.device('cpu')))
pneumonia_model.eval()
print("✅ Pneumonia model loaded successfully!")

# ====== 3. Example inference ======
def predict_arrhythmia(sample):
    sample = sample.astype(np.float32)[..., np.newaxis]
    pred = arrhythmia_model.predict(np.array([sample]))
    return np.argmax(pred, axis=1)[0]

def predict_pneumonia(sample):
    sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # adjust dims
    with torch.no_grad():
        logits = pneumonia_model(sample)
        pred = torch.argmax(logits, dim=1)
    return pred.item()
