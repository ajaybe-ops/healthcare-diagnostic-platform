import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix

DATA_DIR = Path("C:/Users/Lab/OneDrive/Desktop/Arythmia model/processed/")
OUT_DIR = DATA_DIR / "error_analysis"
WEIGHTS_PATH = DATA_DIR / "baseline_cnn_weights.h5"
SPLIT_IDX_PATH = DATA_DIR / "test_idx.npy"
X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def abort(msg):
    logging.error("ERROR: " + msg)
    sys.exit(1)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimal CNN arch matcher for loading
def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

# Load data
def load_data():
    if not (X_PATH.exists() and Y_PATH.exists() and SPLIT_IDX_PATH.exists() and WEIGHTS_PATH.exists()):
        abort("Missing required files for error analysis.")
    X = np.load(X_PATH, mmap_mode='r')
    y = np.load(Y_PATH, mmap_mode='r')
    idx = np.load(SPLIT_IDX_PATH)
    Xtest = X[idx].astype(np.float32)[..., np.newaxis]
    ytest = y[idx]
    return Xtest, ytest

Xtest, ytest = load_data()
num_classes = int(ytest.max()) + 1

model = build_model((360,1), num_classes)
model.load_weights(WEIGHTS_PATH)
y_pred = np.argmax(model.predict(Xtest, batch_size=64), axis=1)
cm = confusion_matrix(ytest, y_pred, labels=list(range(num_classes)))

# Tracking FPs and FNs for each class
false_neg = {i: [] for i in range(num_classes)}
false_pos = {i: [] for i in range(num_classes)}
for i, (yt, yp) in enumerate(zip(ytest, y_pred)):
    if yt != yp:
        false_neg[yt].append(i)  # Missed actual
        false_pos[yp].append(i)  # Wrongly predicted

# Save misclassified npys
def save_npy(group, class_idx, indices):
    fname = OUT_DIR / f"{group}_class{class_idx}.npy"
    np.save(fname, Xtest[indices])
    return fname

for c in range(num_classes):
    if false_neg[c]:
        save_npy("FN", c, false_neg[c])
    if false_pos[c]:
        save_npy("FP", c, false_pos[c])

# Find most confused pairs
cm_nodiag = cm.copy()
np.fill_diagonal(cm_nodiag, 0)
most_confused = np.unravel_index(np.argmax(cm_nodiag), cm_nodiag.shape)

logging.info(f"Most confused: true class {most_confused[0]} mistaken for {most_confused[1]} (count={cm_nodiag[most_confused]})")

# Plots: for main misclassification pair
def plot_beat(idx, fname, title):
    plt.figure(figsize=(8, 2))
    plt.plot(Xtest[idx].squeeze(), lw=1)
    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Signal")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# Sample 5 of the major confusion type, or all if less
FN_idxs = [i for i in false_neg[most_confused[0]] if y_pred[i] == most_confused[1]]
FP_idxs = [i for i in false_pos[most_confused[1]] if ytest[i] == most_confused[0]]

for j, idx in enumerate(FN_idxs[:5]):
    plot_beat(idx, OUT_DIR / f"plot_FN_pair_{most_confused[0]}_as_{most_confused[1]}_{j}.png",
              f"True {most_confused[0]}, Pred {most_confused[1]} (FN)")
for j, idx in enumerate(FP_idxs[:5]):
    plot_beat(idx, OUT_DIR / f"plot_FP_pair_{most_confused[1]}_to_{most_confused[0]}_{j}.png",
              f"Pred {most_confused[1]}, True {most_confused[0]} (FP)")

print(f"Misclassified FN and FP beats saved as .npy files in {OUT_DIR}")
print(f"Plots for main misclassification pair saved in {OUT_DIR}")

