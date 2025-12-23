import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
import logging
from pathlib import Path

DATA_DIR = Path("C:/Users/Lab/OneDrive/Desktop/Arythmia model/processed/")
OUT_DIR = DATA_DIR / "explainability"
WEIGHTS_PATH = DATA_DIR / "baseline_cnn_weights.h5"
X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
TEST_IDX_PATH = DATA_DIR / "test_idx.npy"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def abort(msg):
    logging.error("EXPLAINABILITY ERROR: " + msg)
    sys.exit(1)

def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

def grad_cam_1d(model, x, class_idx):
    grad_model = tf.keras.models.Model([model.inputs], [model.layers[-3].output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]  # shape: (L, C)
    pooled_grads = tf.reduce_mean(grads, axis=0)
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-9)
    # Upsample heatmap if needed
    heatmap = np.interp(np.arange(360), np.linspace(0, len(heatmap)-1, len(heatmap)), heatmap)
    return heatmap

def plot_with_cam(signal, heatmap, savepath, label, pred):
    plt.figure(figsize=(10,3))
    plt.plot(signal, label='ECG Signal', color='black')
    plt.twinx()
    plt.fill_between(np.arange(len(heatmap)), 0, heatmap, color='red', alpha=0.3, label='GradCAM')
    plt.title(f'GradCAM - True: {label} Pred: {pred}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Activation importance')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def main():
    # 1. Load test set beats & model
    for p in [X_PATH, Y_PATH, TEST_IDX_PATH, WEIGHTS_PATH]:
        if not p.exists():
            abort(f"Missing required file: {p}")
    X = np.load(X_PATH, mmap_mode='r')
    y = np.load(Y_PATH, mmap_mode='r')
    idx = np.load(TEST_IDX_PATH)
    Xtest = X[idx].astype(np.float32)[..., np.newaxis]
    ytest = y[idx]
    num_classes = int(ytest.max()) + 1
    model = build_model((360,1), num_classes)
    model.load_weights(WEIGHTS_PATH)
    y_pred = np.argmax(model.predict(Xtest, batch_size=64), axis=1)

    # 2. GradCAM for one typical beat per class
    for c in range(num_classes):
        idxs = np.where((ytest == c) & (y_pred == c))[0]
        if len(idxs):
            beat_idx = idxs[0]
            x = tf.convert_to_tensor(Xtest[beat_idx:beat_idx+1])
            heatmap = grad_cam_1d(model, x, c)
            plot_with_cam(Xtest[beat_idx].squeeze(), heatmap, OUT_DIR/f"GradCAM_class{c}.png", label=c, pred=c)
            logging.info(f"Saved GradCAM for correct class {c}")

    # 3. GradCAM on 1 egregious misclassification
    mismatches = np.where(ytest != y_pred)[0]
    if len(mismatches):
        idx0 = mismatches[0]
        x = tf.convert_to_tensor(Xtest[idx0:idx0+1])
        heatmap = grad_cam_1d(model, x, y_pred[idx0])
        plot_with_cam(Xtest[idx0].squeeze(), heatmap, OUT_DIR/f"GradCAM_misclass_true{ytest[idx0]}_pred{y_pred[idx0]}.png", label=ytest[idx0], pred=y_pred[idx0])
        logging.info(f"Saved GradCAM for misclassification true={ytest[idx0]} pred={y_pred[idx0]}")
    print(f"Saved GradCAM plots to {OUT_DIR}")

if __name__ == "__main__":
    main()

