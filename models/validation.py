import numpy as np
import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import random

# ========== CONFIGURATION ==========
OUTPUT_DIR = Path("C:/Users/Lab/OneDrive/Desktop/Arythmia model/processed/")
X_FILE = OUTPUT_DIR / "X.npy"
Y_FILE = OUTPUT_DIR / "y.npy"
BEAT_LENGTH = 360
N_PLOTS = 5

# ========== LOGGING ==========
logging.basicConfig(
    level=logging.INFO, format='%(levelname)s: %(message)s')

def abort(msg):
    logging.error(msg)
    sys.exit(1)

def main():
    # ========== 1. Load Data (validate existence and memory usage) ==========
    if not X_FILE.exists() or not Y_FILE.exists():
        abort(f"Missing preprocessed files: {X_FILE} or {Y_FILE}")
    try:
        X = np.load(X_FILE, mmap_mode='r')  # Use mmap for memory efficiency
        y = np.load(Y_FILE, mmap_mode='r')
    except Exception as e:
        abort(f"Failed to load .npy files: {e}")

    # ========== 2. Shape Validation ==========
    if X.shape[0] != y.shape[0]:
        abort(f"Mismatch: Beats in X ({X.shape[0]}) != Labels in y ({y.shape[0]})")
    if X.shape[1] != BEAT_LENGTH:
        abort(f"Each ECG beat must have {BEAT_LENGTH} samples, but got {X.shape[1]}")
    logging.info(f"Loaded {X.shape[0]} beats with {BEAT_LENGTH} samples each.")

    # ========== 3. Class Distribution ==========
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logging.info(f"Class distribution: {class_dist}")

    # ========== 4. NaN/Inf/Flatline/Zero-Variance Detection ==========
    nan_mask = np.isnan(X) | np.isinf(X)
    nan_inf_idx = np.any(nan_mask, axis=1)
    num_bad = np.sum(nan_inf_idx)
    if num_bad > 0:
        abort(f"{num_bad} beats have NaN or infinite values.")
    stds = np.std(X, axis=1)
    flatline_idx = np.where(stds < 1e-6)[0]
    num_flat = len(flatline_idx)
    if num_flat > 0:
        abort(f"{num_flat} beats are flatline or zero-variance.")
    logging.info("No NaN, infinite, or flatline beats detected.")

    # ========== 5. Plot 5 Random Beats ==========
    random.seed(42)
    idxs = random.sample(range(X.shape[0]), min(N_PLOTS, X.shape[0]))
    import matplotlib
    matplotlib.use('Agg')   # For headless environments
    fig, axes = plt.subplots(N_PLOTS, 1, figsize=(8, 2 * N_PLOTS), constrained_layout=True)
    for i, ax in enumerate(axes if N_PLOTS > 1 else [axes]):
        idx = idxs[i]
        ax.plot(X[idx])
        ax.set_title(f"Beat #{idx}, Label={y[idx]}")
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Voltage')
    plot_path = OUTPUT_DIR / "random_beats.png"
    plt.savefig(plot_path)
    logging.info(f"Saved {N_PLOTS} random beat plots to {plot_path}")

    # ========== 6. Terminal Summary ==========
    print("================== DATASET HEALTH REPORT ==================")
    print(f"Total number of beats: {X.shape[0]}")
    print(f"ECG segment length: {BEAT_LENGTH}")
    print("Arrhythmia class distribution:")
    for k, v in class_dist.items():
        print(f"  Label {k}: {v} beats")
    print("No NaN, infinite, or flatline beats detected.")
    print(f"Sampled {N_PLOTS} random beats and saved to: {plot_path}")
    print("===========================================================")

if __name__ == "__main__":
    main()

