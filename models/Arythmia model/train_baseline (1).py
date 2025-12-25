import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# =========== SETTINGS ===========
DATA_DIR = Path("C:/Users/Lab/OneDrive/Desktop/Arythmia model/processed/")
WEIGHTS_PATH = DATA_DIR / "baseline_cnn_weights.h5"
METRICS_PATH = DATA_DIR / "baseline_cnn_metrics.json"
LOGS_PATH = DATA_DIR / "baseline_cnn_train.log"

INPUT_SHAPE = (360, 1)
EPOCHS = 12
BATCH_SIZE = 64
SEED = 42

def abort(msg):
    print(f"CRITICAL ERROR: {msg}")
    sys.exit(1)

# =========== 1. LOAD SPLITS ===========
def load_split(indices_file):
    idx = np.load(DATA_DIR / indices_file)
    return idx

X = np.load(DATA_DIR / "X.npy", mmap_mode='r')
y = np.load(DATA_DIR / "y.npy", mmap_mode='r')
train_idx = load_split("train_idx.npy")
val_idx = load_split("val_idx.npy")
test_idx = load_split("test_idx.npy")

# Subset data according to splits
X_train = X[train_idx]
y_train = y[train_idx]
X_val = X[val_idx]
y_val = y[val_idx]
X_test = X[test_idx]
y_test = y[test_idx]

# =========== 2. DATA PREP ===========
# Ensure channels-last and float32
def to_model_input(arr):
    if arr.ndim == 2:
        return arr.astype(np.float32)[..., np.newaxis]
    return arr.astype(np.float32)

X_train = to_model_input(X_train)
X_val = to_model_input(X_val)
X_test = to_model_input(X_test)

num_classes = len(np.unique(y))

# =========== 3. CLASS WEIGHTS ===========
class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# =========== 4. MODEL DEFINITION ===========
def build_cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

model = build_cnn(INPUT_SHAPE, num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========== 5. TRAIN ===========
callback = tf.keras.callbacks.CSVLogger(str(LOGS_PATH))
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=[callback],
    verbose=2
)

model.save_weights(WEIGHTS_PATH)

# =========== 6. EVALUATE ===========
def eval_and_report(X, y, split_name):
    y_pred = np.argmax(model.predict(X, batch_size=BATCH_SIZE), axis=1)
    acc = accuracy_score(y, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, y_pred, average=None, labels=range(num_classes), zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=range(num_classes))
    
    print(f"\n===== {split_name.upper()} METRICS =====")
    print(f"Accuracy: {acc:.4f}")
    print("Per-class:")
    for i in range(num_classes):
        print(f"  Class {i}: Precision={prec[i]:.4f} Recall={rec[i]:.4f} F1={f1[i]:.4f} Support={sup[i]}")
    print(f"Confusion matrix:\n{cm}")
    
    return {
        'split': split_name,
        'accuracy': float(acc),
        'precision': prec.tolist(),
        'recall': rec.tolist(),
        'f1': f1.tolist(),
        'support': sup.tolist(),
        'confusion_matrix': cm.tolist()
    }

metrics = {}
metrics['val'] = eval_and_report(X_val, y_val, 'Validation')
metrics['test'] = eval_and_report(X_test, y_test, 'Test')

with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved model weights to {WEIGHTS_PATH}")
print(f"Saved metrics JSON to {METRICS_PATH}")
print(f"Saved training logs to {LOGS_PATH}")
