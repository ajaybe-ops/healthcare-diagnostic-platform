"""
train_arrhythmia_model.py

Single end-to-end script for training arrhythmia detection model.
Reads MIT-BIH dataset, trains model, saves ONE file: arrhythmia_model.h5

Usage: python train_arrhythmia_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
DATASET_PATH = Path(r"C:\Users\Lab\OneDrive\Desktop\datasets medical\Arrythmia\mit-bih-arrhythmia-database-1.0.0")
MODEL_OUTPUT_PATH = Path(__file__).parent.absolute() / "arrhythmia_model.h5"
BEAT_LENGTH = 360
FS = 360
SEED = 42
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# MIT-BIH annotation to class mapping (AAMI standard)
ANNOTATION_MAP = {
    'N': 0,  # Normal beat
    'L': 1,  # Left bundle branch block
    'R': 2,  # Right bundle branch block
    'A': 3,  # Atrial premature
    'V': 4,  # Premature ventricular
    'F': 5,  # Fusion of ventricular and normal
    'E': 6,  # Ventricular escape
    'J': 6,  # Nodal escape
    '/': 6,  # Paced beat (map to Other)
    'f': 5,  # Fusion
}

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def preprocess_signal(signal):
    """Bandpass filter and normalize ECG signal."""
    # Butterworth bandpass filter (0.5-40 Hz)
    nyquist = FS / 2
    low = 0.5 / nyquist
    high = 40.0 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    # Normalize
    normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    return normalized

def extract_beats(signal, r_peaks, annotation_symbols):
    """Extract ECG beats centered on R-peaks."""
    beats = []
    labels = []
    half_win = BEAT_LENGTH // 2
    
    for i, r_peak in enumerate(r_peaks):
        start = r_peak - half_win
        end = r_peak + half_win
        
        if start >= 0 and end <= len(signal):
            beat = signal[start:end]
            if len(beat) == BEAT_LENGTH:
                symbol = annotation_symbols[i]
                if symbol in ANNOTATION_MAP:
                    beats.append(beat)
                    labels.append(ANNOTATION_MAP[symbol])
    
    if len(beats) == 0:
        return np.array([]), np.array([], dtype=np.int64)
    
    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)

def load_mitbih_dataset(dataset_path):
    """Load all records from MIT-BIH Arrhythmia dataset."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    print(f"Loading MIT-BIH dataset from: {dataset_path}")
    
    all_beats = []
    all_labels = []
    all_record_ids = []
    
    # Get all .hea files (header files indicate records)
    record_files = sorted(set(f.stem for f in dataset_path.glob("*.hea")))
    
    print(f"Found {len(record_files)} record files")
    
    for record_id in record_files:
        try:
            record_path = dataset_path / record_id
            
            # Read signal and annotation
            record = wfdb.rdrecord(str(record_path))
            annotation = wfdb.rdann(str(record_path), 'atr')
            
            # Use first channel (MLII or lead II)
            signal = record.p_signal[:, 0]
            
            # Preprocess signal
            signal = preprocess_signal(signal)
            
            # Get R-peaks and annotations
            r_peaks = annotation.sample
            annotation_symbols = annotation.symbol
            
            # Extract beats
            beats, labels = extract_beats(signal, r_peaks, annotation_symbols)
            
            if len(beats) > 0:
                all_beats.append(beats)
                all_labels.append(labels)
                all_record_ids.extend([int(record_id)] * len(beats))
                
            print(f"  Record {record_id}: {len(beats)} beats extracted")
            
        except Exception as e:
            print(f"  Warning: Failed to process record {record_id}: {e}")
            continue
    
    if len(all_beats) == 0:
        raise ValueError("No beats extracted from dataset!")
    
    # Concatenate all beats
    X = np.concatenate(all_beats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    record_ids = np.array(all_record_ids)
    
    print(f"\nTotal beats extracted: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        class_names = {0: 'Normal', 1: 'LBBB', 2: 'RBBB', 3: 'APC', 4: 'PVC', 5: 'Fusion', 6: 'Other'}
        print(f"  Class {cls} ({class_names.get(cls, 'Unknown')}): {count} beats")
    
    return X, y, record_ids

def create_patient_stratified_split(X, y, record_ids, test_size=0.2, val_size=0.1, random_state=42):
    """Create patient-stratified train/val/test split."""
    unique_records = np.unique(record_ids)
    
    # Split records (patients) first
    records_train, records_temp = train_test_split(
        unique_records, test_size=test_size + val_size, random_state=random_state
    )
    records_val, records_test = train_test_split(
        records_temp, test_size=test_size / (test_size + val_size), random_state=random_state
    )
    
    # Create masks
    train_mask = np.isin(record_ids, records_train)
    val_mask = np.isin(record_ids, records_val)
    test_mask = np.isin(record_ids, records_test)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} beats ({len(records_train)} records)")
    print(f"  Val:   {len(X_val)} beats ({len(records_val)} records)")
    print(f"  Test:  {len(X_test)} beats ({len(records_test)} records)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape, num_classes):
    """Build CNN-LSTM model for ECG arrhythmia classification."""
    inputs = keras.Input(shape=input_shape)
    
    # CNN feature extraction
    x = keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # LSTM for temporal patterns
    x = keras.layers.LSTM(64, return_sequences=True)(x)
    x = keras.layers.LSTM(32, return_sequences=False)(x)
    x = keras.layers.Dropout(0.3)(x)
    
    # Classification head
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    print("=" * 70)
    print("MIT-BIH Arrhythmia Model Training")
    print("=" * 70)
    
    set_seed(SEED)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    X, y, record_ids = load_mitbih_dataset(DATASET_PATH)
    
    # Prepare data shape for model
    X = X[..., np.newaxis]  # Add channel dimension: (N, 360, 1)
    num_classes = len(np.unique(y))
    
    print(f"\n[2/5] Creating patient-stratified splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_patient_stratified_split(
        X, y, record_ids, test_size=0.15, val_size=0.15
    )
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in zip(np.unique(y_train), class_weights)}
    
    print(f"\n[3/5] Building model...")
    model = build_model(input_shape=(BEAT_LENGTH, 1), num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print(f"\n[4/5] Training model...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print(f"\n[5/5] Final evaluation...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nFinal Results:")
    print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    print(f"  Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Save model
    print(f"\nSaving model to: {MODEL_OUTPUT_PATH}")
    model.save(str(MODEL_OUTPUT_PATH))
    
    # Verify model can be loaded
    print("Verifying model can be loaded...")
    loaded_model = keras.models.load_model(str(MODEL_OUTPUT_PATH))
    test_pred = loaded_model.predict(X_test[:1], verbose=0)
    print(f"[OK] Model saved and verified successfully!")
    print(f"[OK] Model file: {MODEL_OUTPUT_PATH}")
    print(f"[OK] File size: {MODEL_OUTPUT_PATH.stat().st_size / (1024*1024):.2f} MB")
    
    print("\n" + "=" * 70)
    print("Training complete! Model saved as: arrhythmia_model.h5")
    print("=" * 70)
    print("\nTo use the model:")
    print(f"  model = keras.models.load_model('{MODEL_OUTPUT_PATH.name}')")

if __name__ == "__main__":
    main()

