"""
prepare_data.py - Helper script to create sample dataset files for testing.

This script creates sample X.npy, y.npy, and record_ids.npy files 
for testing the training pipeline.

For real data, replace this with your actual preprocessing pipeline.
"""

import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "processed"
SCHEMA_PATH = BASE_DIR / "dataset_schema.json"

def load_schema():
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, 'r') as f:
            return json.load(f)
    return {
        "input_signal_shape": [360],
        "data_type": "float32",
        "num_classes": 7,
        "label_encoding": {
            "0": "Normal",
            "1": "Left bundle branch block",
            "2": "Right bundle branch block",
            "3": "Atrial premature",
            "4": "Premature ventricular",
            "5": "Fusion of ventricular and normal",
            "6": "Nodal escape"
        }
    }

def create_sample_data(num_beats=1000, num_patients=50, seed=42):
    """
    Create sample ECG data for testing.
    
    Args:
        num_beats: Number of ECG beats to generate
        num_patients: Number of unique patient IDs
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    schema = load_schema()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    signal_length = schema['input_signal_shape'][0]  # 360
    num_classes = schema['num_classes']  # 7
    
    print(f"Creating sample dataset...")
    print(f"  - Number of beats: {num_beats}")
    print(f"  - Number of patients: {num_patients}")
    print(f"  - Signal length: {signal_length}")
    print(f"  - Number of classes: {num_classes}")
    
    # Generate sample ECG signals (simulated)
    X = []
    y = []
    record_ids = []
    
    # Create patient IDs
    patient_ids = np.arange(num_patients)
    
    # Generate beats for each patient
    beats_per_patient = num_beats // num_patients
    remaining_beats = num_beats % num_patients
    
    beat_idx = 0
    for pid in patient_ids:
        num_patient_beats = beats_per_patient + (1 if pid < remaining_beats else 0)
        
        for _ in range(num_patient_beats):
            # Generate a simple synthetic ECG-like signal
            t = np.linspace(0, 2*np.pi, signal_length)
            
            # Random class label
            class_label = np.random.randint(0, num_classes)
            
            # Create signal with some variation based on class
            base_signal = np.sin(t) + 0.5 * np.sin(2*t)
            noise = np.random.normal(0, 0.1, signal_length)
            
            # Add class-specific characteristics
            if class_label == 0:  # Normal
                signal = base_signal + noise
            elif class_label in [1, 2]:  # Bundle branch blocks
                signal = base_signal * 1.2 + noise
            else:  # Other arrhythmias
                signal = base_signal * 0.8 + noise
            
            # Normalize
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            X.append(signal)
            y.append(class_label)
            record_ids.append(pid)
            beat_idx += 1
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    record_ids = np.array(record_ids, dtype=np.int64)
    
    # Save files
    X_path = DATA_DIR / "X.npy"
    y_path = DATA_DIR / "y.npy"
    record_ids_path = DATA_DIR / "record_ids.npy"
    
    np.save(X_path, X)
    np.save(y_path, y)
    np.save(record_ids_path, record_ids)
    
    print(f"\n[OK] Saved files:")
    print(f"  - {X_path}")
    print(f"  - {y_path}")
    print(f"  - {record_ids_path}")
    
    # Copy schema to processed folder
    schema_path = DATA_DIR / "dataset_schema.json"
    if SCHEMA_PATH.exists() and not schema_path.exists():
        import shutil
        shutil.copy(SCHEMA_PATH, schema_path)
        print(f"  - {schema_path} (copied)")
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - record_ids shape: {record_ids.shape}")
    print(f"  - Unique patients: {len(np.unique(record_ids))}")
    print(f"  - Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = schema['label_encoding'].get(str(cls), f"Class {cls}")
        print(f"    Class {cls} ({class_name}): {count} beats")
    
    print(f"\n[OK] Sample dataset created successfully!")
    print(f"\nNext steps:")
    print(f"  1. Run: python split.py  (to create train/val/test splits)")
    print(f"  2. Run: python train_pytorch.py  (to train the model)")

if __name__ == "__main__":
    import sys
    
    # Allow custom number of beats and patients
    num_beats = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    num_patients = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    create_sample_data(num_beats=num_beats, num_patients=num_patients)

