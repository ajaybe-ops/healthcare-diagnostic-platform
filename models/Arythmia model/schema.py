import json
import numpy as np
from pathlib import Path
import sys

def abort(msg):
    print(f"SCHEMA ERROR: {msg}")
    sys.exit(1)

def validate_schema(data_dir):
    data_dir = Path(data_dir)
    schema_path = data_dir / "dataset_schema.json"
    x_path = data_dir / "X.npy"
    y_path = data_dir / "y.npy"
    # Load schema
    if not schema_path.exists():
        abort(f"Missing schema: {schema_path}")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    # Check files
    for p in [x_path, y_path]:
        if not p.exists():
            abort(f"Missing: {p}")
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    # Signal shape
    expected_shape = tuple(schema['input_signal_shape'])
    if X.shape[1:] != expected_shape:
        abort(f"X shape {X.shape[1:]} != expected {expected_shape}")
    # Data type
    if str(X.dtype) != schema['data_type']:
        abort(f"X dtype {X.dtype} != expected {schema['data_type']}")
    # Sampling rate (not checked in array, for doc only)
    # Label encoding
    possible_labels = set(int(k) for k in schema['label_encoding'].keys())
    actual_labels = set(y.tolist())
    if not actual_labels <= possible_labels:
        abort(f"Found unknown labels: {actual_labels - possible_labels}")
    # Number of classes
    if len(possible_labels) != schema['num_classes']:
        abort(f"num_classes in schema does not match label_encoding size")
    print("Schema validation PASSED.")

if __name__ == "__main__":
    validate_schema("C:/Users/Lab/OneDrive/Desktop/Arythmia model/processed/")

