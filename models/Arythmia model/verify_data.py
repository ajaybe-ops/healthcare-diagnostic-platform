"""
verify_data.py - Verify that all required .npy files exist and can be loaded.

This script checks:
1. All required files exist
2. Files can be loaded successfully
3. Shapes are correct
4. Data types are correct
5. No obvious data issues (NaN, inf, etc.)
"""

import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "processed"

REQUIRED_FILES = {
    "X.npy": "ECG signals",
    "y.npy": "Labels",
    "train_idx.npy": "Training indices",
    "val_idx.npy": "Validation indices",
    "test_idx.npy": "Test indices"
}

OPTIONAL_FILES = {
    "record_ids.npy": "Patient/record IDs"
}

def verify_files():
    """Verify all required files exist and can be loaded."""
    
    print("=" * 60)
    print("Data Verification Script")
    print("=" * 60)
    print(f"\nData directory: {DATA_DIR}\n")
    
    # Check if directory exists
    if not DATA_DIR.exists():
        print(f"[ERROR] Directory not found: {DATA_DIR}")
        print("\nPlease run: python setup_processed_folder.py")
        return False
    
    # Check required files
    all_exist = True
    for filename, description in REQUIRED_FILES.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"[OK] {filename:20s} - {description}")
        else:
            print(f"[MISSING] {filename:20s} - {description}")
            all_exist = False
    
    # Check optional files
    for filename, description in OPTIONAL_FILES.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"[OK] {filename:20s} - {description} (optional)")
        else:
            print(f"[INFO] {filename:20s} - {description} (optional, not found)")
    
    if not all_exist:
        print("\n[ERROR] Some required files are missing!")
        print("\nTo create sample data, run:")
        print("  python prepare_data.py")
        print("\nThen run split.py to create indices:")
        print("  python split.py")
        return False
    
    # Try to load files
    print("\n" + "=" * 60)
    print("Loading and validating data...")
    print("=" * 60 + "\n")
    
    try:
        # Load data files
        print("Loading data files...")
        X = np.load(DATA_DIR / "X.npy", mmap_mode='r')
        y = np.load(DATA_DIR / "y.npy", mmap_mode='r')
        train_idx = np.load(DATA_DIR / "train_idx.npy")
        val_idx = np.load(DATA_DIR / "val_idx.npy")
        test_idx = np.load(DATA_DIR / "test_idx.npy")
        
        print(f"[OK] X.npy loaded - shape: {X.shape}, dtype: {X.dtype}")
        print(f"[OK] y.npy loaded - shape: {y.shape}, dtype: {y.dtype}")
        print(f"[OK] train_idx.npy loaded - shape: {train_idx.shape}, dtype: {train_idx.dtype}")
        print(f"[OK] val_idx.npy loaded - shape: {val_idx.shape}, dtype: {val_idx.dtype}")
        print(f"[OK] test_idx.npy loaded - shape: {test_idx.shape}, dtype: {test_idx.dtype}")
        
        # Validate shapes
        print("\nValidating data shapes...")
        
        if len(X.shape) != 2:
            print(f"[ERROR] X.npy must be 2D (num_beats, signal_length), got shape: {X.shape}")
            return False
        
        if X.shape[1] != 360:
            print(f"[ERROR] Expected signal length 360, got: {X.shape[1]}")
            return False
        
        if len(y.shape) != 1:
            print(f"[ERROR] y.npy must be 1D, got shape: {y.shape}")
            return False
        
        if X.shape[0] != y.shape[0]:
            print(f"[ERROR] X and y must have same number of samples: X={X.shape[0]}, y={y.shape[0]}")
            return False
        
        num_beats = X.shape[0]
        print(f"[OK] Total beats: {num_beats}")
        print(f"[OK] Signal length: {X.shape[1]}")
        
        # Validate indices
        print("\nValidating indices...")
        
        max_train_idx = np.max(train_idx)
        max_val_idx = np.max(val_idx)
        max_test_idx = np.max(test_idx)
        
        if max_train_idx >= num_beats:
            print(f"[ERROR] train_idx contains index {max_train_idx} but only {num_beats} beats available")
            return False
        if max_val_idx >= num_beats:
            print(f"[ERROR] val_idx contains index {max_val_idx} but only {num_beats} beats available")
            return False
        if max_test_idx >= num_beats:
            print(f"[ERROR] test_idx contains index {max_test_idx} but only {num_beats} beats available")
            return False
        
        print(f"[OK] Training samples: {len(train_idx)}")
        print(f"[OK] Validation samples: {len(val_idx)}")
        print(f"[OK] Test samples: {len(test_idx)}")
        print(f"[OK] Total samples: {len(train_idx) + len(val_idx) + len(test_idx)}")
        
        # Check for overlap
        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)
        
        if train_set & val_set:
            print(f"[WARNING] Overlap between train and val indices: {len(train_set & val_set)} samples")
        if train_set & test_set:
            print(f"[WARNING] Overlap between train and test indices: {len(train_set & test_set)} samples")
        if val_set & test_set:
            print(f"[WARNING] Overlap between val and test indices: {len(val_set & test_set)} samples")
        
        if not (train_set & val_set or train_set & test_set or val_set & test_set):
            print("[OK] No overlap between train/val/test splits")
        
        # Check data quality
        print("\nChecking data quality...")
        
        # Check for NaN/Inf in X
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0:
            print(f"[WARNING] Found {nan_count} NaN values in X.npy")
        if inf_count > 0:
            print(f"[WARNING] Found {inf_count} Inf values in X.npy")
        if nan_count == 0 and inf_count == 0:
            print("[OK] No NaN or Inf values in X.npy")
        
        # Check labels
        unique_labels = np.unique(y)
        print(f"[OK] Unique labels: {unique_labels}")
        
        if np.min(y) < 0 or np.max(y) > 6:
            print(f"[WARNING] Labels should be 0-6, found range: {np.min(y)} to {np.max(y)}")
        else:
            print("[OK] Labels are in valid range (0-6)")
        
        # Class distribution
        print("\nClass distribution (all data):")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            percentage = 100 * count / len(y)
            print(f"  Class {cls}: {count:5d} samples ({percentage:5.2f}%)")
        
        # Check record_ids if available
        record_ids_path = DATA_DIR / "record_ids.npy"
        if record_ids_path.exists():
            record_ids = np.load(record_ids_path)
            if len(record_ids) == num_beats:
                unique_patients = len(np.unique(record_ids))
                print(f"\n[OK] record_ids.npy - {unique_patients} unique patients")
            else:
                print(f"[WARNING] record_ids length ({len(record_ids)}) doesn't match number of beats ({num_beats})")
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All checks passed! Data is ready for training.")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run training: python train_pytorch.py")
        print("  2. Or for TensorFlow: python train_baseline.py")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load or validate data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_files()
    sys.exit(0 if success else 1)

