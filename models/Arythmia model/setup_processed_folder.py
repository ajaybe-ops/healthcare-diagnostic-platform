"""
setup_processed_folder.py - Quick setup script for processed folder.

This script creates the processed folder structure and copies necessary files.
Run this before prepare_data.py or when setting up with real data.
"""

from pathlib import Path
import shutil
import json

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "processed"
SCHEMA_PATH = BASE_DIR / "dataset_schema.json"

def setup_processed_folder():
    """Create processed folder and copy schema file."""
    
    print("Setting up processed folder...")
    
    # Create processed directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created directory: {DATA_DIR}")
    
    # Copy schema file
    schema_dst = DATA_DIR / "dataset_schema.json"
    if SCHEMA_PATH.exists():
        if not schema_dst.exists():
            shutil.copy(SCHEMA_PATH, schema_dst)
            print(f"[OK] Copied schema file: {schema_dst}")
        else:
            print(f"[OK] Schema file already exists: {schema_dst}")
    else:
        print(f"[WARNING] Source schema file not found: {SCHEMA_PATH}")
    
    # Create README if it doesn't exist
    readme_path = DATA_DIR / "README.md"
    if not readme_path.exists():
        readme_content = """# Processed Data Directory

This folder contains the preprocessed ECG dataset files required for training.

## Required Files

1. **X.npy** - ECG signal data (shape: num_beats, 360)
2. **y.npy** - Class labels (shape: num_beats)
3. **record_ids.npy** - Patient/record IDs (shape: num_beats)
4. **dataset_schema.json** - Data schema (already present)

## Quick Setup (Sample Data)

To create sample data for testing:
```bash
python prepare_data.py
```

## Next Steps

After creating X.npy, y.npy, and record_ids.npy:
```bash
python split.py  # Creates train_idx.npy, val_idx.npy, test_idx.npy
```
"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"[OK] Created README: {readme_path}")
    
    print("\n[OK] Setup complete!")
    print(f"\nNext steps:")
    print(f"  1. Create dataset files (X.npy, y.npy, record_ids.npy)")
    print(f"     - Option A: Use sample data - python prepare_data.py")
    print(f"     - Option B: Prepare real data from your raw dataset")
    print(f"  2. Create splits - python split.py")
    print(f"  3. Train model - python train_pytorch.py")

if __name__ == "__main__":
    setup_processed_folder()

