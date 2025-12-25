# Quick Start Guide

This guide will help you set up and run the Arrhythmia Classification pipeline from scratch.

## Step 1: Set Up Virtual Environment

### Option A: PowerShell (Recommended for Windows)
```powershell
cd "C:\Users\Lab\OneDrive\Desktop\Arythmia model"
.\setup_venv.ps1
.\venv\Scripts\Activate.ps1
```

### Option B: Command Prompt
```cmd
cd "C:\Users\Lab\OneDrive\Desktop\Arythmia model"
setup_venv.bat
venv\Scripts\activate.bat
```

### Option C: Manual
```bash
python -m venv venv
venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Set Up Data Folder

```bash
python setup_processed_folder.py
```

## Step 4: Prepare Data

### Option A: Use Sample Data (for testing)
```bash
python prepare_data.py 1000 50
python split.py
```

### Option B: Use Your Own Data
1. Place your preprocessed data in `processed/` folder:
   - `X.npy` - ECG signals (shape: num_beats, 360)
   - `y.npy` - Labels (shape: num_beats)
   - `record_ids.npy` - Patient IDs (shape: num_beats)

2. Create splits:
   ```bash
   python split.py
   ```

## Step 5: Verify Data

```bash
python verify_data.py
```

This script checks:
- All required files exist
- Files can be loaded
- Shapes are correct
- No data quality issues

Expected output:
```
[OK] X.npy loaded - shape: (1000, 360), dtype: float32
[OK] y.npy loaded - shape: (1000,), dtype: int64
[OK] train_idx.npy loaded - shape: (700,), dtype: int64
[OK] val_idx.npy loaded - shape: (150,), dtype: int64
[OK] test_idx.npy loaded - shape: (150,), dtype: int64
[SUCCESS] All checks passed! Data is ready for training.
```

## Step 6: Train Model

### PyTorch (Recommended)
```bash
python train_pytorch.py
```

### TensorFlow (if you prefer)
```bash
python train_baseline.py
```

Expected output:
```
Loading data...
Number of classes: 7
Training samples: 700
Validation samples: 150
Test samples: 150
Using device: cpu
Starting training...
Epoch 1/12 - Train Loss: 1.8234, Train Acc: 45.23%, Val Loss: 1.6543, Val Acc: 52.15%
...
✓ Model saved to: arhythmia_model.pth
✓ Metrics saved to: processed/baseline_cnn_metrics.json
```

## Step 7: Run Web Interface

After training, launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser automatically.

## Troubleshooting

### "Module not found" errors
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

### "File not found" errors
- Run: `python setup_processed_folder.py`
- Run: `python prepare_data.py` (for sample data)
- Run: `python split.py` (to create splits)

### "Data validation failed"
- Run: `python verify_data.py` to see detailed errors
- Check that X.npy has shape (N, 360)
- Check that y.npy has shape (N,)
- Ensure all indices are within valid range

### Training doesn't start
- Verify data files: `python verify_data.py`
- Check that model file path is correct
- Ensure you have enough memory for your dataset size

## Complete Workflow Summary

```bash
# 1. Setup
.\setup_venv.ps1
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Data preparation
python setup_processed_folder.py
python prepare_data.py 1000 50  # For sample data
python split.py

# 3. Verification
python verify_data.py

# 4. Training
python train_pytorch.py

# 5. Run app
streamlit run app.py
```

## File Checklist

After setup, you should have:

```
processed/
├── X.npy              ✓
├── y.npy              ✓
├── record_ids.npy     ✓
├── train_idx.npy      ✓
├── val_idx.npy        ✓
├── test_idx.npy       ✓
└── dataset_schema.json ✓

Root directory:
├── arhythmia_model.pth    ✓ (after training)
└── app.py                 ✓
```

## Next Steps

- Review `DATA_PREPARATION_GUIDE.md` for detailed data format requirements
- Check `processed/README.md` for dataset information
- Explore training scripts to customize hyperparameters
- Modify `app.py` to add new features or improve UI

