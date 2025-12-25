# Model Evaluation Guide

This guide explains how to evaluate your trained arrhythmia classification model with comprehensive metrics, visualizations, and inference testing.

## Quick Start

After training your model, evaluate it with:

```bash
python evaluate_model.py
```

This will:
- Evaluate on validation and test sets
- Calculate accuracy, precision, recall, F1-score
- Generate confusion matrices
- Create prediction visualizations
- Save all results to `evaluation_results/`

## Evaluation Scripts

### 1. `evaluate_model.py` - Comprehensive Evaluation

**Full evaluation with metrics and visualizations**

**Usage:**
```bash
python evaluate_model.py
```

**What it does:**
- Loads trained model (PyTorch or TensorFlow)
- Evaluates on validation and test sets
- Calculates comprehensive metrics:
  - Overall accuracy
  - Per-class precision, recall, F1-score
  - Macro-averaged metrics
  - Confusion matrices
- Generates visualizations:
  - Confusion matrix plots (validation and test)
  - Sample predictions with ECG signals
- Saves all results to `evaluation_results/` folder

**Output files:**
```
evaluation_results/
├── evaluation_metrics.json          # Detailed metrics in JSON
├── confusion_matrix_validation.png  # Confusion matrix for validation set
├── confusion_matrix_test.png        # Confusion matrix for test set
└── prediction_samples.png           # 10 sample predictions with ECG signals
```

**Example output:**
```
============================================================
VALIDATION METRICS
============================================================

Overall Accuracy: 0.8533

Macro Averages:
  Precision: 0.8421
  Recall:    0.8356
  F1-Score:  0.8388

Per-Class Metrics:
Class                          Precision    Recall       F1          Support   
--------------------------------------------------------------------------------------------
Normal                         0.9123       0.9456       0.9287       234       
Left bundle branch block       0.8234       0.7892       0.8059       89        
...
```

### 2. `test_model_inference.py` - Quick Inference Test

**Simple script to test model can make predictions**

**Usage:**
```bash
python test_model_inference.py
```

**What it does:**
- Loads the trained model
- Takes a single sample from test set
- Makes a prediction
- Shows true vs predicted label
- Displays confidence scores for all classes

**Useful for:**
- Quick verification that model loads correctly
- Testing inference speed
- Debugging prediction issues

**Example output:**
```
============================================================
Model Inference Test
============================================================

1. Loading model...
   [OK] Model loaded on device: cpu

2. Loading test data...
   [OK] Loaded sample 856
   [OK] True label: 0 (Normal)

3. Making prediction...
   [OK] Prediction complete

4. Results:
   True Label:  0 - Normal
   Predicted:   0 - Normal
   Confidence:  0.9234 (92.34%)

   [CORRECT] Prediction matches true label!
```

## Metrics Explained

### Accuracy
Overall percentage of correct predictions across all classes.

### Precision (Per-Class)
Of all predictions for a class, what percentage were correct?
- High precision = few false positives

### Recall (Per-Class)
Of all actual instances of a class, what percentage were correctly identified?
- High recall = few false negatives

### F1-Score (Per-Class)
Harmonic mean of precision and recall.
- Balanced metric that considers both precision and recall

### Macro Averages
Average of per-class metrics (treats all classes equally).

### Confusion Matrix
Shows how predictions are distributed across true classes.
- Rows = true labels
- Columns = predicted labels
- Diagonal = correct predictions

## Metrics for Arrhythmia Detection

For medical applications, consider:

1. **Sensitivity (Recall)**: Critical for detecting arrhythmias
   - High recall means fewer missed arrhythmias
   - Important for patient safety

2. **Specificity**: Important for Normal class
   - High specificity = fewer false alarms
   - Reduces unnecessary medical interventions

3. **Per-Class Performance**: Some arrhythmias are more critical
   - Focus on classes with higher clinical significance
   - Ensure rare but serious arrhythmias are detected

## Visualizations

### Confusion Matrix
- **Location**: `evaluation_results/confusion_matrix_test.png`
- **Shows**: How predictions are distributed across classes
- **Interpretation**:
  - Dark diagonal = good performance
  - Off-diagonal elements = confusion between classes

### Prediction Samples
- **Location**: `evaluation_results/prediction_samples.png`
- **Shows**: 10 ECG signals with predicted vs true labels
- **Color coding**:
  - Green border = correct prediction
  - Red border = incorrect prediction
- **Useful for**: Identifying patterns in misclassifications

## Workflow

### Complete Evaluation Workflow

```bash
# 1. Train model
python train_pytorch.py

# 2. Quick inference test
python test_model_inference.py

# 3. Comprehensive evaluation
python evaluate_model.py

# 4. Review results
# Check evaluation_results/ folder for:
#   - Metrics JSON file
#   - Confusion matrix plots
#   - Prediction samples
```

### After Evaluation

1. **Review confusion matrix**:
   - Identify which classes are confused
   - Check if confusion is clinically acceptable

2. **Check per-class metrics**:
   - Ensure critical arrhythmias have high recall
   - Verify Normal class has high specificity

3. **Examine prediction samples**:
   - Look at misclassified examples
   - Identify common failure patterns

4. **Compare validation vs test**:
   - Large gap suggests overfitting
   - Similar performance = good generalization

## Interpreting Results

### Good Performance Indicators

✅ **Accuracy > 80%**: Generally good for medical classification
✅ **Macro F1 > 0.75**: Balanced performance across classes
✅ **High recall for critical classes**: Fewer missed arrhythmias
✅ **High precision for Normal**: Fewer false alarms

### Warning Signs

⚠️ **Large validation-test gap**: Model may be overfitting
⚠️ **Low recall for rare classes**: Model struggling with imbalanced data
⚠️ **Confusion between similar classes**: May need better features
⚠️ **Very low precision for Normal**: Too many false positives

## Integration with Training

Both training scripts now include evaluation:

### PyTorch (`train_pytorch.py`)
- Evaluates during training
- Saves best model based on validation accuracy
- Prints metrics after training
- Suggests running `evaluate_model.py` for detailed analysis

### TensorFlow (`train_baseline.py`)
- Evaluates during training
- Saves metrics to JSON
- Prints confusion matrices
- Suggests running `evaluate_model.py` for visualizations

## Custom Evaluation

To evaluate on custom data:

```python
import numpy as np
import torch
from evaluate_model import load_pytorch_model, evaluate_pytorch, calculate_metrics

# Load model
model, device = load_pytorch_model("arhythmia_model.pth", device='cpu')

# Your custom data
X_custom = np.load("custom_X.npy")  # Shape: (N, 360)
y_custom = np.load("custom_y.npy")  # Shape: (N,)

# Evaluate
preds, probs = evaluate_pytorch(model, X_custom, y_custom, device)
metrics, cm = calculate_metrics(y_custom, preds, probs, "Custom Data")
```

## Troubleshooting

### "No trained model found"
- Make sure you've trained a model first
- Check that `arhythmia_model.pth` exists (PyTorch)
- Or `processed/baseline_cnn_weights.h5` exists (TensorFlow)

### "Module not found: seaborn"
```bash
pip install seaborn
```
Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Visualization errors
- Ensure matplotlib backend is set correctly
- For headless servers, use: `matplotlib.use('Agg')`

## Next Steps

After evaluation:
1. ✅ Review metrics and identify weak areas
2. ✅ Examine misclassifications in prediction samples
3. ✅ Consider retraining with different hyperparameters
4. ✅ Use model in production (`app.py` or custom inference)
5. ✅ Document performance for clinical validation

