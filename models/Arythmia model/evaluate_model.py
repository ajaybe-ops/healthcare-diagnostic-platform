"""
evaluate_model.py - Comprehensive model evaluation script.

Evaluates a trained model on validation and test sets with detailed metrics,
confusion matrices, and visualization of predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import sys
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "processed"

# Model paths
PYTORCH_MODEL_PATH = BASE_DIR / "arhythmia_model.pth"
TF_MODEL_PATH = DATA_DIR / "baseline_cnn_weights.h5"

ARRHYTHMIA_CLASSES = {
    0: "Normal",
    1: "Left bundle branch block",
    2: "Right bundle branch block",
    3: "Atrial premature",
    4: "Premature ventricular",
    5: "Fusion of ventricular and normal",
    6: "Nodal escape"
}

def load_pytorch_model(model_path, device='cpu'):
    """Load PyTorch model."""
    class ArrhythmiaCNN(nn.Module):
        def __init__(self, num_classes=7):
            super(ArrhythmiaCNN, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            x = x.squeeze(-1)
            x = self.fc(x)
            return x
    
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', 7)
    model = ArrhythmiaCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, num_classes

def load_tensorflow_model(model_path):
    """Load TensorFlow model."""
    try:
        import tensorflow as tf
        
        # Rebuild model architecture
        def build_cnn(input_shape, num_classes):
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
            x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(x)
            x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            return tf.keras.Model(inputs, outputs)
        
        model = build_cnn((360, 1), 7)
        model.load_weights(str(model_path))
        return model, 7
    except ImportError:
        return None, None

def evaluate_pytorch(model, X, y, device='cpu', batch_size=64):
    """Evaluate PyTorch model."""
    model.eval()
    all_preds = []
    all_probs = []
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X).unsqueeze(1).to(device),
        torch.LongTensor(y).to(device)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for X_batch, _ in dataloader:
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs)

def evaluate_tensorflow(model, X, y, batch_size=64):
    """Evaluate TensorFlow model."""
    X_input = X[..., np.newaxis] if X.ndim == 2 else X
    probs = model.predict(X_input, batch_size=batch_size, verbose=0)
    preds = np.argmax(probs, axis=1)
    return preds, probs

def calculate_metrics(y_true, y_pred, y_proba=None, split_name="Evaluation"):
    """Calculate comprehensive metrics."""
    metrics = {}
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    metrics['accuracy'] = float(accuracy)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(7), zero_division=0
    )
    
    metrics['per_class'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist()
    }
    
    # Macro averages
    metrics['macro'] = {
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1': float(np.mean(f1))
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(7))
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=range(7), 
                                   target_names=[ARRHYTHMIA_CLASSES[i] for i in range(7)],
                                   output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} METRICS")
    print(f"{'='*60}")
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nMacro Averages:")
    print(f"  Precision: {metrics['macro']['precision']:.4f}")
    print(f"  Recall:    {metrics['macro']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['macro']['f1']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 76)
    for i in range(7):
        class_name = ARRHYTHMIA_CLASSES[i]
        print(f"{class_name:<30} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]:<10}")
    
    return metrics, cm

def plot_confusion_matrix(cm, save_path=None, split_name="Evaluation"):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    class_names = [ARRHYTHMIA_CLASSES[i] for i in range(7)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {split_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def plot_predictions(X, y_true, y_pred, y_proba, indices=None, save_dir=None, num_samples=10):
    """Plot ECG signals with predicted vs true labels."""
    if indices is None:
        # Randomly sample indices
        np.random.seed(42)
        indices = np.random.choice(len(X), size=min(num_samples, len(X)), replace=False)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, ax in zip(indices[:num_samples], axes):
        signal = X[idx]
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        confidence = y_proba[idx][pred_label]
        
        ax.plot(signal, 'b-', linewidth=1)
        ax.set_title(
            f'Sample {idx}: True={ARRHYTHMIA_CLASSES[true_label]} | '
            f'Predicted={ARRHYTHMIA_CLASSES[pred_label]} | '
            f'Confidence={confidence:.3f}',
            fontsize=10
        )
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Color code: green for correct, red for incorrect
        if true_label == pred_label:
            ax.spines['top'].set_color('green')
            ax.spines['top'].set_linewidth(2)
        else:
            ax.spines['top'].set_color('red')
            ax.spines['top'].set_linewidth(2)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'prediction_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Prediction samples saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    # Detect model type
    use_pytorch = PYTORCH_MODEL_PATH.exists()
    use_tensorflow = TF_MODEL_PATH.exists()
    
    if not (use_pytorch or use_tensorflow):
        print("[ERROR] No trained model found!")
        print(f"  Expected PyTorch model at: {PYTORCH_MODEL_PATH}")
        print(f"  Expected TensorFlow weights at: {TF_MODEL_PATH}")
        print("\nPlease train a model first:")
        print("  python train_pytorch.py  (for PyTorch)")
        print("  python train_baseline.py (for TensorFlow)")
        sys.exit(1)
    
    # Load data
    print("Loading data...")
    X = np.load(DATA_DIR / "X.npy", mmap_mode='r')
    y = np.load(DATA_DIR / "y.npy", mmap_mode='r')
    val_idx = np.load(DATA_DIR / "val_idx.npy")
    test_idx = np.load(DATA_DIR / "test_idx.npy")
    
    X_val = X[val_idx].astype(np.float32)
    y_val = y[val_idx]
    X_test = X[test_idx].astype(np.float32)
    y_test = y[test_idx]
    
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Load model
    if use_pytorch:
        print(f"\nLoading PyTorch model from: {PYTORCH_MODEL_PATH}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, num_classes = load_pytorch_model(PYTORCH_MODEL_PATH, device)
        print(f"Using device: {device}")
        
        # Evaluate
        print("\nEvaluating on validation set...")
        val_preds, val_probs = evaluate_pytorch(model, X_val, y_val, device)
        
        print("\nEvaluating on test set...")
        test_preds, test_probs = evaluate_pytorch(model, X_test, y_test, device)
    else:
        print(f"\nLoading TensorFlow model from: {TF_MODEL_PATH}")
        model, num_classes = load_tensorflow_model(TF_MODEL_PATH)
        if model is None:
            print("[ERROR] Failed to load TensorFlow model")
            sys.exit(1)
        
        # Evaluate
        print("\nEvaluating on validation set...")
        val_preds, val_probs = evaluate_tensorflow(model, X_val, y_val)
        
        print("\nEvaluating on test set...")
        test_preds, test_probs = evaluate_tensorflow(model, X_test, y_test)
    
    # Calculate metrics
    val_metrics, val_cm = calculate_metrics(y_val, val_preds, val_probs, "Validation")
    test_metrics, test_cm = calculate_metrics(y_test, test_preds, test_probs, "Test")
    
    # Save metrics
    results_dir = BASE_DIR / "evaluation_results"
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'validation': val_metrics,
        'test': test_metrics
    }
    
    results_path = results_dir / "evaluation_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Metrics saved to: {results_path}")
    
    # Plot confusion matrices
    plot_confusion_matrix(val_cm, results_dir / "confusion_matrix_validation.png", "Validation")
    plot_confusion_matrix(test_cm, results_dir / "confusion_matrix_test.png", "Test")
    
    # Plot prediction samples
    print("\nGenerating prediction visualizations...")
    plot_predictions(X_test, y_test, test_preds, test_probs, 
                     save_dir=results_dir, num_samples=10)
    
    print(f"\n{'='*60}")
    print("[SUCCESS] Evaluation complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_dir}")
    print(f"  - evaluation_metrics.json")
    print(f"  - confusion_matrix_validation.png")
    print(f"  - confusion_matrix_test.png")
    print(f"  - prediction_samples.png")

if __name__ == "__main__":
    main()

