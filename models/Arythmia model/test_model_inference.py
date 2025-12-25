"""
test_model_inference.py - Quick test script to verify model can make predictions.

Tests model loading and inference on a single sample from the test set.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "processed"
MODEL_PATH = BASE_DIR / "arhythmia_model.pth"

ARRHYTHMIA_CLASSES = {
    0: "Normal",
    1: "Left bundle branch block",
    2: "Right bundle branch block",
    3: "Atrial premature",
    4: "Premature ventricular",
    5: "Fusion of ventricular and normal",
    6: "Nodal escape"
}

def load_model():
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
    
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        print("Please train the model first: python train_pytorch.py")
        return None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    num_classes = checkpoint.get('num_classes', 7)
    model = ArrhythmiaCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

def test_inference():
    """Test model inference on a single sample."""
    print("=" * 60)
    print("Model Inference Test")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    model, device = load_model()
    if model is None:
        return False
    
    print(f"   [OK] Model loaded on device: {device}")
    
    # Load test data
    print("\n2. Loading test data...")
    test_idx_path = DATA_DIR / "test_idx.npy"
    if not test_idx_path.exists():
        print(f"   [ERROR] Test indices not found: {test_idx_path}")
        print("   Please run: python split.py")
        return False
    
    X = np.load(DATA_DIR / "X.npy", mmap_mode='r')
    y = np.load(DATA_DIR / "y.npy", mmap_mode='r')
    test_idx = np.load(test_idx_path)
    
    if len(test_idx) == 0:
        print("   [ERROR] Test set is empty")
        return False
    
    # Get a single sample
    sample_idx = test_idx[0]
    X_sample = X[sample_idx].astype(np.float32)
    y_true = y[sample_idx]
    
    print(f"   [OK] Loaded sample {sample_idx}")
    print(f"   [OK] True label: {y_true} ({ARRHYTHMIA_CLASSES[y_true]})")
    
    # Make prediction
    print("\n3. Making prediction...")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_sample).unsqueeze(0).unsqueeze(1).to(device)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(outputs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    print(f"   [OK] Prediction complete")
    print(f"\n4. Results:")
    print(f"   True Label:  {y_true} - {ARRHYTHMIA_CLASSES[y_true]}")
    print(f"   Predicted:   {pred_class} - {ARRHYTHMIA_CLASSES[pred_class]}")
    print(f"   Confidence:  {confidence:.4f} ({confidence*100:.2f}%)")
    
    if pred_class == y_true:
        print(f"\n   [CORRECT] Prediction matches true label!")
    else:
        print(f"\n   [INCORRECT] Prediction does not match true label")
    
    # Show all class probabilities
    print(f"\n5. All class probabilities:")
    probs_np = probs[0].cpu().numpy()
    for i in range(7):
        marker = " <-- PREDICTED" if i == pred_class else ""
        print(f"   Class {i} ({ARRHYTHMIA_CLASSES[i]:<30}): {probs_np[i]:.4f} ({probs_np[i]*100:.2f}%){marker}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Model inference test passed!")
    print("=" * 60)
    print("\nTo evaluate on full test set, run:")
    print("  python evaluate_model.py")
    
    return True

if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)

