import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "processed"
MODEL_DIR = BASE_DIR
SCHEMA_PATH = BASE_DIR / "dataset_schema.json"

MODEL_PATH = MODEL_DIR / "arhythmia_model.pth"
METRICS_PATH = DATA_DIR / "baseline_cnn_metrics.json"

INPUT_SHAPE = (1, 360)
EPOCHS = 12
BATCH_SIZE = 64
SEED = 42
LEARNING_RATE = 0.001

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def abort(msg):
    print(f"CRITICAL ERROR: {msg}")
    sys.exit(1)

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ArrhythmiaCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(ArrhythmiaCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device, class_weights=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def evaluate_model(model, test_loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(num_classes), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    return {
        'accuracy': float(acc),
        'precision': prec.tolist(),
        'recall': rec.tolist(),
        'f1': f1.tolist(),
        'support': sup.tolist(),
        'confusion_matrix': cm.tolist()
    }

def main():
    set_seed(SEED)
    
    if not DATA_DIR.exists():
        abort(f"Data directory not found: {DATA_DIR}")
    
    X_path = DATA_DIR / "X.npy"
    y_path = DATA_DIR / "y.npy"
    train_idx_path = DATA_DIR / "train_idx.npy"
    val_idx_path = DATA_DIR / "val_idx.npy"
    test_idx_path = DATA_DIR / "test_idx.npy"
    
    if not all([p.exists() for p in [X_path, y_path, train_idx_path, val_idx_path, test_idx_path]]):
        abort(f"Required data files not found in {DATA_DIR}")
    
    print("Loading data...")
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)
    
    X_train = X[train_idx].astype(np.float32)
    y_train = y[train_idx]
    X_val = X[val_idx].astype(np.float32)
    y_val = y[val_idx]
    X_test = X[test_idx].astype(np.float32)
    y_test = y[test_idx]
    
    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ArrhythmiaCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting training...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'input_shape': INPUT_SHAPE,
                'epoch': epoch,
                'val_acc': val_acc
            }, MODEL_PATH)
            print(f"  -> Saved model with validation accuracy: {val_acc:.2f}%")
    
    print(f"\nLoading best model for evaluation...")
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device, num_classes)
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, num_classes)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    metrics = {
        'val': val_metrics,
        'test': test_metrics
    }
    
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] Model saved to: {MODEL_PATH}")
    print(f"[OK] Metrics saved to: {METRICS_PATH}")
    print(f"\nTo evaluate the model in detail, run:")
    print(f"  python evaluate_model.py")

if __name__ == "__main__":
    main()

