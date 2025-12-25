import json
import numpy as np
from pathlib import Path
import sys
import logging

DATA_DIR = Path("C:/Users/Lab/OneDrive/Desktop/Arythmia model/processed/")
METRICS_PATH = DATA_DIR / "baseline_cnn_metrics.json"
AUDIT_PATH = DATA_DIR / "metrics_audit_report.json"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def abort(msg):
    logging.error("AUDIT FAILURE: " + msg)
    sys.exit(1)

def safe_load_metrics():
    if not METRICS_PATH.exists():
        abort(f"Missing metrics file: {METRICS_PATH}")
    with open(METRICS_PATH) as f:
        m = json.load(f)
    return m

def macro_f1_vs_accuracy(m):
    acc = m['test']['accuracy']
    macro_f1 = np.mean(m['test']['f1'])
    return abs(acc - macro_f1)

# ============== MAIN AUDIT ==============
def main():
    metrics = safe_load_metrics()
    test = metrics['test']

    passed = True
    explanations = []

    # Class-wise audit
    recalls = np.array(test['recall'])
    fails_lowrecall = np.where(recalls < 0.40)[0].tolist()
    if fails_lowrecall:
        passed = False
        for c in fails_lowrecall:
            explanations.append(f"Recall < 0.40 for class {c}: {recalls[c]:.3f}")

    # Macro vs accuracy
    acc = test['accuracy']
    macro_f1 = float(np.mean(test['f1']))
    weighted_f1 = float(np.average(test['f1'], weights=test['support']))
    if acc - macro_f1 > 0.10:
        passed = False
        explanations.append(f"Accuracy ({acc:.3f}) much higher than Macro F1 ({macro_f1:.3f}): likely class imbalance.")

    # Report
    print("================ METRICS AUDIT ================")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    if fails_lowrecall:
        print(f"UNSAFE: Recall < 0.40 for class(es): {fails_lowrecall}")
    if acc - macro_f1 > 0.10:
        print(f"UNSAFE: Accuracy is inflated relative to Macro F1 (>0.10 difference)")
    if passed:
        print("PASS: No major metric hazards detected.")
    else:
        print("FAIL: At least one metric hazard detected.")
    print("===============================================")

    # Save audit report
    result = {
        'pass': passed,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'low_recall_classes': fails_lowrecall,
        'bad_accuracy_vs_f1': acc - macro_f1 > 0.10,
        'explanations': explanations
    }
    with open(AUDIT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    if not passed:
        abort("Audit failed: Unsafe metric condition(s). See above.")

if __name__ == "__main__":
    main()

