import numpy as np
import sys
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path("C:/Users/Lab/OneDrive/Desktop/Arythmia model/processed/")
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
ID_PATH = DATA_DIR / "record_ids.npy"

TRAIN_IDX_PATH = DATA_DIR / "train_idx.npy"
VAL_IDX_PATH = DATA_DIR / "val_idx.npy"
TEST_IDX_PATH = DATA_DIR / "test_idx.npy"

def abort(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)

def main():
    # --- Load files ---
    for p in [X_PATH, Y_PATH, ID_PATH]:
        if not p.exists():
            abort(f"Missing required file: {p}")
    X = np.load(X_PATH, mmap_mode='r')
    y = np.load(Y_PATH, mmap_mode='r')
    record_ids = np.load(ID_PATH, mmap_mode='r')
    if not (len(X) == len(y) == len(record_ids)):
        abort(f"X, y, record_ids lengths do not match! {len(X)}, {len(y)}, {len(record_ids)}")

    # --- Unique patient/record IDs ---
    patients = np.array(sorted(set(record_ids)))
    rng = np.random.RandomState(SEED)
    rng.shuffle(patients)
    n_total = len(patients)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)

    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    # --- Assign indices by patient ---
    idxs = {'train': [], 'val': [], 'test': []}
    for i, pid in enumerate(record_ids):
        if pid in train_p:
            idxs['train'].append(i)
        elif pid in val_p:
            idxs['val'].append(i)
        elif pid in test_p:
            idxs['test'].append(i)
        else:
            abort(f"Patient {pid} not assigned to any split!")

    # --- Assert no leakage ---
    set_inter = set(record_ids[idxs['train']]) & set(record_ids[idxs['val']])
    if set_inter:
        abort(f"Patient(s) {set_inter} leaked between train and val!")
    set_inter = set(record_ids[idxs['train']]) & set(record_ids[idxs['test']])
    if set_inter:
        abort(f"Patient(s) {set_inter} leaked between train and test!")
    set_inter = set(record_ids[idxs['val']]) & set(record_ids[idxs['test']])
    if set_inter:
        abort(f"Patient(s) {set_inter} leaked between val and test!")

    # --- Save indices ---
    np.save(TRAIN_IDX_PATH, np.array(idxs['train'], dtype=np.int64))
    np.save(VAL_IDX_PATH, np.array(idxs['val'], dtype=np.int64))
    np.save(TEST_IDX_PATH, np.array(idxs['test'], dtype=np.int64))

    # --- Print stats ---
    def split_stats(name, idx):
        pids = set(record_ids[idx])
        cnts = Counter(y[idx])
        print(f"{name}: {len(idx)} beats, {len(pids)} patients")
        print(f"  Class dist:", dict(cnts))
    print("Summary of splits:")
    split_stats("Train", idxs['train'])
    split_stats("Val", idxs['val'])
    split_stats("Test", idxs['test'])

if __name__ == "__main__":
    main()

