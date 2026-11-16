"""Download, split and standardise the hls4ml jet-tagging HLF dataset (OpenML 42468)."""
import argparse
import os

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(PROJECT_DIR, "data", "jets_hlf.npz")
CLASSES = ["g", "q", "t", "w", "z"]


def prepare(path=DEFAULT_PATH, seed=42, val_frac=0.1, test_frac=0.2):
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    ds = fetch_openml(data_id=42468, as_frame=False, parser="liac-arff",
                      data_home=os.path.join(PROJECT_DIR, "data", "openml_cache"))
    X = ds.data.astype(np.float32)
    labels = np.asarray(ds.target).astype(str)
    classes = sorted(np.unique(labels))
    y = np.searchsorted(classes, labels).astype(np.int64)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac / (1 - test_frac), random_state=seed, stratify=y_tmp)

    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(a).astype(np.float32) for a in (X_train, X_val, X_test))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test, classes=np.array(classes),
             feature_names=np.array(ds.feature_names))
    print(f"saved {path}: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}, classes {classes}")
    return path


def load(path=DEFAULT_PATH):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    prepare(a.out, a.seed)
