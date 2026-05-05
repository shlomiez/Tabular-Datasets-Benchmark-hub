"""Dataset loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder, StandardScaler


def build_dataset_paths(data_root: Path) -> dict[str, Path]:
    """Build canonical paths for the benchmark datasets."""
    return {
        "Breast": data_root / "Challenging Data Collections" / "Unaltered Data Sources" / "Breast.mat",
        "madelon": data_root / "Challenging Data Collections" / "Unaltered Data Sources" / "madelon.mat",
        "SMK-CAN-187": data_root / "Standard Difficulty Data Collections" / "scikit-feature" / "SMK-CAN-187.mat",
        "colon": data_root / "Standard Difficulty Data Collections" / "scikit-feature" / "colon.mat",
        "leukemia": data_root / "Standard Difficulty Data Collections" / "scikit-feature" / "leukemia.mat",
        "RELATHE": data_root / "Standard Difficulty Data Collections" / "scikit-feature" / "RELATHE.mat",
    }


def dataset_path_report(dataset_paths: dict[str, Path]) -> pd.DataFrame:
    """Return a small status table for path existence checks."""
    return pd.DataFrame(
        [{"dataset": name, "path": str(path), "exists": path.exists()} for name, path in dataset_paths.items()]
    )


def _load_mat_xy(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(mat_path)

    feature_keys = ("X", "x", "data", "features", "fea")
    label_keys = ("y", "Y", "labels", "label", "target", "gnd")

    X = next((mat[k] for k in feature_keys if k in mat), None)
    y = next((mat[k] for k in label_keys if k in mat), None)

    if X is None or y is None:
        raise KeyError(f"Could not find feature/label keys in {mat_path.name}")

    return np.asarray(X), np.asarray(y).reshape(-1)


def load_dataset_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset from disk, currently supporting .mat files."""
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Unsupported dataset format: {path.name}. Only .mat files are supported.")
    return _load_mat_xy(path)


def load_available_datasets(dataset_paths: dict[str, Path]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Load all available datasets and return missing names as a second output."""
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing: list[str] = []

    for dataset_name, dataset_path in dataset_paths.items():
        if dataset_path.exists():
            loaded[dataset_name] = load_dataset_xy(dataset_path)
        else:
            missing.append(dataset_name)

    return loaded, missing


def _as_numpy_array(value: Any) -> np.ndarray:
    if sp.issparse(value):
        return value.toarray()
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


def extract_xy(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) from tuple/list/mapping/object dataset inputs."""
    if isinstance(dataset, (tuple, list)):
        if len(dataset) != 2:
            raise ValueError("Dataset tuple/list must have exactly (X, y).")
        return _as_numpy_array(dataset[0]), np.asarray(dataset[1])

    if isinstance(dataset, dict):
        feature_keys = ("X", "x", "data", "features")
        label_keys = ("y", "Y", "labels", "target", "targets")
        X = next((dataset[k] for k in feature_keys if k in dataset), None)
        y = next((dataset[k] for k in label_keys if k in dataset), None)
        if X is None or y is None:
            raise ValueError("Dataset mapping must include feature and label entries.")
        return _as_numpy_array(X), np.asarray(y)

    for feature_attr, label_attr in (("data", "target"), ("X", "y"), ("features", "labels")):
        if hasattr(dataset, feature_attr) and hasattr(dataset, label_attr):
            return _as_numpy_array(getattr(dataset, feature_attr)), np.asarray(getattr(dataset, label_attr))

    raise TypeError("Unsupported dataset format.")


def encode_labels(y: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """Encode labels as consecutive integer classes."""
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(np.asarray(y).ravel())
    return y_encoded, encoder


def prepare_fold_arrays(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale train/test arrays and convert sparse outputs into dense arrays."""
    scaler = StandardScaler(with_mean=not sp.issparse(X_train))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if sp.issparse(X_train_scaled):
        X_train_scaled = X_train_scaled.toarray()
    if sp.issparse(X_test_scaled):
        X_test_scaled = X_test_scaled.toarray()

    return np.asarray(X_train_scaled, dtype=np.float32), np.asarray(X_test_scaled, dtype=np.float32)
