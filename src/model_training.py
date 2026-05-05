"""Model fitting and metric evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute ROC-AUC for binary and multiclass settings."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_proba.ndim == 1 or y_proba.shape[1] == 1:
        return float(roc_auc_score(y_true, y_proba.ravel()))
    if y_proba.shape[1] == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))


def fit_extra_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    n_estimators: int = 200,
    max_depth: int | None = 4,
    min_samples_leaf: int = 1,
    max_features: str | float | int | None = "sqrt",
) -> ExtraTreesClassifier:
    """Fit the ExtraTrees baseline/prediction model."""
    clf = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(model: Any, X_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, float]:
    """Return AUC and accuracy for fitted classifier models."""
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)
    return {
        "auc": compute_auc(y_eval, y_proba),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
    }


def fit_prediction_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prediction_model_type: str,
    random_state: int,
    etree_params: Mapping[str, Any],
    device: torch.device,
    cache_dir: Path,
    model_dir: Path,
):
    """Fit a downstream model selected by prediction_model_type."""
    if prediction_model_type == "etree":
        etree_params_local = dict(etree_params)
        etree_params_local.setdefault("n_estimators", 200)
        etree_params_local.setdefault("max_depth", 4)
        return fit_extra_trees(X_train, y_train, random_state=random_state, **etree_params_local)

    if prediction_model_type == "tabiclv2":
        try:
            from tabicl import TabICLClassifier
        except ImportError as exc:
            raise ImportError("tabicl is not installed. Install it to use prediction_model_type='tabiclv2'.") from exc

        model = TabICLClassifier(
            disk_offload_dir=str(cache_dir),
            kv_cache=False,
            random_state=random_state,
            device=device,
            model_path=model_dir,
            verbose=True,
        )
        model.fit(X_train, y_train)
        return model

    raise ValueError(f"Unknown prediction_model_type: {prediction_model_type}")
