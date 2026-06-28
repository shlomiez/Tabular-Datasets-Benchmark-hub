"""STG/LSPIN feature selection routines and helper utilities."""

from __future__ import annotations

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

for featselectlib_path in (
    project_root / "project-featselectlib",
    project_root.parent / "project-featselectlib",
):
    if featselectlib_path.exists() and str(featselectlib_path) not in sys.path:
        sys.path.insert(0, str(featselectlib_path))
        break

from typing import Any, Sequence

import numpy as np
import torch


LAMBDA_SEQUENCE_DEFAULT = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0)


def build_feature_ratios() -> list[float]:
    """Build default feature-ratio sequence used in ratio mode."""
    ratios = [1.0] + [round(v, 2) for v in np.arange(0.9, 0.0, -0.1)] + [0.05]
    return sorted(set(ratios), reverse=True)


def build_lambda_sequence(
    initial_lam: float = 0.1,
    multipliers: Sequence[float] = (5.0, 2.0, 5.0, 2.0, 5.0, 2.0),
) -> list[float]:
    """Build log-like lambda sequence."""
    values = [float(initial_lam)]
    current = float(initial_lam)
    for multiplier in multipliers:
        current *= float(multiplier)
        values.append(float(current))
    return values


def ratio_to_k(p: int, ratio: float) -> int:
    """Convert ratio to feature count with minimum of one selected feature."""
    return max(1, int(round(ratio * p)))


def _select_from_scores(
    scores: np.ndarray,
    feature_selection_method: str,
    k: int | None,
) -> tuple[np.ndarray, int]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    active_indices = np.where(scores > 0)[0]
    active_count = int(active_indices.size)

    if feature_selection_method == "features_ratio":
        if k is None:
            raise ValueError("k must be provided when feature_selection_method='features_ratio'.")
        selected_indices = np.argsort(scores)[-k:][::-1]
        return selected_indices, active_count

    if feature_selection_method == "lamda_tuning":
        if active_count > 0:
            return active_indices, active_count
        fallback = np.array([int(np.argmax(scores))], dtype=np.int64)
        return fallback, 0

    raise ValueError(f"Unknown feature_selection_method: {feature_selection_method}")


def _effective_lspin_batch_size(n_samples: int, requested_batch_size: int, batch_normalization: bool) -> int:
    if requested_batch_size <= 0:
        requested_batch_size = n_samples

    batch_size = int(min(requested_batch_size, n_samples))
    if not batch_normalization:
        return max(1, batch_size)

    if n_samples <= 1:
        return 1

    if batch_size <= 1:
        return 2

    if n_samples % batch_size == 1:
        for candidate in range(batch_size + 1, n_samples + 1):
            if n_samples % candidate != 1:
                return candidate
    return batch_size


def fit_stg_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int | None,
    random_state: int,
    feature_selection_method: str = "features_ratio",
    epochs: int = 50,
    learning_rate: float = 1e-4,
    hidden_dims: Sequence[int] = (128, 64),
    sigma: float = 0.5,
    lam: float = 0.1,
    batch_size: int = 64,
    optimizer: str = "SGD",
    activation: str = "tanh",
    early_stopping_patience: int = 25,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_min_epochs: int = 50,
) -> tuple[np.ndarray, list[float], int]:
    """Train STG and return selected indices, loss history, and active gate count."""
    try:
        from featselectlib.supervised_feature_selection.stg.python.stg import STG
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'featselectlib'.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    selector = STG(
        device=device,
        input_dim=n_features,
        output_dim=n_classes,
        hidden_dims=list(hidden_dims),
        activation=activation,
        sigma=sigma,
        lam=lam,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        task_type="classification",
        random_state=random_state,
    )

    train_loader = selector.get_dataloader(X_train, y_train, shuffle=True)
    train_losses: list[float] = []
    best_loss = float("inf")
    bad_epochs = 0

    for epoch_index in range(epochs):
        epoch_meters = selector.train_epoch(train_loader)
        epoch_loss = None
        if hasattr(epoch_meters, "avg") and epoch_meters.avg is not None:
            epoch_loss = epoch_meters.avg.get("loss")

        if epoch_loss is None:
            continue

        current_loss = float(epoch_loss)
        train_losses.append(current_loss)
        if current_loss < best_loss - early_stopping_min_delta:
            best_loss = current_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (epoch_index + 1) >= early_stopping_min_epochs and bad_epochs >= early_stopping_patience:
            print(f"STG early stopping at epoch {epoch_index + 1} (best loss={best_loss:.6f})")
            break

    gate_values = selector.get_gates("prob")
    if isinstance(gate_values, torch.Tensor):
        gate_values = gate_values.detach().cpu().numpy()

    gate_array = np.asarray(gate_values, dtype=np.float32).flatten()
    active_count = int(np.sum(gate_array > 0))

    selected_indices, _ = _select_from_scores(gate_array, feature_selection_method, k)
    return selected_indices, train_losses, active_count


def fit_lspin_selector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int | None,
    random_state: int,
    feature_selection_method: str = "features_ratio",
    epochs: int = 50,
    learning_rate: float = 0.1,
    hidden_layers: Sequence[int] = (128, 64),
    gating_hidden_layers: Sequence[int] = (64,),
    sigma: float = 0.5,
    lam: float = 0.5,
    batch_size: int = 64,
    batch_normalization: bool = True,
    activation_gating: str = "tanh",
    activation_pred: str = "tanh",
    optimizer: str = "SGD",
    early_stopping_patience: int = 25,
    early_stopping_min_delta: float = 1e-4,
    early_stopping_min_epochs: int = 50,
    **_: Any,
) -> tuple[np.ndarray, list[float], float, Any]:
    """Train LSPIN and return selected indices, loss history, mean active gates, and model object."""
    try:
        from featselectlib.supervised_feature_selection.Lspin import DataSetMeta, Lspin
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'featselectlib'.") from exc

    n_features = X_train.shape[1]
    labels = np.asarray(y_train).ravel()
    classes = np.unique(labels)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    y_index = np.array([class_to_index[label] for label in labels], dtype=np.int64)
    y_one_hot = np.eye(len(classes), dtype=np.float32)[y_index]
    metadata = np.zeros((X_train.shape[0], 1), dtype=np.float32)

    train_dataset = DataSetMeta(np.asarray(X_train, dtype=np.float32), y_one_hot, metadata)
    model = Lspin(
        input_node=n_features,
        hidden_layers_node=list(hidden_layers),
        output_node=len(classes),
        gating_net_hidden_layers_node=list(gating_hidden_layers),
        display_step=epochs + 1,
        activation_gating=activation_gating,
        activation_pred=activation_pred,
        feature_selection=True,
        batch_normalization=batch_normalization,
        sigma=sigma,
        lam=lam,
        seed=random_state,
        val=False,
    )

    effective_batch_size = _effective_lspin_batch_size(X_train.shape[0], int(batch_size), bool(batch_normalization))

    train_losses, _, _ = model.train_model(
        train_dataset,
        batch_size=effective_batch_size,
        num_epoch=epochs,
        lr=learning_rate,
        compute_sim=False,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_restore_best=True,
        early_stopping_min_epochs=early_stopping_min_epochs,
    )

    gate_matrix = model.get_prob_alpha(torch.as_tensor(X_train, dtype=torch.float32, device=model.device))
    if isinstance(gate_matrix, torch.Tensor):
        gate_matrix = gate_matrix.detach().cpu().numpy()

    if gate_matrix.ndim == 1:
        global_scores = gate_matrix.reshape(-1)
        gate_matrix_2d = gate_matrix.reshape(1, -1)
    else:
        global_scores = gate_matrix.mean(axis=0)
        gate_matrix_2d = gate_matrix

    active_counts_per_sample = np.sum(gate_matrix_2d > 0, axis=1)
    average_active_count = float(np.mean(active_counts_per_sample))

    selected_indices, _ = _select_from_scores(global_scores, feature_selection_method, k)
    return selected_indices, train_losses, average_active_count, model


def fit_concrete_selector(
    X_train: np.ndarray,
    k: int,
    num_epochs: int = 100,
    random_state: int | None = None,
    decoder_factory: Any | None = None,
) -> tuple[np.ndarray, list[float], Any]:
    """Fit Concrete Autoencoder and return selected indices, loss history, and selector object.

    Decoder factory should be a callable that accepts input_dim and returns a Keras layer/function.
    """
    try:
        from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'concrete_autoencoder'.") from exc

    # Default simple linear decoder if none provided
    if decoder_factory is None:
        try:
            from keras.layers import Dense

            def decoder_factory_inner(input_dim: int):
                return lambda x: Dense(input_dim)(x)

            decoder_factory = decoder_factory_inner
        except Exception:
            decoder_factory = None

    input_dim = X_train.shape[1]
    output_fn = decoder_factory(input_dim) if decoder_factory is not None else None

    selector = ConcreteAutoencoderFeatureSelector(K=k, output_function=output_fn, num_epochs=num_epochs, tryout_limit=1)

    # Fit on X_train (reconstruct X)
    selector.fit(X_train, X_train)

    history = getattr(getattr(selector, "model", None), "history", None)
    train_losses: list[float] = []
    if history is not None and hasattr(history, "history") and "loss" in history.history:
        train_losses = list(history.history["loss"])

    indices = selector.get_indices()
    return np.asarray(indices, dtype=np.int64), train_losses, selector
