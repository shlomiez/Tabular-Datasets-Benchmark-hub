"""Cross-validation experiment engine for baseline, STG, and LSPIN."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from src.data_preprocessing import encode_labels, extract_xy, prepare_fold_arrays
from src.feature_selection import fit_lspin_selector, fit_stg_selector, ratio_to_k
from src.model_training import evaluate_classifier, fit_prediction_model
from src.utils import set_global_seed


def checkpoint_key(dataset_name: str, feature_selection_method: str, selection_value: str, fold_index: int) -> str:
    """Build stable key for checkpointed fold jobs."""
    return f"{dataset_name}|{feature_selection_method}|{selection_value}|{fold_index}"


def get_checkpoint_path(output_dir: Path, dataset_name: str) -> Path:
    """Return per-dataset checkpoint path."""
    return output_dir / f"checkpoint_{dataset_name}.pkl"


def load_checkpoint(output_dir: Path, dataset_name: str) -> pd.DataFrame:
    """Load checkpoint if available."""
    checkpoint_path = get_checkpoint_path(output_dir, dataset_name)
    if checkpoint_path.exists():
        return pd.read_pickle(checkpoint_path)
    return pd.DataFrame()


def save_checkpoint(output_dir: Path, dataset_name: str, fold_df: pd.DataFrame) -> None:
    """Persist fold-level checkpoint."""
    checkpoint_path = get_checkpoint_path(output_dir, dataset_name)
    fold_df.to_pickle(checkpoint_path)


def expand_loss_histories(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Transform fold loss lists into long-form per-epoch records."""
    rows = []
    for row in fold_df.itertuples(index=False):
        common = {
            "dataset": row.dataset,
            "fold": int(row.fold),
            "feature_selection_method": getattr(row, "feature_selection_method", "features_ratio"),
            "selection_value": str(getattr(row, "selection_value", "")),
            "stg_feature_ratio": float(getattr(row, "stg_feature_ratio", np.nan)),
            "lspin_feature_ratio": float(getattr(row, "lspin_feature_ratio", np.nan)),
            "lambda_value": float(getattr(row, "lambda_value", np.nan)),
            "lspin_lambda_value": float(getattr(row, "lspin_lambda_value", np.nan)),
        }

        stg_losses = list(getattr(row, "stg_train_loss_history", []) or [])
        lspin_losses = list(getattr(row, "lspin_train_loss_history", []) or [])

        for epoch_index, loss_value in enumerate(stg_losses, start=1):
            rows.append({**common, "algorithm": "STG", "epoch": epoch_index, "train_loss": float(loss_value)})
        for epoch_index, loss_value in enumerate(lspin_losses, start=1):
            rows.append({**common, "algorithm": "LSPIN", "epoch": epoch_index, "train_loss": float(loss_value)})

    return pd.DataFrame(rows)


def summarize_fold_results(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold metrics into a summary table."""
    rows: list[dict[str, Any]] = []
    group_cols = [
        "dataset",
        "feature_selection_method",
        "selection_value",
        "lambda_value",
        "lspin_lambda_value",
        "p",
    ]

    for keys, part in fold_df.groupby(group_cols, dropna=False):
        (
            dataset_name,
            feature_selection_method,
            selection_value,
            lambda_value,
            lspin_lambda_value,
            p,
        ) = keys

        stg_k_mean = part["stg_k"].mean() if "stg_k" in part.columns else np.nan
        stg_k_value = int(round(float(stg_k_mean))) if pd.notna(stg_k_mean) else np.nan
        stg_feature_ratio_value = float(stg_k_value / int(p)) if pd.notna(stg_k_value) and int(p) > 0 else np.nan

        lspin_k_mean = part["lspin_k"].mean() if "lspin_k" in part.columns else np.nan
        lspin_k_value = int(round(float(lspin_k_mean))) if pd.notna(lspin_k_mean) else np.nan
        lspin_feature_ratio_value = float(lspin_k_value / int(p)) if pd.notna(lspin_k_value) and int(p) > 0 else np.nan

        rows.append(
            {
                "dataset": dataset_name,
                "feature_selection_method": feature_selection_method,
                "selection_value": selection_value,
                "stg_feature_ratio": stg_feature_ratio_value,
                "lspin_feature_ratio": lspin_feature_ratio_value,
                "lambda_value": float(lambda_value) if pd.notna(lambda_value) else np.nan,
                "lspin_lambda_value": float(lspin_lambda_value) if pd.notna(lspin_lambda_value) else np.nan,
                "p": int(p),
                "stg_k": stg_k_value,
                "lspin_k": lspin_k_value,
                "stg_selected_features_mean": part["stg_selected_features"].mean(),
                "lspin_selected_features_mean": part["lspin_selected_features"].mean(),
                "stg_ignored_features_mean": part["stg_ignored_features"].mean(),
                "lspin_ignored_features_mean": part["lspin_ignored_features"].mean(),
                "baseline_train_auc_mean": part["baseline_train_auc"].mean(),
                "stg_train_auc_mean": part["stg_train_auc"].mean(),
                "lspin_train_auc_mean": part["lspin_train_auc"].mean(),
                "baseline_auc_mean": part["baseline_auc"].mean(),
                "stg_auc_mean": part["stg_auc"].mean(),
                "lspin_auc_mean": part["lspin_auc"].mean(),
                "stg_auc_delta_mean": part["stg_auc_delta"].mean(),
                "lspin_auc_delta_mean": part["lspin_auc_delta"].mean(),
                "baseline_train_accuracy_mean": part["baseline_train_accuracy"].mean(),
                "stg_train_accuracy_mean": part["stg_train_accuracy"].mean(),
                "lspin_train_accuracy_mean": part["lspin_train_accuracy"].mean(),
                "baseline_accuracy_mean": part["baseline_accuracy"].mean(),
                "stg_accuracy_mean": part["stg_accuracy"].mean(),
                "lspin_accuracy_mean": part["lspin_accuracy"].mean(),
                "stg_accuracy_delta_mean": part["stg_accuracy_delta"].mean(),
                "lspin_accuracy_delta_mean": part["lspin_accuracy_delta"].mean(),
            }
        )

    sort_cols = ["dataset", "feature_selection_method", "selection_value"]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def run_fold_experiment(
    dataset_name: str,
    fold_index: int,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    stg_params: Mapping[str, Any],
    lspin_params: Mapping[str, Any],
    etree_params: Mapping[str, Any],
    feature_selection_method: str,
    selection_value: Any,
    device: torch.device,
    cache_dir: Path,
    model_dir: Path,
    prediction_model_type: str = "etree",
    k: int | None = None,
    lambda_value: float | None = None,
    lspin_lambda_value: float | None = None,
    run_stg: bool = True,
    run_lspin: bool = True,
    evaluation_mode: str = "full",
) -> dict[str, Any]:
    """Run one CV fold across baseline and optional feature selectors."""
    if evaluation_mode not in {"full", "selector_only"}:
        raise ValueError("evaluation_mode must be either 'full' or 'selector_only'.")

    X_train_scaled, X_test_scaled = prepare_fold_arrays(X_train, X_test)

    if evaluation_mode == "full":
        baseline_model = fit_prediction_model(
            X_train_scaled,
            y_train,
            prediction_model_type,
            random_state,
            etree_params,
            device=device,
            cache_dir=cache_dir,
            model_dir=model_dir,
        )
        baseline_train_metrics = evaluate_classifier(baseline_model, X_train_scaled, y_train)
        baseline_metrics = evaluate_classifier(baseline_model, X_test_scaled, y_test)
    else:
        baseline_train_metrics = {"auc": np.nan, "accuracy": np.nan}
        baseline_metrics = {"auc": np.nan, "accuracy": np.nan}

    stg_cfg = dict(stg_params)
    lspin_cfg = dict(lspin_params)

    if feature_selection_method == "lamda_tuning":
        if lambda_value is None:
            raise ValueError("lambda_value is required when feature_selection_method='lamda_tuning'.")
        stg_cfg["lam"] = float(lambda_value)
        lspin_cfg["lam"] = float(lspin_lambda_value if lspin_lambda_value is not None else lambda_value)

    if lspin_cfg.get("batch_size", 64) == -1:
        lspin_cfg["batch_size"] = X_train_scaled.shape[0]

    if run_stg:
        stg_features, stg_train_loss_history, stg_active_count = fit_stg_selector(
            X_train_scaled,
            y_train,
            k=k,
            random_state=random_state,
            feature_selection_method=feature_selection_method,
            **stg_cfg,
        )

        if len(stg_features) == 0:
            stg_features = [0]
            stg_active_count = 0

        if evaluation_mode == "full":
            stg_model = fit_prediction_model(
                X_train_scaled[:, stg_features],
                y_train,
                prediction_model_type,
                random_state,
                etree_params,
                device=device,
                cache_dir=cache_dir,
                model_dir=model_dir,
            )
            stg_train_metrics = evaluate_classifier(stg_model, X_train_scaled[:, stg_features], y_train)
            stg_metrics = evaluate_classifier(stg_model, X_test_scaled[:, stg_features], y_test)
        else:
            stg_train_metrics = {"auc": np.nan, "accuracy": np.nan}
            stg_metrics = {"auc": np.nan, "accuracy": np.nan}
    else:
        stg_train_loss_history = []
        stg_active_count = 0
        stg_train_metrics = {"auc": np.nan, "accuracy": np.nan}
        stg_metrics = {"auc": np.nan, "accuracy": np.nan}

    if run_lspin:
        lspin_features, lspin_train_loss_history, lspin_mean_active_count, lspin_model_obj = fit_lspin_selector(
            X_train_scaled,
            y_train,
            k=k,
            random_state=random_state,
            feature_selection_method=feature_selection_method,
            **lspin_cfg,
        )

        train_gate_matrix = lspin_model_obj.get_prob_alpha(
            torch.as_tensor(X_train_scaled, dtype=torch.float32, device=lspin_model_obj.device)
        )
        if isinstance(train_gate_matrix, torch.Tensor):
            train_gate_matrix = train_gate_matrix.detach().cpu().numpy()

        if train_gate_matrix.ndim == 1:
            train_gate_matrix = train_gate_matrix.reshape(1, -1)

        train_mask = train_gate_matrix > 0
        X_train_lspin_masked = X_train_scaled * train_mask
        active_cols = np.any(train_mask, axis=0)

        if not np.any(active_cols):
            active_cols[0] = True

        X_train_lspin_pruned = X_train_lspin_masked[:, active_cols]
        if evaluation_mode == "full":
            lspin_model_pred = fit_prediction_model(
                X_train_lspin_pruned,
                y_train,
                prediction_model_type,
                random_state,
                etree_params,
                device=device,
                cache_dir=cache_dir,
                model_dir=model_dir,
            )
            lspin_train_metrics = evaluate_classifier(lspin_model_pred, X_train_lspin_pruned, y_train)
        else:
            lspin_train_metrics = {"auc": np.nan, "accuracy": np.nan}

        test_gate_matrix = lspin_model_obj.get_prob_alpha(
            torch.as_tensor(X_test_scaled, dtype=torch.float32, device=lspin_model_obj.device)
        )
        if isinstance(test_gate_matrix, torch.Tensor):
            test_gate_matrix = test_gate_matrix.detach().cpu().numpy()

        if test_gate_matrix.ndim == 1:
            test_gate_matrix = test_gate_matrix.reshape(1, -1)

        test_mask = test_gate_matrix > 0
        X_test_lspin_masked = X_test_scaled * test_mask
        X_test_lspin_pruned = X_test_lspin_masked[:, active_cols]
        if evaluation_mode == "full":
            lspin_metrics = evaluate_classifier(lspin_model_pred, X_test_lspin_pruned, y_test)
        else:
            lspin_metrics = {"auc": np.nan, "accuracy": np.nan}
    else:
        _ = []
        lspin_features = []
        lspin_train_loss_history = []
        lspin_mean_active_count = 0
        lspin_train_metrics = {"auc": np.nan, "accuracy": np.nan}
        lspin_metrics = {"auc": np.nan, "accuracy": np.nan}

    p = int(X_train_scaled.shape[1])
    stg_k = int(stg_active_count)
    lspin_k = int(lspin_mean_active_count)
    stg_ignored = int(max(0, p - stg_k))
    lspin_ignored = int(max(0, p - lspin_k))
    stg_feature_ratio = float(stg_k / p) if p > 0 else np.nan
    lspin_feature_ratio = float(lspin_k / p) if p > 0 else np.nan

    return {
        "dataset": dataset_name,
        "fold": fold_index,
        "p": p,
        "stg_k": stg_k,
        "lspin_k": lspin_k,
        "feature_selection_method": feature_selection_method,
        "selection_value": selection_value,
        "stg_feature_ratio": stg_feature_ratio,
        "lspin_feature_ratio": lspin_feature_ratio,
        "lambda_value": float(lambda_value) if lambda_value is not None else np.nan,
        "lspin_lambda_value": float(lspin_lambda_value) if lspin_lambda_value is not None else np.nan,
        "stg_selected_features": int(stg_active_count),
        "lspin_selected_features": int(lspin_mean_active_count),
        "stg_ignored_features": stg_ignored,
        "lspin_ignored_features": lspin_ignored,
        "baseline_train_auc": baseline_train_metrics["auc"],
        "baseline_train_accuracy": baseline_train_metrics["accuracy"],
        "baseline_auc": baseline_metrics["auc"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "stg_train_auc": stg_train_metrics["auc"],
        "stg_train_accuracy": stg_train_metrics["accuracy"],
        "stg_auc": stg_metrics["auc"],
        "stg_accuracy": stg_metrics["accuracy"],
        "stg_train_loss_history": stg_train_loss_history,
        "lspin_train_auc": lspin_train_metrics["auc"],
        "lspin_train_accuracy": lspin_train_metrics["accuracy"],
        "lspin_auc": lspin_metrics["auc"],
        "lspin_accuracy": lspin_metrics["accuracy"],
        "lspin_train_loss_history": lspin_train_loss_history,
        "stg_auc_delta": (
            stg_metrics["auc"] - baseline_metrics["auc"]
            if evaluation_mode == "full"
            else np.nan
        ),
        "lspin_auc_delta": (
            lspin_metrics["auc"] - baseline_metrics["auc"]
            if evaluation_mode == "full"
            else np.nan
        ),
        "stg_accuracy_delta": (
            stg_metrics["accuracy"] - baseline_metrics["accuracy"]
            if evaluation_mode == "full"
            else np.nan
        ),
        "lspin_accuracy_delta": (
            lspin_metrics["accuracy"] - baseline_metrics["accuracy"]
            if evaluation_mode == "full"
            else np.nan
        ),
    }


def run_dataset_experiment(
    dataset_name: str,
    dataset: Any,
    output_dir: Path,
    device: torch.device,
    cache_dir: Path,
    model_dir: Path,
    random_state: int = 42,
    n_splits: int = 5,
    stg_params: Mapping[str, Any] | None = None,
    lspin_params: Mapping[str, Any] | None = None,
    etree_params: Mapping[str, Any] | None = None,
    feature_selection_method: str = "features_ratio",
    feature_ratios: list[float] | None = None,
    lambda_values: list[float] | None = None,
    stg_lambda_values: list[float] | None = None,
    lspin_lambda_values: list[float] | None = None,
    evaluation_mode: str = "full",
    prediction_model_type: str = "etree",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full CV experiment for one dataset."""
    set_global_seed(random_state)
    stg_params = dict(stg_params or {})
    lspin_params = dict(lspin_params or {})
    etree_params = dict(etree_params or {})

    if feature_selection_method not in {"features_ratio", "lamda_tuning"}:
        raise ValueError("feature_selection_method must be either 'features_ratio' or 'lamda_tuning'.")
    if evaluation_mode not in {"full", "selector_only"}:
        raise ValueError("evaluation_mode must be either 'full' or 'selector_only'.")

    X, y = extract_xy(dataset)
    y_encoded, _ = encode_labels(y)
    p = X.shape[1]

    if feature_selection_method == "features_ratio":
        selection_values = [float(v) for v in (feature_ratios or [])]
    else:
        if stg_lambda_values is not None or lspin_lambda_values is not None:
            stg_values = [float(v) for v in (stg_lambda_values or [])]
            lspin_values = [float(v) for v in (lspin_lambda_values or [])]
            if len(stg_values) != len(lspin_values):
                raise ValueError("stg_lambda_values and lspin_lambda_values must have the same length.")
            selection_values = list(zip(stg_values, lspin_values))
        else:
            selection_values = [(float(v), float(v)) for v in (lambda_values or [])]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows: list[dict[str, Any]] = []
    checkpoint_df = load_checkpoint(output_dir, dataset_name)

    key_cols = {"dataset", "feature_selection_method", "selection_value", "fold"}
    if not checkpoint_df.empty and not key_cols.issubset(checkpoint_df.columns):
        print(f"Checkpoint for {dataset_name} is incompatible. Rebuilding from scratch.")
        checkpoint_df = pd.DataFrame()

    completed_keys = set()
    if not checkpoint_df.empty:
        completed_keys = {
            checkpoint_key(
                row.dataset,
                str(row.feature_selection_method),
                str(row.selection_value),
                int(row.fold),
            )
            for row in checkpoint_df.itertuples(index=False)
        }
        fold_rows.extend(checkpoint_df.to_dict(orient="records"))
        print(f"Resuming {dataset_name} from {len(completed_keys)} completed jobs")

    stg_enabled = True
    lspin_enabled = True

    for selection_value in selection_values:
        if not stg_enabled and not lspin_enabled:
            print(f"Early stopping lambda tuning for {dataset_name}: both selectors reached zero features")
            break

        if feature_selection_method == "lamda_tuning":
            stg_lambda_value, lspin_lambda_value = selection_value
            lambda_value = float(stg_lambda_value)
            selection_key = f"stg={float(stg_lambda_value):.6g}|lspin={float(lspin_lambda_value):.6g}"
            ratio = None
        else:
            lambda_value = None
            lspin_lambda_value = None
            ratio = float(selection_value)
            selection_key = f"ratio={ratio:.6g}"

        k = ratio_to_k(p, ratio) if ratio is not None else None

        for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded), start=1):
            current_key = checkpoint_key(dataset_name, feature_selection_method, selection_key, fold_index)
            if current_key in completed_keys:
                continue

            row = run_fold_experiment(
                dataset_name=dataset_name,
                fold_index=fold_index,
                X_train=X[train_idx],
                X_test=X[test_idx],
                y_train=y_encoded[train_idx],
                y_test=y_encoded[test_idx],
                random_state=random_state + fold_index,
                stg_params=stg_params,
                lspin_params=lspin_params,
                etree_params=etree_params,
                feature_selection_method=feature_selection_method,
                selection_value=selection_key,
                device=device,
                cache_dir=cache_dir,
                model_dir=model_dir,
                k=k,
                lambda_value=lambda_value,
                lspin_lambda_value=lspin_lambda_value,
                run_stg=stg_enabled,
                run_lspin=lspin_enabled,
                evaluation_mode=evaluation_mode,
                prediction_model_type=prediction_model_type,
            )
            fold_rows.append(row)
            save_checkpoint(output_dir, dataset_name, pd.DataFrame(fold_rows))

        if feature_selection_method == "lamda_tuning":
            current_folds = [row for row in fold_rows if row["selection_value"] == selection_key]
            if len(current_folds) == n_splits:
                if stg_enabled:
                    stg_k_avg = np.mean([row["stg_selected_features"] for row in current_folds])
                    if stg_k_avg <= 0:
                        print(f"Disabling STG for higher lambdas after {selection_key}")
                        stg_enabled = False

                if lspin_enabled:
                    lspin_k_avg = np.mean([row["lspin_selected_features"] for row in current_folds])
                    if lspin_k_avg <= 0:
                        print(f"Disabling LSPIN for higher lambdas after {selection_key}")
                        lspin_enabled = False

    fold_df = pd.DataFrame(fold_rows)
    loss_history_df = expand_loss_histories(fold_df)
    summary_df = summarize_fold_results(fold_df)
    return summary_df, fold_df, loss_history_df
