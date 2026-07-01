"""Cross-validation experiment engine for baseline, STG, and LSPIN."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data_preprocessing import encode_labels, extract_xy, prepare_fold_arrays
from src.feature_selection import fit_lspin_selector, fit_stg_selector, ratio_to_k
from src.model_training import evaluate_classifier, fit_prediction_model
from src.utils import set_global_seed
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from src.feature_selection import fit_concrete_selector
from typing import Tuple


def _is_nan(value: Any) -> bool:
    """Return True when value is None/NaN-like."""
    return value is None or (pd.isna(value) if not isinstance(value, str) else False)


def _format_key_part(value: Any) -> str:
    """Format key values consistently for stable checkpoint identity."""
    if _is_nan(value):
        return "nan"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def _parse_legacy_selection(selection_value: Any, feature_selection_method: str) -> tuple[Any, float | None, float | None]:
    """Parse legacy selection_value strings from older checkpoints."""
    if feature_selection_method == "lamda_tuning":
        if isinstance(selection_value, str) and selection_value.startswith("stg=") and "|lspin=" in selection_value:
            stg_part, lspin_part = selection_value.split("|lspin=", maxsplit=1)
            try:
                return None, float(stg_part.replace("stg=", "", 1)), float(lspin_part)
            except ValueError:
                return None, None, None
        return None, None, None

    if isinstance(selection_value, str) and selection_value.startswith("ratio="):
        try:
            return float(selection_value.replace("ratio=", "", 1)), None, None
        except ValueError:
            return selection_value, None, None
    return selection_value, None, None


def checkpoint_key(
    dataset_name: str,
    feature_selection_method: str,
    fold_index: int,
    selection_value: Any = None,
    stg_lambda_value: float | None = None,
    lspin_lambda_value: float | None = None,
) -> str:
    """Build stable key for checkpointed fold jobs."""
    if feature_selection_method == "lamda_tuning":
        selection_part = f"stg={_format_key_part(stg_lambda_value)}|lspin={_format_key_part(lspin_lambda_value)}"
    else:
        selection_part = f"selection={_format_key_part(selection_value)}"
    return f"{dataset_name}|{feature_selection_method}|{selection_part}|{fold_index}"


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


def get_selector_checkpoint_path(output_dir: Path, dataset_name: str, selector_name: str) -> Path:
    """Return checkpoint path for single-split selector runs."""
    safe_selector_name = selector_name.replace("/", "_").replace(" ", "_")
    return output_dir / f"checkpoint_{dataset_name}_{safe_selector_name}.pkl"


def load_selector_checkpoint(output_dir: Path, dataset_name: str, selector_name: str) -> pd.DataFrame:
    """Load single-split selector checkpoint if available."""
    checkpoint_path = get_selector_checkpoint_path(output_dir, dataset_name, selector_name)
    if checkpoint_path.exists():
        return pd.read_pickle(checkpoint_path)
    return pd.DataFrame()


def save_selector_checkpoint(output_dir: Path, dataset_name: str, selector_name: str, selector_df: pd.DataFrame) -> None:
    """Persist single-split selector checkpoint."""
    checkpoint_path = get_selector_checkpoint_path(output_dir, dataset_name, selector_name)
    selector_df.to_pickle(checkpoint_path)


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
            "stg_lambda_value": float(getattr(row, "stg_lambda_value", np.nan)),
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
        "stg_lambda_value",
        "lspin_lambda_value",
        "p",
    ]

    for keys, part in fold_df.groupby(group_cols, dropna=False):
        (
            dataset_name,
            feature_selection_method,
            selection_value,
            stg_lambda_value,
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
                "stg_lambda_value": float(stg_lambda_value) if pd.notna(stg_lambda_value) else np.nan,
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
    statistical_prefilter_k: int = 1000,
    k: int | None = None,
    stg_lambda_value: float | None = None,
    lspin_lambda_value: float | None = None,
    run_stg: bool = True,
    run_lspin: bool = True,
    evaluation_mode: str = "full",
    use_peeling: bool = False,
    peeling_tau: int = 50,
    peeling_low_auc_threshold: float = 0.70,
) -> dict[str, Any]:
    """Run one CV fold across baseline and optional feature selectors."""
    print(
        f"[Fold {fold_index}] Starting | dataset={dataset_name} | method={feature_selection_method} "
        f"| selection={selection_value}"
    )

    if evaluation_mode not in {"full", "selector_only"}:
        raise ValueError("evaluation_mode must be either 'full' or 'selector_only'.")

    if use_peeling:
        from src.peeling import peeling_procedure
        X_train, y_train, peeling_status = peeling_procedure(
            X_train, y_train, X_test, y_test,
            tau=peeling_tau,
            low_auc_threshold=peeling_low_auc_threshold,
            random_state=random_state
        )

    X_train_scaled, X_test_scaled = prepare_fold_arrays(X_train, X_test)

    if prediction_model_type == "tabiclv2":
        X_train_scaled, X_test_scaled, _ = _apply_statistical_prefilter(
            X_train_scaled,
            X_test_scaled,
            y_train,
            prefilter_k=statistical_prefilter_k,
        )

    if evaluation_mode == "full":
        print(f"[Fold {fold_index}] Baseline training started ({prediction_model_type}).")
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
        print(
            f"[Fold {fold_index}] Baseline complete | "
            f"AUC={baseline_metrics['auc']:.4f} | ACC={baseline_metrics['accuracy']:.4f}"
        )
    else:
        baseline_train_metrics = {"auc": np.nan, "accuracy": np.nan}
        baseline_metrics = {"auc": np.nan, "accuracy": np.nan}

    stg_cfg = dict(stg_params)
    lspin_cfg = dict(lspin_params)

    if feature_selection_method == "lamda_tuning":
        if stg_lambda_value is None:
            raise ValueError("stg_lambda_value is required when feature_selection_method='lamda_tuning'.")
        stg_cfg["lam"] = float(stg_lambda_value)
        lspin_cfg["lam"] = float(lspin_lambda_value if lspin_lambda_value is not None else stg_lambda_value)

    if lspin_cfg.get("batch_size", 64) == -1:
        lspin_cfg["batch_size"] = X_train_scaled.shape[0]

    if run_stg:
        print(f"[Fold {fold_index}] STG selector training started.")
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

        print(
            f"[Fold {fold_index}] STG selector complete | selected={int(stg_active_count)}"
        )

        if evaluation_mode == "full":
            print(f"[Fold {fold_index}] STG downstream model training started ({prediction_model_type}).")
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
            print(
                f"[Fold {fold_index}] STG downstream complete | "
                f"AUC={stg_metrics['auc']:.4f} | ACC={stg_metrics['accuracy']:.4f}"
            )
        else:
            stg_train_metrics = {"auc": np.nan, "accuracy": np.nan}
            stg_metrics = {"auc": np.nan, "accuracy": np.nan}
    else:
        stg_train_loss_history = []
        stg_active_count = 0
        stg_train_metrics = {"auc": np.nan, "accuracy": np.nan}
        stg_metrics = {"auc": np.nan, "accuracy": np.nan}

    if run_lspin:
        print(f"[Fold {fold_index}] LSPIN selector training started.")
        lspin_features, lspin_train_loss_history, lspin_mean_active_count, lspin_model_obj = fit_lspin_selector(
            X_train_scaled,
            y_train,
            k=k,
            random_state=random_state,
            feature_selection_method=feature_selection_method,
            **lspin_cfg,
        )

        print(
            f"[Fold {fold_index}] LSPIN selector complete | selected_mean={float(lspin_mean_active_count):.2f}"
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
            print(f"[Fold {fold_index}] LSPIN downstream model training started ({prediction_model_type}).")
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
            print(
                f"[Fold {fold_index}] LSPIN downstream complete | "
                f"AUC={lspin_metrics['auc']:.4f} | ACC={lspin_metrics['accuracy']:.4f}"
            )
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

    print(
        f"[Fold {fold_index}] Done | baseline_auc={baseline_metrics['auc']:.4f} "
        f"| stg_auc={stg_metrics['auc']:.4f} | lspin_auc={lspin_metrics['auc']:.4f}"
    )

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
        "stg_lambda_value": float(stg_lambda_value) if stg_lambda_value is not None else np.nan,
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


def _apply_statistical_prefilter(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, prefilter_k: int = 1000
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Apply VarianceThreshold then optional SelectKBest (fit on train only).

    Returns transformed X_train, X_test and the fitted selector object (SelectKBest or None).
    """
    X_train_np = np.asarray(X_train)
    X_test_np = np.asarray(X_test)

    var_filter = VarianceThreshold()
    X_train_np = var_filter.fit_transform(X_train_np)
    X_test_np = var_filter.transform(X_test_np)

    selector = None
    if X_train_np.shape[1] > prefilter_k:
        selector = SelectKBest(score_func=f_classif, k=prefilter_k)
        X_train_np = selector.fit_transform(X_train_np, y_train)
        X_test_np = selector.transform(X_test_np)

    return X_train_np, X_test_np, selector


def run_baseline_prefilter_tabicl(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    prediction_model_type: str,
    device: torch.device,
    output_dir: Path,
    cache_dir: Path | None,
    model_dir: Path | None,
    random_state: int,
    prefilter_k: int = 1000,
    baseline_k_cap: int = 100,
) -> dict[str, Any]:
    """Run baseline: VarianceThreshold -> SelectKBest (to <= baseline_k_cap) -> TabICL/etree evaluation."""
    X_tr_pf, X_te_pf, selector = _apply_statistical_prefilter(X_train, X_test, y_train, prefilter_k=prefilter_k)

    # Cap features to baseline_k_cap using SelectKBest (fit on train)
    k_actual = min(baseline_k_cap, X_tr_pf.shape[1])
    if X_tr_pf.shape[1] > k_actual:
        cap_selector = SelectKBest(score_func=f_classif, k=k_actual)
        X_tr_pf = cap_selector.fit_transform(X_tr_pf, y_train)
        X_te_pf = cap_selector.transform(X_te_pf)
    else:
        X_te_pf = X_te_pf

    # Train predictor and evaluate
    model = fit_prediction_model(
        X_tr_pf,
        y_train,
        prediction_model_type,
        random_state,
        {},
        device=device,
        cache_dir=cache_dir,
        model_dir=model_dir,
    )
    train_metrics = evaluate_classifier(model, X_tr_pf, y_train)
    test_metrics = evaluate_classifier(model, X_te_pf, y_test)

    return {
        "baseline_prefilter_features": int(X_tr_pf.shape[1]),
        "baseline_train_auc": train_metrics.get("auc", np.nan),
        "baseline_auc": test_metrics.get("auc", np.nan),
        "baseline_train_accuracy": train_metrics.get("accuracy", np.nan),
        "baseline_accuracy": test_metrics.get("accuracy", np.nan),
    }


def run_concrete_single_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int,
    prediction_model_type: str,
    random_state: int,
    device: torch.device,
    prefilter_k: int = 1000,
    num_epochs: int = 100,
    cache_dir: Path | None = None,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Concrete Autoencoder on a single train/test split then train TabICL on selected features."""
    X_tr_pf, X_te_pf, _ = _apply_statistical_prefilter(X_train, X_test, y_train, prefilter_k=prefilter_k)

    indices, loss_history, selector_obj = fit_concrete_selector(X_tr_pf, k=k, num_epochs=num_epochs)

    X_tr_sel = X_tr_pf[:, indices]
    X_te_sel = X_te_pf[:, indices]

    # Fit predictor
    model = fit_prediction_model(
        X_tr_sel,
        y_train,
        prediction_model_type,
        random_state,
        {},
        device=device,
        cache_dir=cache_dir,
        model_dir=model_dir,
    )
    train_metrics = evaluate_classifier(model, X_tr_sel, y_train)
    test_metrics = evaluate_classifier(model, X_te_sel, y_test)

    return {
        "concrete_k": int(k),
        "concrete_selected_count": int(len(indices)),
        "concrete_train_loss_history": loss_history,
        "concrete_train_auc": train_metrics.get("auc", np.nan),
        "concrete_auc": test_metrics.get("auc", np.nan),
        "concrete_train_accuracy": train_metrics.get("accuracy", np.nan),
        "concrete_accuracy": test_metrics.get("accuracy", np.nan),
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
    run_stg: bool = True,
    run_lspin: bool = True,
    evaluation_mode: str = "full",
    prediction_model_type: str = "etree",
    statistical_prefilter_k: int = 1000,
    use_peeling: bool = False,
    peeling_tau: int = 50,
    peeling_low_auc_threshold: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full CV experiment for one dataset."""
    print(
        f"[Dataset] Starting experiment | dataset={dataset_name} | n_splits={n_splits} "
        f"| mode={feature_selection_method} | eval={evaluation_mode}"
    )
    set_global_seed(random_state)
    stg_params = dict(stg_params or {})
    lspin_params = dict(lspin_params or {})
    etree_params = dict(etree_params or {})

    # Backward-compatible normalization: accept both spellings.
    if feature_selection_method == "lambda_tuning":
        feature_selection_method = "lamda_tuning"

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

    print(f"[Dataset] Selection settings to run: {len(selection_values)}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_rows: list[dict[str, Any]] = []
    checkpoint_df = load_checkpoint(output_dir, dataset_name)

    key_cols = {"dataset", "feature_selection_method", "selection_value", "fold"}
    if not checkpoint_df.empty and not key_cols.issubset(checkpoint_df.columns):
        print(f"Checkpoint for {dataset_name} is incompatible. Rebuilding from scratch.")
        checkpoint_df = pd.DataFrame()

    completed_keys = set()
    if not checkpoint_df.empty:
        for row in checkpoint_df.itertuples(index=False):
            method = str(getattr(row, "feature_selection_method"))
            raw_selection = getattr(row, "selection_value", None)
            parsed_selection, parsed_stg_lambda, parsed_lspin_lambda = _parse_legacy_selection(raw_selection, method)

            stg_lambda_for_key = getattr(row, "stg_lambda_value", parsed_stg_lambda)
            lspin_lambda_for_key = getattr(row, "lspin_lambda_value", parsed_lspin_lambda)

            completed_keys.add(
                checkpoint_key(
                    row.dataset,
                    method,
                    int(row.fold),
                    selection_value=parsed_selection,
                    stg_lambda_value=stg_lambda_for_key,
                    lspin_lambda_value=lspin_lambda_for_key,
                )
            )

        fold_rows.extend(checkpoint_df.to_dict(orient="records"))
        print(f"Resuming {dataset_name} from {len(completed_keys)} completed jobs")

    stg_enabled = True
    lspin_enabled = True

    for selection_idx, selection_value in enumerate(selection_values, start=1):
        if not stg_enabled and not lspin_enabled:
            print(f"Early stopping lambda tuning for {dataset_name}: both selectors reached zero features")
            break

        if feature_selection_method == "lamda_tuning":
            print(
                f"[Dataset] Setting {selection_idx}/{len(selection_values)} | "
                f"STG lambda={selection_value[0]:.6g}, LSPIN lambda={selection_value[1]:.6g}"
            )
            stg_lambda_value, lspin_lambda_value = selection_value
            row_selection_value = float(stg_lambda_value)
            ratio = None
        else:
            print(
                f"[Dataset] Setting {selection_idx}/{len(selection_values)} | "
                f"feature_ratio={selection_value:.6g}"
            )
            stg_lambda_value = None
            lspin_lambda_value = None
            ratio = float(selection_value)
            row_selection_value = ratio

        k = ratio_to_k(p, ratio) if ratio is not None else None

        for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded), start=1):
            current_key = checkpoint_key(
                dataset_name,
                feature_selection_method,
                fold_index,
                selection_value=row_selection_value,
                stg_lambda_value=stg_lambda_value,
                lspin_lambda_value=lspin_lambda_value,
            )
            if current_key in completed_keys:
                print(f"[Fold {fold_index}] Skipping (checkpoint hit)")
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
                selection_value=row_selection_value,
                device=device,
                cache_dir=cache_dir,
                model_dir=model_dir,
                k=k,
                stg_lambda_value=stg_lambda_value,
                lspin_lambda_value=lspin_lambda_value,
                run_stg=stg_enabled and run_stg,
                run_lspin=lspin_enabled and run_lspin,
                evaluation_mode=evaluation_mode,
                prediction_model_type=prediction_model_type,
                statistical_prefilter_k=statistical_prefilter_k,
                use_peeling=use_peeling,
                peeling_tau=peeling_tau,
                peeling_low_auc_threshold=peeling_low_auc_threshold,
            )
            fold_rows.append(row)
            save_checkpoint(output_dir, dataset_name, pd.DataFrame(fold_rows))
            print(f"[Fold {fold_index}] Checkpoint saved.")

        if feature_selection_method == "lamda_tuning":
            current_folds = [
                row
                for row in fold_rows
                if float(row.get("stg_lambda_value",  np.nan)) == float(stg_lambda_value)
                and float(row.get("lspin_lambda_value", np.nan)) == float(lspin_lambda_value)
            ]
            if len(current_folds) == n_splits:
                if stg_enabled:
                    stg_k_avg = np.mean([row["stg_selected_features"] for row in current_folds])
                    if stg_k_avg <= 0:
                        print(
                            "Disabling STG for higher lambdas after "
                            f"stg={float(stg_lambda_value):.6g}|lspin={float(lspin_lambda_value):.6g}"
                        )
                        stg_enabled = False

                if lspin_enabled:
                    lspin_k_avg = np.mean([row["lspin_selected_features"] for row in current_folds])
                    if lspin_k_avg <= 0:
                        print(
                            "Disabling LSPIN for higher lambdas after "
                            f"stg={float(stg_lambda_value):.6g}|lspin={float(lspin_lambda_value):.6g}"
                        )
                        lspin_enabled = False

    fold_df = pd.DataFrame(fold_rows)
    loss_history_df = expand_loss_histories(fold_df)
    summary_df = summarize_fold_results(fold_df)
    print(
        f"[Dataset] Complete | rows={len(fold_df)} | summary_rows={len(summary_df)} "
        f"| loss_rows={len(loss_history_df)}"
    )
    return summary_df, fold_df, loss_history_df


def run_single_split_selector_experiment(
    dataset_name: str,
    dataset: Any,
    output_dir: Path,
    device: torch.device,
    cache_dir: Path,
    model_dir: Path,
    random_state: int,
    prediction_model_type: str,
    selector_name: str,
    concrete_k_values: list[int] | None = None,
    concrete_epochs: int = 100,
    concrete_prefilter_k: int = 1000,
    baseline_postprefilter_k_cap: int = 100,
) -> pd.DataFrame:
    """Run Concrete or baseline selector sweeps on a single train/test split."""
    X, y = extract_xy(dataset)
    y_encoded, _ = encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=random_state,
        stratify=y_encoded,
    )

    selector_key = selector_name.strip().lower()
    rows: list[dict[str, Any]] = []

    checkpoint_df = load_selector_checkpoint(output_dir, dataset_name, selector_key)
    if not checkpoint_df.empty and not {"dataset", "selector"}.issubset(checkpoint_df.columns):
        print(f"Selector checkpoint for {dataset_name}/{selector_key} is incompatible. Rebuilding from scratch.")
        checkpoint_df = pd.DataFrame()

    completed_keys: set[str] = set()
    if not checkpoint_df.empty:
        key_columns = ["dataset", "selector"]
        if selector_key in {"concrete", "concrete_autoencoder"}:
            key_columns.append("concrete_k")
        completed_keys = {
            "|".join(str(getattr(row, column)) for column in key_columns)
            for row in checkpoint_df.itertuples(index=False)
        }
        rows.extend(checkpoint_df.to_dict(orient="records"))

    if selector_key in {"concrete", "concrete_autoencoder"} and prediction_model_type != "tabiclv2":
        raise ValueError("Concrete Autoencoder is only supported when prediction_model_type='tabiclv2'.")

    if selector_key in {"concrete", "concrete_autoencoder"}:
        for k in concrete_k_values or [10]:
            current_key = "|".join([dataset_name, "concrete_autoencoder", str(int(k))])
            if current_key in completed_keys:
                continue

            metrics = run_concrete_single_split(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                k=int(k),
                prediction_model_type=prediction_model_type,
                random_state=random_state,
                prefilter_k=concrete_prefilter_k,
                num_epochs=concrete_epochs,
                cache_dir=cache_dir,
                model_dir=model_dir,
                device=device,
            )
            rows.append({"dataset": dataset_name, "selector": "concrete_autoencoder", **metrics})
            save_selector_checkpoint(output_dir, dataset_name, selector_key, pd.DataFrame(rows))

        return pd.DataFrame(rows)

    if selector_key in {"baseline", "prefilter_baseline", "variance_selectkbest_baseline"}:
        current_key = "|".join([dataset_name, "baseline"])
        if current_key not in completed_keys:
            metrics = run_baseline_prefilter_tabicl(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                prediction_model_type=prediction_model_type,
                device=device,
                output_dir=output_dir,
                cache_dir=cache_dir,
                model_dir=model_dir,
                random_state=random_state,
                prefilter_k=concrete_prefilter_k,
                baseline_k_cap=baseline_postprefilter_k_cap,
            )
            rows.append({"dataset": dataset_name, "selector": "baseline", **metrics})
            save_selector_checkpoint(output_dir, dataset_name, selector_key, pd.DataFrame(rows))

        return pd.DataFrame(rows)

    raise ValueError(f"Unsupported single-split selector: {selector_name}")
