import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import numpy as np
import yaml
import pandas as pd
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

# Import necessary modules
from tabicl import TabICLClassifier

from src.utils import ensure_dir
from src.config import resolve_paths
from src.data_preprocessing import build_dataset_paths, load_dataset_xy
from src.model_training import evaluate_classifier, fit_extra_trees
from src.plotting import plot_concrete_results

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_pipeline(config_path: str, resume_output_dir_override: str) -> Dict[str, Dict[str, float]]:
    """
    Run an end-to-end pipeline:
    1. Parse configuration.
    2. Setup output directories.
    3. Load and split dataset.
    4. Select features using Concrete Autoencoder.
    5. Predict using TabICLv2.
    6. Evaluate predictions and save outputs (plots, CSV, config).
    """
    # 1. Parse Configuration
    config = load_config(config_path)
    
    dataset_path = config.get('dataset_path')
    dataset_names = config.get('dataset_names') or []
    target_column = config.get('target_column_name', 'target')
    task_type = config.get('task_type', 'classification')
    k_features_cfg = config.get('concrete_k_values', 10)
    concrete_device = str(config.get('concrete_device', 'cpu')).strip().lower()
    resume_output_dir = resume_output_dir_override or config.get('resume_output_dir', None)
    resume_from_checkpoints = bool(config.get('resume_from_checkpoints', True))
    skip_completed = bool(config.get('skip_completed', True))

    if concrete_device not in {'cpu', 'gpu', 'auto'}:
        raise ValueError("config.yml option 'concrete_device' must be one of: cpu, gpu, auto")

    # Configure TensorFlow device behavior before importing Concrete Autoencoder.
    # Defaulting to CPU avoids common GPU OOM failures during TF context init.
    if concrete_device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

    from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
    from keras.layers import Dense

    if concrete_device in {'gpu', 'auto'}:
        try:
            import tensorflow as tf
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            # If memory growth cannot be configured, continue and rely on TF defaults.
            pass

    print(f"Concrete Autoencoder device mode: {concrete_device}")

    if not dataset_names and not dataset_path:
        raise ValueError("config.yml must specify 'dataset_names' or 'dataset_path'")

    if dataset_path is not None and not target_column:
        if Path(dataset_path).exists() and Path(dataset_path).suffix.lower() != '.mat':
            df = pd.read_csv(dataset_path)
            print(f"Columns in dataset: {df.columns.tolist()}")
        raise ValueError("config.yml must specify 'target_column_name' for CSV datasets")

    if isinstance(k_features_cfg, (list, tuple)):
        k_values = [int(v) for v in k_features_cfg]
    else:
        k_values = [int(k_features_cfg)]

    # 2. Setup Output Directories (Similar to main.py / pipeline.py)
    base_dir = Path.cwd()
    paths = resolve_paths(base_dir=base_dir)
    if resume_output_dir:
        output_dir = ensure_dir(Path(resume_output_dir).expanduser().resolve())
        print(f"Resuming concrete run in existing output directory: {output_dir}")
    else:
        output_dir = ensure_dir(paths.run_output_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")

    config_source_path = Path(config_path).resolve()
    if config_source_path.exists():
        shutil.copy2(config_source_path, output_dir / "config.yml")
    else:
        print(f"Warning: config file not found at {config_source_path}; saving in-memory config copy only.")

    run_hparams = {
        "feature_selector": "concrete_autoencoder",
        "prediction_model": "TabICLClassifier",
        "baseline_model": {
            "name": "ExtraTrees",
            "n_estimators": 100,
            "max_depth": 3,
        },
        "split": {
            "test_size": 0.2,
            "random_state": 42,
        },
        "prefilter": {
            "variance_threshold": "enabled",
            "select_k_best_k": 1000,
            "score_func": "f_classif",
        },
        "concrete_autoencoder": {
            "k_values": [int(v) for v in k_values],
            "num_epochs": 100,
            "tryout_limit": 1,
            "device_mode": concrete_device,
        },
    }
    with (output_dir / "hyperparameters.yaml").open("w") as f:
        yaml.safe_dump(run_hparams, f, sort_keys=False)

    with (output_dir / "effective_config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    
    results: dict[str, Dict[str, float]] = {}
    concrete_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []

    summary_csv = output_dir / "iterative_feature_curve_summary.csv"
    if summary_csv.exists():
        existing_summary_df = pd.read_csv(summary_csv)
        if not existing_summary_df.empty:
            concrete_rows.extend(existing_summary_df.to_dict(orient="records"))

    loss_csv = output_dir / "iterative_feature_curve_loss_history.csv"
    if loss_csv.exists():
        existing_loss_df = pd.read_csv(loss_csv)
        if not existing_loss_df.empty:
            loss_rows.extend(existing_loss_df.to_dict(orient="records"))

    completed_keys: set[tuple[str, int]] = set()
    for row in concrete_rows:
        dataset_value = row.get("dataset_name")
        k_value = row.get("k_features")
        if pd.notna(dataset_value) and pd.notna(k_value):
            completed_keys.add((str(dataset_value), int(k_value)))

    if dataset_names:
        dataset_paths = build_dataset_paths(paths.data_root)
        datasets_to_run = []
        for name in dataset_names:
            dataset_path_obj = dataset_paths.get(name)
            if dataset_path_obj is None:
                print(f"Skipping unknown dataset name: {name}")
                continue
            if not dataset_path_obj.exists():
                print(f"Skipping missing dataset: {name} at {dataset_path_obj}")
                continue
            datasets_to_run.append((name, dataset_path_obj))
    else:
        datasets_to_run = [(Path(dataset_path).stem, Path(dataset_path))]

    for dataset_name, dataset_path_obj in datasets_to_run:
        # 3. Data Preparation
        if dataset_path_obj.suffix.lower() == '.mat':
            X, y = load_dataset_xy(dataset_path_obj)
        else:
            df = pd.read_csv(dataset_path_obj)
            X = df.drop(columns=[target_column])
            y = df[target_column]

        for cae_k in k_values:
            run_label = f"{dataset_name}_k{cae_k}"
            print(f"\n=== Running {run_label} ===")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 3.5 Statistical Pre-filtering
            X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
            X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)

            print("Applying VarianceThreshold pre-filtering...")
            variance_filter = VarianceThreshold()
            X_train_np = variance_filter.fit_transform(X_train_np)
            X_test_np = variance_filter.transform(X_test_np)

            prefilter_k = 1000
            if X_train_np.shape[1] > prefilter_k:
                print(f"Pre-filtering to top {prefilter_k} features using SelectKBest...")
                prefilter = SelectKBest(score_func=f_classif, k=prefilter_k)
                X_train_np = prefilter.fit_transform(X_train_np, y_train)
                X_test_np = prefilter.transform(X_test_np)
            else:
                print("Skipping SelectKBest (feature count <= 1000 after variance filtering).")

            # 4. Feature Selection (Concrete Autoencoder)
            print(f"Selecting top {cae_k} features using Concrete Autoencoder...")

            if skip_completed and (dataset_name, int(cae_k)) in completed_keys:
                print(f"Skipping {run_label}: already present in {summary_csv.name}")
                continue

            def decoder(x):
                return Dense(X_train_np.shape[1])(x)

            selector = ConcreteAutoencoderFeatureSelector(
                K=cae_k,
                output_function=decoder,
                num_epochs=100,
                # Set tryout_limit to 1 to prevent epoch doubling
                tryout_limit=1,
            )

            checkpoint_path = checkpoints_dir / f"{dataset_name}_k{cae_k}_concrete_autoencoder.keras"
            indices_path = checkpoints_dir / f"{dataset_name}_k{cae_k}_indices.npy"
            checkpoint_status = "saved"
            history = None

            if resume_from_checkpoints and indices_path.exists():
                indices = np.load(indices_path).astype(int)
                checkpoint_status = "loaded_indices"
                print(f"Loaded feature indices from checkpoint: {indices_path.name}")
            else:
                # Fit strictly on training data (Autoencoder reconstructs X_train)
                try:
                    selector.fit(X_train_np, X_train_np, val_X=X_test_np, val_Y=X_test_np)
                except RuntimeError as exc:
                    msg = str(exc)
                    if 'RESOURCE_EXHAUSTED' in msg or 'CUDA_ERROR_OUT_OF_MEMORY' in msg:
                        raise RuntimeError(
                            "Concrete Autoencoder ran out of GPU memory. "
                            "Set concrete_device: cpu in config.yml (or keep the default cpu) "
                            "to run feature selection on CPU."
                        ) from exc
                    raise

                history = getattr(getattr(selector, "model", None), "history", None)

                try:
                    if getattr(selector, "model", None) is not None:
                        selector.model.save(checkpoint_path)
                    else:
                        checkpoint_status = "unsupported"
                except Exception as exc:
                    checkpoint_status = f"failed: {exc.__class__.__name__}"
                    checkpoint_path = None

                indices = selector.get_indices()
                np.save(indices_path, np.asarray(indices, dtype=np.int64))

            # Transform both train and test sets structure
            # Note: ConcreteAutoencoder's transform method may fail on Pandas Dataframes or has a bug (using 1D index on 2D array)
            # so we extract indices manually to be safe.
            X_train_reduced = X_train_np[:, indices]
            X_test_reduced = X_test_np[:, indices]

            # 4. Prediction (TabICLv2)
            print(f"Training TabICLv2 model for {task_type}...")
            if task_type == 'classification':
                model = TabICLClassifier()
            else:
                # Assuming there is a regressor or using the classifier if the package handles it
                raise NotImplementedError(
                    f"Task type '{task_type}' not currently implemented in this example (expected 'classification')."
                )

            # Fit using reduced datasets
            model.fit(X_train_reduced, y_train)

            # 6. Evaluation and output saving
            metrics: Dict[str, float] = {}
            train_metrics: Dict[str, float] = {}
            baseline_metrics: Dict[str, float] = {}
            baseline_train_metrics: Dict[str, float] = {}
            if task_type == 'classification':
                predictions = model.predict(X_test_reduced)
                metrics['Accuracy'] = accuracy_score(y_test, predictions)
                # Assuming predictions contains labels; if it has probabilities, compute AUC
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_test_reduced)
                    if probas.shape[1] == 2:
                        metrics['AUC'] = roc_auc_score(y_test, probas[:, 1])
                    else:
                        metrics['AUC'] = roc_auc_score(y_test, probas, multi_class='ovr')

                train_metrics = evaluate_classifier(model, X_train_reduced, y_train)

                baseline_model = fit_extra_trees(X_train, np.asarray(y_train), random_state=42)
                baseline_train_metrics = evaluate_classifier(baseline_model, X_train, y_train)
                baseline_metrics = evaluate_classifier(baseline_model, X_test, y_test)

            print(f"Pipeline evaluation completed. Metrics: {metrics}")

            # Save metrics cleanly into a CSV file inside the output directory
            metrics_df = pd.DataFrame(
                [
                    {
                        "dataset_name": dataset_name,
                        "selector": "concrete_autoencoder",
                        "feature_selection_method": "single_split",
                        "selection_value": cae_k,
                        "fold_count": 1,
                        "k_features": cae_k,
                        "Accuracy": metrics.get("Accuracy", np.nan),
                        "AUC": metrics.get("AUC", np.nan),
                        "train_Accuracy": train_metrics.get("accuracy", np.nan),
                        "train_AUC": train_metrics.get("auc", np.nan),
                        "baseline_Accuracy": baseline_metrics.get("accuracy", np.nan),
                        "baseline_AUC": baseline_metrics.get("auc", np.nan),
                        "baseline_train_Accuracy": baseline_train_metrics.get("accuracy", np.nan),
                        "baseline_train_AUC": baseline_train_metrics.get("auc", np.nan),
                        "checkpoint_path": str(checkpoint_path) if isinstance(checkpoint_path, Path) else np.nan,
                        "checkpoint_status": checkpoint_status,
                    }
                ]
            )
            row_dict = metrics_df.iloc[0].to_dict()
            concrete_rows.append(row_dict)
            completed_keys.add((dataset_name, int(cae_k)))

            history_dict = getattr(history, "history", {}) if history is not None else {}
            train_loss_history = history_dict.get("loss", []) if isinstance(history_dict, dict) else []
            val_loss_history = history_dict.get("val_loss", []) if isinstance(history_dict, dict) else []

            if train_loss_history:
                for epoch_idx, loss_value in enumerate(train_loss_history, start=1):
                    loss_rows.append(
                        {
                            "dataset": dataset_name,
                            "selector": "concrete_autoencoder",
                            "algorithm": "ConcreteAutoencoder",
                            "selection_value": cae_k,
                            "k_features": int(cae_k),
                            "epoch": epoch_idx,
                            "split": "train",
                            "train_loss": float(loss_value),
                        }
                    )
                for epoch_idx, loss_value in enumerate(val_loss_history, start=1):
                    loss_rows.append(
                        {
                            "dataset": dataset_name,
                            "selector": "concrete_autoencoder",
                            "algorithm": "ConcreteAutoencoder",
                            "selection_value": cae_k,
                            "k_features": int(cae_k),
                            "epoch": epoch_idx,
                            "split": "validation",
                            "train_loss": float(loss_value),
                        }
                    )
            else:
                loss_rows.append(
                    {
                        "dataset": dataset_name,
                        "selector": "concrete_autoencoder",
                        "algorithm": "ConcreteAutoencoder",
                        "selection_value": cae_k,
                        "k_features": int(cae_k),
                        "epoch": np.nan,
                        "split": "train",
                        "train_loss": np.nan,
                    }
                )

            results[run_label] = metrics

    if concrete_rows:
        concrete_df = pd.DataFrame(concrete_rows)
        concrete_df["k_features"] = pd.to_numeric(concrete_df["k_features"], errors="coerce")
        concrete_df = concrete_df.dropna(subset=["dataset_name", "k_features"])
        concrete_df["k_features"] = concrete_df["k_features"].astype(int)
        concrete_df = concrete_df.drop_duplicates(subset=["dataset_name", "k_features"], keep="last")
        concrete_df = concrete_df.sort_values(["dataset_name", "k_features"]).reset_index(drop=True)

        concrete_df.to_csv(summary_csv, index=False)

        loss_df = pd.DataFrame(loss_rows)
        if not loss_df.empty:
            loss_df = loss_df.drop_duplicates(
                subset=["dataset", "selector", "algorithm", "selection_value", "k_features", "epoch", "split"],
                keep="last",
            )
        loss_df.to_csv(loss_csv, index=False)

        for dataset_name, dataset_df in concrete_df.groupby("dataset_name", sort=True):
            dataset_csv_path = output_dir / f"{dataset_name}_concrete_metrics.csv"
            dataset_df.to_csv(dataset_csv_path, index=False)

        plot_concrete_results(concrete_df, output_dir)

        print("Saved outputs:")
        print(f"- {summary_csv}")
        print(f"- {loss_csv}")
        print(f"- {output_dir / 'config.yml'}")
        print(f"- {output_dir / 'effective_config.yaml'}")
        print(f"- {output_dir / 'hyperparameters.yaml'}")
        print("- Fold-level CSV omitted: this pipeline runs single split (no cross-validation).")
    else:
        pd.DataFrame(
            columns=[
                "dataset_name",
                "selector",
                "feature_selection_method",
                "selection_value",
                "fold_count",
                "k_features",
                "Accuracy",
                "AUC",
                "train_Accuracy",
                "train_AUC",
                "baseline_Accuracy",
                "baseline_AUC",
                "baseline_train_Accuracy",
                "baseline_train_AUC",
                "checkpoint_path",
                "checkpoint_status",
            ]
        ).to_csv(output_dir / "iterative_feature_curve_summary.csv", index=False)

        pd.DataFrame(
            columns=[
                "dataset",
                "selector",
                "algorithm",
                "selection_value",
                "k_features",
                "epoch",
                "split",
                "train_loss",
            ]
        ).to_csv(output_dir / "iterative_feature_curve_loss_history.csv", index=False)

        print("Fold-level CSV omitted: this pipeline runs single split (no cross-validation).")

    print(f"Pipeline completed successfully. Outputs saved in: {output_dir}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Concrete Autoencoder + TabICL pipeline")
    parser.add_argument("--config", default="config.yml", help="Path to configuration YAML file")
    parser.add_argument(
        "--resume-output-dir",
        default=None,
        help="Existing output directory to resume from using saved checkpoints",
    )
    args = parser.parse_args()

    config_path = str(Path(args.config).expanduser().resolve())
    resume_output_dir = None
    if args.resume_output_dir:
        resume_output_dir = str(Path(args.resume_output_dir).expanduser().resolve())

    results = run_pipeline(config_path, resume_output_dir_override=resume_output_dir)
    print(results)
