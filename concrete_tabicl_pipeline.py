import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from tabicl import TabICLClassifier
from keras.layers import Dense

from src.utils import ensure_dir
from src.config import resolve_paths
from src.data_preprocessing import build_dataset_paths, load_dataset_xy
from src.plotting import plot_concrete_metrics, plot_auc_and_accuracy, plot_loss_curves

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_pipeline(config_path: str) -> Dict[str, Dict[str, float]]:
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
    output_dir = ensure_dir(paths.run_output_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    
    # Save a copy of the configuration into the run's output folder
    results: dict[str, Dict[str, float]] = {}

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

            def decoder(x):
                return Dense(X_train_np.shape[1])(x)

            selector = ConcreteAutoencoderFeatureSelector(
                K=cae_k,
                output_function=decoder,
                num_epochs=100,
                # Set tryout_limit to 1 to prevent epoch doubling
                tryout_limit=1,
            )

            # Fit strictly on training data (Autoencoder reconstructs X_train)
            selector.fit(X_train_np, X_train_np, val_X=X_test_np, val_Y=X_test_np)
            history = getattr(getattr(selector, "model", None), "history", None)

            # Transform both train and test sets structure
            # Note: ConcreteAutoencoder's transform method may fail on Pandas Dataframes or has a bug (using 1D index on 2D array)
            # so we extract indices manually to be safe.
            indices = selector.get_indices()
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

            # Generate predictions
            predictions = model.predict(X_test_reduced)

            # 6. Evaluation and output saving
            metrics: Dict[str, float] = {}
            if task_type == 'classification':
                metrics['Accuracy'] = accuracy_score(y_test, predictions)
                # Assuming predictions contains labels; if it has probabilities, compute AUC
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_test_reduced)
                    if probas.shape[1] == 2:
                        metrics['AUC'] = roc_auc_score(y_test, probas[:, 1])
                    else:
                        metrics['AUC'] = roc_auc_score(y_test, probas, multi_class='ovr')

            print(f"Pipeline evaluation completed. Metrics: {metrics}")

            # Reuse existing plotting styles (AUC/Accuracy + loss curves)
            auc_value = metrics.get("AUC", np.nan)
            accuracy_value = metrics.get("Accuracy", np.nan)
            summary_df = pd.DataFrame([
                {
                    "baseline_auc_mean": auc_value,
                    "stg_auc_mean": auc_value,
                    "lspin_auc_mean": auc_value,
                    "baseline_accuracy_mean": accuracy_value,
                    "stg_accuracy_mean": accuracy_value,
                    "lspin_accuracy_mean": accuracy_value,
                }
            ])
            x_values = np.array([cae_k])
            plot_auc_and_accuracy(
                run_label,
                summary_df,
                x_values,
                x_values,
                x_values,
                False,
                "Selected features (k)",
                "linear",
                plots_dir,
                "concrete",
            )

            loss_df = pd.DataFrame()
            if history is not None and "loss" in history.history:
                loss_df = pd.DataFrame(
                    {
                        "algorithm": ["Concrete"] * len(history.history["loss"]),
                        "epoch": list(range(1, len(history.history["loss"]) + 1)),
                        "train_loss": history.history["loss"],
                    }
                )
            plot_loss_curves(run_label, loss_df, plots_dir, "concrete")

            # Save the output plot
            plot_concrete_metrics(run_label, metrics, plots_dir)

            # Save metrics cleanly into a CSV file inside the output directory
            metrics_df = pd.DataFrame([metrics])
            metrics_df.insert(0, "dataset_name", dataset_name)
            metrics_df.insert(1, "k_features", cae_k)
            metrics_csv_path = output_dir / f"{run_label}_concrete_metrics.csv"
            metrics_df.to_csv(metrics_csv_path, index=False)

            results[run_label] = metrics

    print(f"Pipeline completed successfully. Outputs saved in: {output_dir}")

    return results
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    print(f"Pipeline completed successfully. Outputs saved in: {output_dir}")
    
    return predictions, metrics

if __name__ == "__main__":
    results = run_pipeline("config.yml")
    print(results)
