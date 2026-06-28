"""End-to-end pipeline orchestration for STG/LSPIN benchmarking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ExperimentConfig, get_device, resolve_paths
from src.data_preprocessing import build_dataset_paths, dataset_path_report, extract_xy, load_available_datasets
from src.experiment import run_dataset_experiment, run_single_split_selector_experiment
from src.hyperparameters import get_hyperparameters, output_hyperparameters_to_yaml
from src.plotting import plot_concrete_results, plot_metrics
from src.utils import ensure_dir, set_global_seed



def run_pipeline(base_dir: Path | None = None, config: ExperimentConfig | None = None) -> dict[str, Path]:
    """Run the full experiment pipeline and return output artifact paths."""
    config = config or ExperimentConfig()
    paths = resolve_paths(base_dir=base_dir)

    output_dir = ensure_dir(paths.run_output_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    if config.prediction_model_type == "tabiclv2":
        cache_dir = ensure_dir(paths.base_dir / "tabicl_cache")
        model_dir = ensure_dir(output_dir / "tabiclv2_model")
    else:        
        cache_dir = None
        model_dir = None

    set_global_seed(config.seed)
    device = get_device()

    print(f"Base directory: {paths.base_dir}")
    print(f"Data root: {paths.data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Torch device: {device}")

    dataset_paths = build_dataset_paths(paths.data_root)
    path_status = dataset_path_report(dataset_paths)
    print(path_status.to_string(index=False))

    loaded_datasets, missing_datasets = load_available_datasets(dataset_paths)
    if missing_datasets:
        print("Missing datasets:", ", ".join(missing_datasets))

    selected = {
        name: loaded_datasets[name]
        for name in config.dataset_names
        if name in loaded_datasets
    }

    if not selected:
        raise RuntimeError("No requested datasets were loaded. Check THESIS_DATA_ROOT and dataset_names.")

    summary_tables: dict[str, pd.DataFrame] = {}
    fold_tables: dict[str, pd.DataFrame] = {}
    loss_tables: dict[str, pd.DataFrame] = {}
    selector_tables: dict[str, pd.DataFrame] = {}
    concrete_tables: dict[str, pd.DataFrame] = {}

    selector_names = [selector.lower() for selector in (config.feature_selectors or [])]
    if not selector_names:
        selector_names = ["stg", "lspin"]

    supported_selectors = {
        "stg",
        "lspin",
        "concrete",
        "concrete_autoencoder",
        "baseline",
        "prefilter_baseline",
        "variance_selectkbest_baseline",
    }
    unknown_selectors = sorted(set(selector_names) - supported_selectors)
    if unknown_selectors:
        raise ValueError(f"Unsupported feature selectors in config.feature_selectors: {unknown_selectors}")

    if config.prediction_model_type != "tabiclv2" and any(
        selector in {"concrete", "concrete_autoencoder"} for selector in selector_names
    ):
        raise ValueError("Concrete Autoencoder can only be used when prediction_model_type='tabiclv2'.")

    selector_set = set(selector_names)
    cv_selector_set = {selector for selector in selector_set if selector in {"stg", "lspin"}}
    single_split_selector_set = selector_set - cv_selector_set

    for dataset_name, dataset in selected.items():
        X_data, _ = extract_xy(dataset)
        stg_params, lspin_params, etree_params = get_hyperparameters(dataset_name, len(X_data))

        dataset_lambda_ranges = config.lambda_ranges_by_dataset.get(
            dataset_name,
            {"stg": config.lambda_values, "lspin": config.lambda_values},
        )

        if cv_selector_set:
            summary_df, fold_df, loss_df = run_dataset_experiment(
                dataset_name=dataset_name,
                dataset=dataset,
                output_dir=output_dir,
                device=device,
                cache_dir=cache_dir,
                model_dir=model_dir,
                random_state=config.seed,
                n_splits=config.n_splits,
                stg_params=stg_params,
                lspin_params=lspin_params,
                etree_params=etree_params,
                feature_selection_method=config.feature_selection_method,
                feature_ratios=config.feature_ratios,
                lambda_values=config.lambda_values,
                stg_lambda_values=dataset_lambda_ranges["stg"],
                lspin_lambda_values=dataset_lambda_ranges["lspin"],
                run_stg="stg" in cv_selector_set,
                run_lspin="lspin" in cv_selector_set,
                evaluation_mode=config.evaluation_mode,
                prediction_model_type=config.prediction_model_type,
                statistical_prefilter_k=config.concrete_prefilter_k,
                use_peeling=config.use_peeling,
                peeling_tau=config.peeling_tau,
                peeling_low_auc_threshold=config.peeling_low_auc_threshold,
            )

            summary_tables[dataset_name] = summary_df
            fold_tables[dataset_name] = fold_df
            loss_tables[dataset_name] = loss_df

            plot_metrics(dataset_name, summary_df, loss_df, config, plots_dir)

        for selector_name in single_split_selector_set:
            selector_df = run_single_split_selector_experiment(
                output_dir=output_dir,
                dataset_name=dataset_name,
                dataset=dataset,
                device=device,
                cache_dir=cache_dir,
                model_dir=model_dir,
                random_state=config.seed,
                prediction_model_type=config.prediction_model_type,
                selector_name=selector_name,
                concrete_k_values=config.concrete_k_values,
                concrete_epochs=config.concrete_epochs,
                concrete_prefilter_k=config.concrete_prefilter_k,
                baseline_postprefilter_k_cap=config.baseline_postprefilter_k_cap,
            )
            selector_tables[f"{dataset_name}:{selector_name}"] = selector_df
            if selector_name in {"concrete", "concrete_autoencoder"}:
                concrete_tables[dataset_name] = _write_concrete_metric_csv(output_dir, dataset_name, selector_df)

    combined_summary_df = pd.concat(summary_tables.values(), ignore_index=True) if summary_tables else pd.DataFrame()
    combined_fold_df = pd.concat(fold_tables.values(), ignore_index=True) if fold_tables else pd.DataFrame()
    combined_loss_df = pd.concat(loss_tables.values(), ignore_index=True) if loss_tables else pd.DataFrame()
    combined_selector_df = pd.concat(selector_tables.values(), ignore_index=True) if selector_tables else pd.DataFrame()
    combined_concrete_df = pd.concat(concrete_tables.values(), ignore_index=True) if concrete_tables else pd.DataFrame()

    summary_csv = output_dir / "iterative_feature_curve_summary.csv"
    fold_csv = output_dir / "iterative_feature_curve_fold_results.csv"
    loss_csv = output_dir / "iterative_feature_curve_loss_history.csv"
    selector_csv = output_dir / "feature_selector_results.csv"

    combined_summary_df.to_csv(summary_csv, index=False)
    combined_fold_df.to_csv(fold_csv, index=False)
    combined_loss_df.to_csv(loss_csv, index=False)
    if not combined_selector_df.empty:
        combined_selector_df.to_csv(selector_csv, index=False)
    output_hyperparameters_to_yaml(
        path=output_dir / "hyperparameters.yaml",
        stg_params=stg_params,
        lspin_params=lspin_params,
        etree_params=etree_params
    )

    print("Saved outputs:")
    print(f"- {summary_csv}")
    print(f"- {fold_csv}")
    print(f"- {loss_csv}")
    if not combined_selector_df.empty:
        print(f"- {selector_csv}")
    print(f"- {output_dir / 'hyperparameters.yaml'}")

    if not combined_concrete_df.empty:
        plot_concrete_results(combined_concrete_df, output_dir)

    return {
        "output_dir": output_dir,
        "summary_csv": summary_csv,
        "fold_csv": fold_csv,
        "loss_csv": loss_csv,
        "selector_csv": selector_csv,
        "plots_dir": plots_dir,
    }


def _write_concrete_metric_csv(output_dir: Path, dataset_name: str, selector_df: pd.DataFrame) -> pd.DataFrame:
    """Write one Concrete Autoencoder metric CSV for a dataset and return the normalized rows."""
    if selector_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "k_features",
                "Accuracy",
                "AUC",
                "train_Accuracy",
                "train_AUC",
                "baseline_Accuracy",
                "baseline_AUC",
                "baseline_train_Accuracy",
                "baseline_train_AUC",
            ]
        )

    concrete_rows = selector_df[selector_df["selector"].str.lower() == "concrete_autoencoder"].copy()
    if concrete_rows.empty:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "k_features",
                "Accuracy",
                "AUC",
                "train_Accuracy",
                "train_AUC",
                "baseline_Accuracy",
                "baseline_AUC",
                "baseline_train_Accuracy",
                "baseline_train_AUC",
            ]
        )

    metrics_df = pd.DataFrame(
        {
            "dataset_name": dataset_name,
            "k_features": pd.to_numeric(concrete_rows["concrete_k"], errors="coerce"),
            "Accuracy": pd.to_numeric(concrete_rows["concrete_accuracy"], errors="coerce"),
            "AUC": pd.to_numeric(concrete_rows["concrete_auc"], errors="coerce"),
            "train_Accuracy": pd.to_numeric(concrete_rows.get("concrete_train_accuracy"), errors="coerce"),
            "train_AUC": pd.to_numeric(concrete_rows.get("concrete_train_auc"), errors="coerce"),
            "baseline_Accuracy": pd.to_numeric(concrete_rows.get("baseline_accuracy"), errors="coerce"),
            "baseline_AUC": pd.to_numeric(concrete_rows.get("baseline_auc"), errors="coerce"),
            "baseline_train_Accuracy": pd.to_numeric(concrete_rows.get("baseline_train_accuracy"), errors="coerce"),
            "baseline_train_AUC": pd.to_numeric(concrete_rows.get("baseline_train_auc"), errors="coerce"),
        }
    ).dropna(subset=["k_features"])

    metrics_df["k_features"] = metrics_df["k_features"].astype(int)
    metrics_df = metrics_df.sort_values("k_features").reset_index(drop=True)
    metrics_csv_path = output_dir / f"{dataset_name}_concrete_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    return metrics_df
