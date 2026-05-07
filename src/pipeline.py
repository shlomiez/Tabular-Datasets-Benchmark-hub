"""End-to-end pipeline orchestration for STG/LSPIN benchmarking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ExperimentConfig, get_device, resolve_paths
from src.data_preprocessing import build_dataset_paths, dataset_path_report, extract_xy, load_available_datasets
from src.experiment import run_dataset_experiment
from src.hyperparameters import get_hyperparameters, output_hyperparameters_to_yaml, output_hyperparameters_to_yaml
from src.plotting import (
    plot_and_save_lambda_feature_count_curve,
    plot_and_save_lambda_ignored_feature_count_curve,
    plot_and_save_loss_curve,
)
from src.utils import ensure_dir, set_global_seed


def run_pipeline(base_dir: Path | None = None, config: ExperimentConfig | None = None) -> dict[str, Path]:
    """Run the full experiment pipeline and return output artifact paths."""
    config = config or ExperimentConfig()
    paths = resolve_paths(base_dir=base_dir)

    output_dir = ensure_dir(paths.run_output_dir)
    plots_dir = ensure_dir(output_dir / "plots")
    cache_dir = ensure_dir(paths.base_dir / "tabicl_cache")
    model_dir = ensure_dir(output_dir / "tabiclv2_model")

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

    for dataset_name, dataset in selected.items():
        X_data, _ = extract_xy(dataset)
        stg_params, lspin_params, etree_params = get_hyperparameters(dataset_name, len(X_data))

        dataset_lambda_ranges = config.lambda_ranges_by_dataset.get(
            dataset_name,
            {"stg": config.lambda_values, "lspin": config.lambda_values},
        )

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
            evaluation_mode=config.evaluation_mode,
            prediction_model_type=config.prediction_model_type,
            use_peeling=config.use_peeling,
            peeling_tau=config.peeling_tau,
            peeling_low_auc_threshold=config.peeling_low_auc_threshold,
        )

        summary_tables[dataset_name] = summary_df
        fold_tables[dataset_name] = fold_df
        loss_tables[dataset_name] = loss_df

        plot_and_save_loss_curve(
            loss_df=loss_df,
            dataset_name=dataset_name,
            plots_dir=plots_dir,
            title=f"{dataset_name}: Train Loss vs Epoch",
        )

        if config.feature_selection_method == "lamda_tuning":
            plot_and_save_lambda_feature_count_curve(
                summary_df=summary_df,
                dataset_name=dataset_name,
                plots_dir=plots_dir,
            )
            plot_and_save_lambda_ignored_feature_count_curve(
                summary_df=summary_df,
                dataset_name=dataset_name,
                plots_dir=plots_dir,
            )

    combined_summary_df = pd.concat(summary_tables.values(), ignore_index=True)
    combined_fold_df = pd.concat(fold_tables.values(), ignore_index=True)
    combined_loss_df = pd.concat(loss_tables.values(), ignore_index=True)

    summary_csv = output_dir / "iterative_feature_curve_summary.csv"
    fold_csv = output_dir / "iterative_feature_curve_fold_results.csv"
    loss_csv = output_dir / "iterative_feature_curve_loss_history.csv"

    combined_summary_df.to_csv(summary_csv, index=False)
    combined_fold_df.to_csv(fold_csv, index=False)
    combined_loss_df.to_csv(loss_csv, index=False)
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
    print(f"- {output_dir / 'hyperparameters.yaml'}")

    return {
        "output_dir": output_dir,
        "summary_csv": summary_csv,
        "fold_csv": fold_csv,
        "loss_csv": loss_csv,
        "plots_dir": plots_dir,
    }
