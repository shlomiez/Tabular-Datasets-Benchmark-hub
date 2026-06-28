import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Visual settings for publication quality
lw = 3      # line width
ms = 8      # marker size
fs_title = 16
fs_label = 14
fs_tick = 12
fs_legend = 11

def plot_auc_and_accuracy(dataset_name, part, x_stg, x_lspin, x_base, has_train, x_label, x_scale, plots_dir, suffix):
    """Generate and save AUC and Accuracy plots."""
    fig_metrics, axes_metrics = plt.subplots(1, 2, figsize=(16, 6))

    # AUC Plot
    if has_train:
        axes_metrics[0].plot(x_base, part["baseline_train_auc_mean"], marker="o", linestyle="-", linewidth=lw, markersize=ms, color="tab:green", label="Train - Baseline")
        axes_metrics[0].plot(x_stg, part["stg_train_auc_mean"], marker="s", linestyle="-", linewidth=lw, markersize=ms, color="tab:orange", label="Train - STG")
        axes_metrics[0].plot(x_lspin, part["lspin_train_auc_mean"], marker="^", linestyle="-", linewidth=lw, markersize=ms, color="tab:blue", label="Train - LSPIN")
        
    axes_metrics[0].plot(x_base, part["baseline_auc_mean"], marker="o", linestyle="--", linewidth=lw, markersize=ms, color="tab:green", label="Test - Baseline")
    axes_metrics[0].plot(x_stg, part["stg_auc_mean"], marker="s", linestyle="--", linewidth=lw, markersize=ms, color="tab:orange", label="Test - STG")
    axes_metrics[0].plot(x_lspin, part["lspin_auc_mean"], marker="^", linestyle="--", linewidth=lw, markersize=ms, color="tab:blue", label="Test - LSPIN")
    
    axes_metrics[0].set_title(f"{dataset_name}: AUC", fontsize=fs_title, fontweight='bold')
    axes_metrics[0].set_xlabel(x_label, fontsize=fs_label, fontweight='bold')
    axes_metrics[0].set_ylabel("AUC", fontsize=fs_label, fontweight='bold')
    axes_metrics[0].set_xscale(x_scale)
    axes_metrics[0].grid(True, alpha=0.3)
    axes_metrics[0].tick_params(axis='both', labelsize=fs_tick)
    axes_metrics[0].legend(fontsize=fs_legend, loc="best")

    # Accuracy Plot
    if has_train:
        axes_metrics[1].plot(x_base, part["baseline_train_accuracy_mean"], marker="o", linestyle="-", linewidth=lw, markersize=ms, color="tab:green", label="Train - Baseline")
        axes_metrics[1].plot(x_stg, part["stg_train_accuracy_mean"], marker="s", linestyle="-", linewidth=lw, markersize=ms, color="tab:orange", label="Train - STG")
        axes_metrics[1].plot(x_lspin, part["lspin_train_accuracy_mean"], marker="^", linestyle="-", linewidth=lw, markersize=ms, color="tab:blue", label="Train - LSPIN")
        
    axes_metrics[1].plot(x_base, part["baseline_accuracy_mean"], marker="o", linestyle="--", linewidth=lw, markersize=ms, color="tab:green", label="Test - Baseline")
    axes_metrics[1].plot(x_stg, part["stg_accuracy_mean"], marker="s", linestyle="--", linewidth=lw, markersize=ms, color="tab:orange", label="Test - STG")
    axes_metrics[1].plot(x_lspin, part["lspin_accuracy_mean"], marker="^", linestyle="--", linewidth=lw, markersize=ms, color="tab:blue", label="Test - LSPIN")
    
    axes_metrics[1].set_title(f"{dataset_name}: Accuracy", fontsize=fs_title, fontweight='bold')
    axes_metrics[1].set_xlabel(x_label, fontsize=fs_label, fontweight='bold')
    axes_metrics[1].set_ylabel("Accuracy", fontsize=fs_label, fontweight='bold')
    axes_metrics[1].set_xscale(x_scale)
    axes_metrics[1].grid(True, alpha=0.3)
    axes_metrics[1].tick_params(axis='both', labelsize=fs_tick)
    axes_metrics[1].legend(fontsize=fs_legend, loc="best")

    fig_metrics.tight_layout()
    metrics_plot_path = plots_dir / f"{dataset_name}_metrics_curves_{suffix}.png"
    fig_metrics.savefig(metrics_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig_metrics)


def plot_loss_curves(dataset_name, loss_part, plots_dir, suffix):
    """Generate and save loss curves plot."""
    if loss_part.empty:
        return
        
    loss_summary = (
        loss_part.groupby(["algorithm", "epoch"], as_index=False)["train_loss"]
        .mean()
        .sort_values(["algorithm", "epoch"])
    )
    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    for algorithm_name, algo_part in loss_summary.groupby("algorithm"):
        color = "tab:orange" if algorithm_name == "STG" else "tab:blue"
        marker = "s" if algorithm_name == "STG" else "^"
        ax_loss.plot(
            algo_part["epoch"],
            algo_part["train_loss"],
            marker=marker,
            markevery=max(1, len(algo_part)//20),
            linewidth=lw,
            markersize=ms,
            color=color,
            label=algorithm_name,
        )
    ax_loss.set_title(f"{dataset_name}: Train Loss vs Epoch", fontsize=fs_title, fontweight='bold')
    ax_loss.set_xlabel("Epoch", fontsize=fs_label, fontweight='bold')
    ax_loss.set_ylabel("Train loss (MSE)", fontsize=fs_label, fontweight='bold')
    ax_loss.tick_params(axis='both', labelsize=fs_tick)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=fs_legend)
    fig_loss.tight_layout()

    loss_plot_path = plots_dir / f"{dataset_name}_train_loss_curve_{suffix}.png"
    fig_loss.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig_loss)


def plot_lambda_tuning_features(dataset_name, part, plots_dir):
    """Generate and save selected features vs lambda plot."""
    stg_lambdas = part["lambda_value"]
    lspin_lambdas = part.get("lspin_lambda_value", part["lambda_value"])
    
    # Selected Features Plot
    fig_sel, axes_sel = plt.subplots(1, 2, figsize=(16, 6))
    
    # STG
    axes_sel[0].plot(stg_lambdas, part["stg_selected_features_mean"], marker="s", linewidth=lw, markersize=ms, color="tab:orange", label="STG")
    axes_sel[0].set_title(f"{dataset_name}: STG Selected Features", fontsize=fs_title, fontweight='bold')
    axes_sel[0].set_xlabel(r"$\lambda$ (STG)", fontsize=fs_label, fontweight='bold')
    axes_sel[0].set_ylabel("Number of Selected Features", fontsize=fs_label, fontweight='bold')
    axes_sel[0].set_xscale('log')
    axes_sel[0].tick_params(axis='both', labelsize=fs_tick)
    axes_sel[0].grid(True, alpha=0.3)
    axes_sel[0].legend(fontsize=fs_legend)

    # LSPIN
    axes_sel[1].plot(lspin_lambdas, part["lspin_selected_features_mean"], marker="^", linewidth=lw, markersize=ms, color="tab:blue", label="LSPIN")
    axes_sel[1].set_title(f"{dataset_name}: LSPIN Selected Features", fontsize=fs_title, fontweight='bold')
    axes_sel[1].set_xlabel(r"$\lambda$ (LSPIN)", fontsize=fs_label, fontweight='bold')
    axes_sel[1].set_ylabel("Number of Selected Features", fontsize=fs_label, fontweight='bold')
    axes_sel[1].set_xscale('log')
    axes_sel[1].tick_params(axis='both', labelsize=fs_tick)
    axes_sel[1].grid(True, alpha=0.3)
    axes_sel[1].legend(fontsize=fs_legend)

    fig_sel.tight_layout()
    sel_plot_path = plots_dir / f"{dataset_name}_lambda_selected_features_summary.png"
    fig_sel.savefig(sel_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig_sel)


def _safe_filename(value: str) -> str:
    """Convert a label into a filesystem-friendly filename stem."""
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value).strip("_")


def _coerce_optional_metric_column(frame: pd.DataFrame, candidate_columns: tuple[str, ...]) -> pd.Series:
    """Return the first available metric column coerced to numeric, else NaNs."""
    for column_name in candidate_columns:
        if column_name in frame.columns:
            return pd.to_numeric(frame[column_name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _load_concrete_metric_files(folder: Path) -> pd.DataFrame:
    """Combine all per-run concrete metric CSVs into a single dataframe."""
    metric_paths = sorted(folder.glob("*_concrete_metrics.csv"))
    if not metric_paths:
        return pd.DataFrame()

    frames = []
    metric_aliases = {
        "Accuracy": ("Accuracy", "concrete_accuracy"),
        "AUC": ("AUC", "concrete_auc"),
        "train_Accuracy": ("train_Accuracy", "Train_Accuracy", "concrete_train_accuracy"),
        "train_AUC": ("train_AUC", "Train_AUC", "concrete_train_auc"),
        "baseline_Accuracy": ("baseline_Accuracy", "Baseline_Accuracy", "baseline_accuracy"),
        "baseline_AUC": ("baseline_AUC", "Baseline_AUC", "baseline_auc"),
        "baseline_train_Accuracy": (
            "baseline_train_Accuracy",
            "Baseline_train_Accuracy",
            "baseline_train_accuracy",
        ),
        "baseline_train_AUC": (
            "baseline_train_AUC",
            "Baseline_train_AUC",
            "baseline_train_auc",
        ),
    }
    for metric_path in metric_paths:
        frame = pd.read_csv(metric_path)
        required_columns = {"dataset_name", "k_features"}
        if not required_columns.issubset(frame.columns):
            continue

        frame = frame.copy()
        frame.insert(0, "source_file", metric_path.name)

        for canonical_name, candidate_columns in metric_aliases.items():
            frame[canonical_name] = _coerce_optional_metric_column(frame, candidate_columns)

        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["k_features"] = pd.to_numeric(combined["k_features"], errors="coerce")
    for column_name in (
        "Accuracy",
        "AUC",
        "train_Accuracy",
        "train_AUC",
        "baseline_Accuracy",
        "baseline_AUC",
        "baseline_train_Accuracy",
        "baseline_train_AUC",
    ):
        combined[column_name] = pd.to_numeric(combined[column_name], errors="coerce")
    combined = combined.dropna(subset=["dataset_name", "k_features"])
    combined = combined.sort_values(["dataset_name", "k_features"]).reset_index(drop=True)
    combined["k_features"] = combined["k_features"].astype(int)
    return combined


def _plot_concrete_metric_trends(dataset_name: str, part: pd.DataFrame, plots_dir: Path) -> None:
    """Plot AUC and Accuracy versus selected features for one dataset."""
    part = part.sort_values("k_features").copy()
    if part.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x_values = part["k_features"]
    axis_specs = (
        (axes[0], "AUC", "AUC"),
        (axes[1], "Accuracy", "Accuracy"),
    )

    for axis, metric_label, metric_column in axis_specs:
        plotted_any = False
        baseline_train_column = f"baseline_train_{metric_column}"
        baseline_test_column = f"baseline_{metric_column}"
        train_column = f"train_{metric_column}"

        if part[baseline_train_column].notna().any():
            axis.plot(
                x_values,
                part[baseline_train_column],
                marker="o",
                linestyle="-",
                linewidth=lw,
                markersize=ms,
                color="tab:green",
                label="Train - Baseline",
            )
            plotted_any = True

        if part[train_column].notna().any():
            axis.plot(
                x_values,
                part[train_column],
                marker="s",
                linestyle="-",
                linewidth=lw,
                markersize=ms,
                color="tab:orange",
                label="Train - Concrete AE",
            )
            plotted_any = True

        if part[baseline_test_column].notna().any():
            axis.plot(
                x_values,
                part[baseline_test_column],
                marker="o",
                linestyle="--",
                linewidth=lw,
                markersize=ms,
                color="tab:green",
                label="Test - Baseline",
            )
            plotted_any = True

        if part[metric_column].notna().any():
            axis.plot(
                x_values,
                part[metric_column],
                marker="s",
                linestyle="--",
                linewidth=lw,
                markersize=ms,
                color="tab:orange",
                label="Test - Concrete AE",
            )
            plotted_any = True

        axis.set_title(f"{dataset_name}: {metric_label}", fontsize=fs_title, fontweight="bold")
        axis.set_xlabel("Selected features (k)", fontsize=fs_label, fontweight="bold")
        axis.set_ylabel(metric_label, fontsize=fs_label, fontweight="bold")
        axis.set_xticks(x_values.tolist())

        metric_values = part[
            [metric_column, train_column, baseline_test_column, baseline_train_column]
        ].to_numpy(dtype=float)
        finite_values = metric_values[np.isfinite(metric_values)]
        y_max = 1.0 if finite_values.size == 0 else max(1.0, float(finite_values.max()) * 1.1)
        axis.set_ylim(0, y_max)

        axis.tick_params(axis="both", labelsize=fs_tick)
        axis.grid(True, alpha=0.3)
        if plotted_any:
            axis.legend(fontsize=fs_legend, loc="best")

    fig.tight_layout()
    plot_path = plots_dir / f"{_safe_filename(dataset_name)}_concrete_performance.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_concrete_results(input_dir: str):
    """Build one combined CSV and one performance plot per dataset for Concrete Autoencoder runs."""
    folder = Path(input_dir)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    combined_df = _load_concrete_metric_files(folder)
    if combined_df.empty:
        print(f"No Concrete Autoencoder metric CSVs found in {folder}")
        return

    summary_csv_path = folder / "concrete_metrics_summary.csv"
    combined_df.to_csv(summary_csv_path, index=False)

    has_train_metrics = combined_df[["train_Accuracy", "train_AUC"]].notna().any().any()
    has_baseline_metrics = combined_df[
        [
            "baseline_Accuracy",
            "baseline_AUC",
            "baseline_train_Accuracy",
            "baseline_train_AUC",
        ]
    ].notna().any().any()

    if not has_train_metrics:
        print("Train metrics unavailable in concrete CSVs; plotting test metrics only.")
    if not has_baseline_metrics:
        print("Baseline etree/no-FS metrics unavailable in concrete CSVs; plotting without baseline.")

    plots_dir = folder / "edited_plots" / "concrete_summary"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, part in combined_df.groupby("dataset_name", sort=True):
        _plot_concrete_metric_trends(dataset_name, part, plots_dir)

    print(f"Concrete summary CSV saved to '{summary_csv_path}'")
    print(f"Concrete dataset plots saved to '{plots_dir}'")

def plot_results(input_dir: str):
    folder = Path(input_dir)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    concrete_metric_paths = sorted(folder.glob("*_concrete_metrics.csv"))
    standard_summary_path = folder / "iterative_feature_curve_summary.csv"
    if concrete_metric_paths and not standard_summary_path.exists():
        plot_concrete_results(input_dir)
        return

    # Check for the expected CSV files
    summary_path = standard_summary_path
    loss_path = folder / "iterative_feature_curve_loss_history.csv"

    if not summary_path.exists():
        # Fallback or pattern match if names differ slightly
        summaries = list(folder.glob("*summary.csv"))
        if summaries:
            summary_path = summaries[0]
        else:
            print(f"Warning: Could not find summary CSV in {folder}")
            summary_df = pd.DataFrame()
    
    if summary_path.exists():
        combined_summary_df = pd.read_csv(summary_path)
    else:
        combined_summary_df = pd.DataFrame()

    if not loss_path.exists():
        losses = list(folder.glob("*loss_history.csv"))
        if losses:
            loss_path = losses[0]
        else:
            loss_path = None
            
    if loss_path and loss_path.exists():
        combined_loss_df = pd.read_csv(loss_path)
    else:
        combined_loss_df = pd.DataFrame()

    if combined_summary_df.empty:
        print("No summary data found to plot. Exiting.")
        return

    plots_dir = folder / "plots_publication"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine method and axis details
    current_method = combined_summary_df["feature_selection_method"].dropna().iloc[0] if "feature_selection_method" in combined_summary_df.columns else "features_ratio"
    
    if current_method == "features_ratio":
        sort_col = "feature_ratio"
        x_label = "Retained features ratio (k/p)"
        x_scale = "linear"
        suffix = "feature_ratio"
    else:
        sort_col = "lambda_value"
        x_label = "Number of Selected Features (Gate > 0)"
        x_scale = "linear"
        suffix = "num_features"

    datasets = sorted(combined_summary_df["dataset"].unique())
    
    for dataset_name in datasets:
        print(f"Generating plots for '{dataset_name}'...")
        part = combined_summary_df[combined_summary_df["dataset"] == dataset_name].copy()
        
        if "feature_selection_method" in part.columns:
            part = part[part["feature_selection_method"] == current_method]
        
        if sort_col in part.columns:
            part = part.sort_values(sort_col)
            
        if part.empty:
            continue

        if current_method == "features_ratio":
            x_stg = part["feature_ratio"]
            x_lspin = part["feature_ratio"]
            x_base = part["feature_ratio"]
        else:
            x_stg = part["stg_selected_features_mean"]
            x_lspin = part["lspin_selected_features_mean"]
            x_base = part["stg_selected_features_mean"] 

        train_cols = {
            "baseline_train_auc_mean", "stg_train_auc_mean", "lspin_train_auc_mean",
            "baseline_train_accuracy_mean", "stg_train_accuracy_mean", "lspin_train_accuracy_mean",
        }
        has_train = train_cols.issubset(part.columns)

        # ---- 1. AUC and Accuracy Plot ----
        plot_auc_and_accuracy(
            dataset_name, part, x_stg, x_lspin, x_base, has_train, 
            x_label, x_scale, plots_dir, suffix
        )

        # ---- 2. Loss Plot ----
        if not combined_loss_df.empty:
            loss_part = combined_loss_df[combined_loss_df["dataset"] == dataset_name].copy()
            if "feature_selection_method" in loss_part.columns:
                loss_part = loss_part[loss_part["feature_selection_method"] == current_method]
            
            if not loss_part.empty:
                plot_loss_curves(dataset_name, loss_part, plots_dir, suffix)

        # ---- 3. Lambda Tuning Features Plot ----
        if current_method in ["lamda_tuning", "lambda_tuning"] and {
            "stg_selected_features_mean", "lspin_selected_features_mean", "lambda_value"
        }.issubset(part.columns):
            plot_lambda_tuning_features(dataset_name, part, plots_dir)

    print(f"All plots saved securely inside '{plots_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from evaluation output CSVs.")
    parser.add_argument("folder_path", nargs="?", default=".", help="Path to the output directory.")
    parser.add_argument(
        "--concrete",
        action="store_true",
        help="Force Concrete Autoencoder summary CSV and per-dataset plots.",
    )

    args = parser.parse_args()
    if args.concrete:
        plot_concrete_results(args.folder_path)
    else:
        plot_results(args.folder_path)