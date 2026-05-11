"""Plotting helpers for loss and feature-selection diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def plot_metrics(dataset_name: str, summary_df: pd.DataFrame, loss_df: pd.DataFrame, config, plots_dir: Path) -> None:
    """Helper to structure data and call plotting functions based on configuration."""
    if summary_df.empty:
        return

    if config.feature_selection_method == "features_ratio":
        sort_col = "feature_ratio"
        x_label = "Retained features ratio (k/p)"
        x_scale = "linear"
        suffix = "feature_ratio"
    else:
        sort_col = "lambda_value"
        x_label = "Number of Selected Features (Gate > 0)"
        x_scale = "linear"
        suffix = "num_features"

    if sort_col in summary_df.columns:
        plot_df = summary_df.sort_values(sort_col)
    else:
        plot_df = summary_df.copy()

    if config.feature_selection_method == "features_ratio":
        x_stg = plot_df["feature_ratio"]
        x_lspin = plot_df["feature_ratio"]
        x_base = plot_df["feature_ratio"]
    else:
        x_stg = plot_df["stg_selected_features_mean"]
        x_lspin = plot_df["lspin_selected_features_mean"]
        x_base = plot_df["stg_selected_features_mean"]

    train_cols = {
        "baseline_train_auc_mean", "stg_train_auc_mean", "lspin_train_auc_mean",
        "baseline_train_accuracy_mean", "stg_train_accuracy_mean", "lspin_train_accuracy_mean",
    }
    has_train = train_cols.issubset(plot_df.columns)

    # 1. AUC and Accuracy
    plot_auc_and_accuracy(
        dataset_name, plot_df, x_stg, x_lspin, x_base, has_train, 
        x_label, x_scale, plots_dir, suffix
    )

    # 2. Loss Curves
    if not loss_df.empty:
        plot_loss_curves(dataset_name, loss_df, plots_dir, suffix)

    # 3. Lambda Tuning Features
    if config.feature_selection_method in ["lamda_tuning", "lambda_tuning"]:
        plot_lambda_tuning_features(dataset_name, plot_df, plots_dir)


