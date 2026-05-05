"""Plotting helpers for loss and feature-selection diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_and_save_loss_curve(
    loss_df: pd.DataFrame,
    dataset_name: str,
    plots_dir: Path,
    title: str | None = None,
) -> None:
    """Save average STG/LSPIN training-loss curve for one dataset."""
    if loss_df.empty:
        print(f"Loss history is missing for {dataset_name}; skipped loss plot.")
        return

    loss_summary = (
        loss_df.groupby(["algorithm", "epoch"], as_index=False)["train_loss"]
        .mean()
        .sort_values(["algorithm", "epoch"])
    )

    fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
    color_by_algorithm = {"STG": "tab:orange", "LSPIN": "tab:blue"}
    for algorithm_name, algo_part in loss_summary.groupby("algorithm"):
        ax_loss.plot(
            algo_part["epoch"],
            algo_part["train_loss"],
            marker="o",
            linewidth=2,
            color=color_by_algorithm.get(algorithm_name),
            label=algorithm_name,
        )

    ax_loss.set_title(title or f"{dataset_name}: Train Loss vs Epoch")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()
    fig_loss.tight_layout()

    loss_plot_path = plots_dir / f"{dataset_name}_train_loss_curve.png"
    fig_loss.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss plot for {dataset_name}: {loss_plot_path}")
    plt.close(fig_loss)


def plot_and_save_lambda_feature_count_curve(
    summary_df: pd.DataFrame,
    dataset_name: str,
    plots_dir: Path,
) -> None:
    """Save selected-features vs lambda chart with dual STG/LSPIN lambda labels."""
    if summary_df.empty or "lambda_value" not in summary_df.columns:
        print(f"Lambda summary is missing for {dataset_name}; skipped lambda feature-count plot.")
        return

    part = summary_df[summary_df["dataset"] == dataset_name].copy()
    part = part[part["feature_selection_method"] == "lamda_tuning"].sort_values("lambda_value")
    if part.empty:
        print(f"No lambda-tuning rows for {dataset_name}; skipped lambda feature-count plot.")
        return

    x_positions = np.arange(len(part))
    stg_labels = [f"{v:.3g}" for v in part["lambda_value"].to_numpy()]
    lspin_labels = [f"{v:.3g}" for v in part.get("lspin_lambda_value", part["lambda_value"]).to_numpy()]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_positions, part["lspin_selected_features_mean"], marker="o", linewidth=2, color="tab:blue", label="LSPIN")
    ax.plot(x_positions, part["stg_selected_features_mean"], marker="o", linewidth=2, color="tab:orange", label="STG")
    ax.set_xlabel("STG lambda")
    ax.set_ylabel("Number of Selected Features (Gate > 0)")
    ax.set_title(f"{dataset_name}: Selected Features vs Lambda")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stg_labels, rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(x_positions)
    ax_top.set_xticklabels(lspin_labels, rotation=30, ha="left")
    ax_top.set_xlabel("LSPIN lambda")

    fig.tight_layout()

    out_path = plots_dir / f"{dataset_name}_lambda_selected_features.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved lambda feature-count plot for {dataset_name}: {out_path}")
    plt.close(fig)


def plot_and_save_lambda_ignored_feature_count_curve(
    summary_df: pd.DataFrame,
    dataset_name: str,
    plots_dir: Path,
) -> None:
    """Save ignored-features vs lambda chart with dual STG/LSPIN lambda labels."""
    if summary_df.empty or "lambda_value" not in summary_df.columns:
        print(f"Lambda summary is missing for {dataset_name}; skipped lambda ignored-feature plot.")
        return

    part = summary_df[summary_df["dataset"] == dataset_name].copy()
    part = part[part["feature_selection_method"] == "lamda_tuning"].sort_values("lambda_value")
    if part.empty:
        print(f"No lambda-tuning rows for {dataset_name}; skipped lambda ignored-feature plot.")
        return

    p_values = part["p"].to_numpy(dtype=float)
    stg_selected = part["stg_selected_features_mean"].to_numpy(dtype=float)
    lspin_selected = part["lspin_selected_features_mean"].to_numpy(dtype=float)
    stg_ignored = p_values - stg_selected
    lspin_ignored = p_values - lspin_selected

    x_positions = np.arange(len(part))
    stg_labels = [f"{v:.3g}" for v in part["lambda_value"].to_numpy()]
    lspin_labels = [f"{v:.3g}" for v in part.get("lspin_lambda_value", part["lambda_value"]).to_numpy()]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_positions, lspin_ignored, marker="o", linewidth=2, color="tab:blue", label="LSPIN")
    ax.plot(x_positions, stg_ignored, marker="o", linewidth=2, color="tab:orange", label="STG")
    ax.set_xlabel("STG lambda")
    ax.set_ylabel("Number of Ignored Features")
    ax.set_title(f"{dataset_name}: Ignored Features vs Lambda")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stg_labels, rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(x_positions)
    ax_top.set_xticklabels(lspin_labels, rotation=30, ha="left")
    ax_top.set_xlabel("LSPIN lambda")

    fig.tight_layout()

    out_path = plots_dir / f"{dataset_name}_lambda_ignored_features.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved lambda ignored-feature plot for {dataset_name}: {out_path}")
    plt.close(fig)
