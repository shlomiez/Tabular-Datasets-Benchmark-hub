import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import resolve_paths
from src.data_preprocessing import build_dataset_paths, encode_labels, load_dataset_xy
from src.model_training import evaluate_classifier, fit_extra_trees


# Visual settings aligned with plot_results.py publication defaults.
lw = 3
ms = 8
fs_title = 16
fs_label = 14
fs_tick = 12
fs_legend = 11


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

	frames: list[pd.DataFrame] = []
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
	metric_columns = [
		"Accuracy",
		"AUC",
		"train_Accuracy",
		"train_AUC",
		"baseline_Accuracy",
		"baseline_AUC",
		"baseline_train_Accuracy",
		"baseline_train_AUC",
	]
	for column_name in metric_columns:
		combined[column_name] = pd.to_numeric(combined[column_name], errors="coerce")

	combined = combined.dropna(subset=["dataset_name", "k_features"])
	combined["k_features"] = combined["k_features"].astype(int)
	combined = combined.sort_values(["dataset_name", "k_features"]).reset_index(drop=True)
	return combined


def _baseline_candidate_params() -> list[tuple[str, dict[str, object]]]:
	"""Three ExtraTrees baseline variants evaluated on full features (no preprocessing)."""
	return [
		("etree_100x_depth3", {"n_estimators": 100, "max_depth": 3, "max_features": "sqrt"}),
		("etree_200x_depth4", {"n_estimators": 200, "max_depth": 3, "max_features": "sqrt"}),
		("etree_100x_depth5", {"n_estimators": 100, "max_depth": 5, "max_features": "sqrt"}),
	]


def _compute_dataset_baseline(dataset_name: str, seed: int, data_root: Path) -> tuple[dict[str, object], pd.DataFrame]:
	"""Compute best-of-3 full-feature ExtraTrees baseline for one dataset."""
	dataset_paths = build_dataset_paths(data_root)
	if dataset_name not in dataset_paths:
		raise ValueError(f"Unknown dataset name '{dataset_name}' in concrete metrics.")

	dataset_path = dataset_paths[dataset_name]
	if not dataset_path.exists():
		raise FileNotFoundError(f"Dataset file not found for '{dataset_name}': {dataset_path}")

	X, y = load_dataset_xy(dataset_path)
	y_encoded, _ = encode_labels(y)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y_encoded,
		test_size=0.2,
		random_state=seed,
		stratify=y_encoded,
	)

	rows: list[dict[str, object]] = []
	for variant_name, params in _baseline_candidate_params():
		model = fit_extra_trees(X_train, y_train, random_state=seed, **params)
		train_metrics = evaluate_classifier(model, X_train, y_train)
		test_metrics = evaluate_classifier(model, X_test, y_test)
		rows.append(
			{
				"dataset_name": dataset_name,
				"baseline_variant": variant_name,
				"variant_n_estimators": int(params["n_estimators"]),
				"variant_max_depth": str(params["max_depth"]),
				"baseline_train_AUC": float(train_metrics["auc"]),
				"baseline_train_Accuracy": float(train_metrics["accuracy"]),
				"baseline_AUC": float(test_metrics["auc"]),
				"baseline_Accuracy": float(test_metrics["accuracy"]),
				"baseline_feature_count": int(X_train.shape[1]),
			}
		)

	candidates_df = pd.DataFrame(rows)
	best_row = candidates_df.sort_values(
		by=["baseline_AUC", "baseline_Accuracy", "baseline_train_AUC", "baseline_train_Accuracy"],
		ascending=[False, False, False, False],
	).iloc[0]

	selected = {
		"dataset_name": dataset_name,
		"baseline_variant": str(best_row["baseline_variant"]),
		"baseline_feature_count": int(best_row["baseline_feature_count"]),
		"baseline_train_AUC": float(best_row["baseline_train_AUC"]),
		"baseline_train_Accuracy": float(best_row["baseline_train_Accuracy"]),
		"baseline_AUC": float(best_row["baseline_AUC"]),
		"baseline_Accuracy": float(best_row["baseline_Accuracy"]),
	}
	return selected, candidates_df


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
	plot_path = plots_dir / f"{_safe_filename(dataset_name)}_concrete_performance_fixed.png"
	fig.savefig(plot_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def fix_output(folder_path: str, seed: int = 42) -> None:
	"""Backfill concrete outputs with baseline metrics and regenerate corrected plots."""
	folder = Path(folder_path).expanduser().resolve()
	if not folder.exists() or not folder.is_dir():
		raise FileNotFoundError(f"Directory does not exist: {folder}")

	combined_df = _load_concrete_metric_files(folder)
	if combined_df.empty:
		raise RuntimeError(f"No concrete metric CSV files found in {folder}")

	data_root = resolve_paths(base_dir=Path.cwd()).data_root
	selected_rows: list[dict[str, object]] = []
	candidate_tables: list[pd.DataFrame] = []
	for dataset_name in sorted(combined_df["dataset_name"].dropna().astype(str).unique()):
		selected, candidates = _compute_dataset_baseline(dataset_name=dataset_name, seed=seed, data_root=data_root)
		selected_rows.append(selected)
		candidate_tables.append(candidates)
		print(
			"Selected baseline for"
			f" {dataset_name}: {selected['baseline_variant']} "
			f"(test AUC={selected['baseline_AUC']:.6f}, test accuracy={selected['baseline_Accuracy']:.6f})"
		)

	selected_df = pd.DataFrame(selected_rows)
	candidates_df = pd.concat(candidate_tables, ignore_index=True)

	fixed_df = combined_df.merge(selected_df, on="dataset_name", how="left", suffixes=("", "_selected"))

	# Ensure canonical baseline columns use selected values from the best full-feature variant.
	fixed_df["baseline_Accuracy"] = fixed_df["baseline_Accuracy_selected"]
	fixed_df["baseline_AUC"] = fixed_df["baseline_AUC_selected"]
	fixed_df["baseline_train_Accuracy"] = fixed_df["baseline_train_Accuracy_selected"]
	fixed_df["baseline_train_AUC"] = fixed_df["baseline_train_AUC_selected"]

	fixed_df = fixed_df.drop(
		columns=[
			"baseline_Accuracy_selected",
			"baseline_AUC_selected",
			"baseline_train_Accuracy_selected",
			"baseline_train_AUC_selected",
		]
	)

	fixed_df = fixed_df.sort_values(["dataset_name", "k_features"]).reset_index(drop=True)

	summary_fixed_path = folder / "concrete_metrics_summary_fixed.csv"
	candidates_path = folder / "baseline_candidates_summary_fixed.csv"
	selected_path = folder / "baseline_selection_summary_fixed.csv"

	fixed_df.to_csv(summary_fixed_path, index=False)
	candidates_df.sort_values(["dataset_name", "baseline_variant"]).to_csv(candidates_path, index=False)
	selected_df.sort_values("dataset_name").to_csv(selected_path, index=False)

	for dataset_name, part in fixed_df.groupby("dataset_name", sort=True):
		dataset_csv_path = folder / f"{_safe_filename(dataset_name)}_concrete_metrics_fixed.csv"
		part.to_csv(dataset_csv_path, index=False)

	plots_dir = folder / "edited_plots" / "concrete_summary_fixed"
	plots_dir.mkdir(parents=True, exist_ok=True)
	for dataset_name, part in fixed_df.groupby("dataset_name", sort=True):
		_plot_concrete_metric_trends(dataset_name, part, plots_dir)

	print(f"Saved fixed concrete summary CSV: {summary_fixed_path}")
	print(f"Saved baseline candidate table: {candidates_path}")
	print(f"Saved selected baseline table: {selected_path}")
	print(f"Saved fixed per-dataset CSV files in: {folder}")
	print(f"Saved fixed plots in: {plots_dir}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Fix concrete outputs by backfilling baseline metrics and plots.")
	parser.add_argument("folder_path", nargs="?", default=".", help="Path to an output directory with *_concrete_metrics.csv files.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split when recomputing baselines.")
	args = parser.parse_args()
	fix_output(args.folder_path, seed=args.seed)


if __name__ == "__main__":
	main()
