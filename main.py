"""CLI entry point for the refactored thesis feature-selection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ExperimentConfig
from src.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for pipeline configuration overrides."""
    parser = argparse.ArgumentParser(description="Run STG/LSPIN benchmarking pipeline")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Project base directory")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--feature-selection-method",
        choices=["features_ratio", "lamda_tuning"],
        default="lamda_tuning",
        help="Selection mode",
    )
    parser.add_argument(
        "--prediction-model-type",
        choices=["etree", "tabiclv2"],
        default="tabiclv2",
        help="Downstream model type",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["full", "selector_only"],
        default="full",
        help="Run full predictor evaluation or selector-only diagnostics",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["RELATHE"],
        help="Datasets to run, e.g. RELATHE madelon",
    )
    return parser


def main() -> None:
    """Parse arguments and execute the full pipeline."""
    args = build_parser().parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        n_splits=args.n_splits,
        feature_selection_method=args.feature_selection_method,
        evaluation_mode=args.evaluation_mode,
        prediction_model_type=args.prediction_model_type,
        dataset_names=args.datasets,
    )

    outputs = run_pipeline(base_dir=args.base_dir, config=config)
    print("\nPipeline completed successfully.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
