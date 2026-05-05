"""Configuration models and environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.utils import ensure_dir


@dataclass
class PathConfig:
    """Filesystem configuration for project paths."""

    base_dir: Path
    data_root: Path
    output_root: Path

    @property
    def run_output_dir(self) -> Path:
        """Return output directory for the current execution."""
        return self.output_root


@dataclass
class ExperimentConfig:
    """Top-level experiment controls."""

    seed: int = 42
    n_splits: int = 5
    feature_selection_method: str = "lamda_tuning"
    evaluation_mode: str = "full"
    prediction_model_type: str = "tabiclv2"
    dataset_names: list[str] = field(default_factory=lambda: ["RELATHE"])
    feature_ratios: list[float] = field(
        default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    )
    lambda_values: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    lambda_ranges_by_dataset: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: {
            "madelon": {
                "stg": np.geomspace(0.9, 3, num=7).tolist(),
                "lspin": np.geomspace(0.01, 10, num=7).tolist(),
            },
            "RELATHE": {
                "stg": np.geomspace(10, 17, num=7).tolist(),
                "lspin": np.geomspace(0.03, 200, num=7).tolist(),
            },
        }
    )


def resolve_paths(base_dir: Path | None = None) -> PathConfig:
    """Resolve data/output paths using explicit args first, then environment variables."""
    env_base_dir = os.environ.get("THESIS_BASE_DIR")
    env_data_root = os.environ.get("THESIS_DATA_ROOT")
    env_output_dir = os.environ.get("THESIS_OUTPUT_DIR")

    project_base_dir = (
        Path(env_base_dir).expanduser().resolve()
        if env_base_dir
        else (base_dir.resolve() if base_dir is not None else Path.cwd().resolve())
    )

    data_root = (
        Path(env_data_root).expanduser().resolve()
        if env_data_root
        else (project_base_dir / "data").resolve()
    )

    output_root = (
        Path(env_output_dir).expanduser().resolve()
        if env_output_dir
        else (project_base_dir / "output").resolve()
    )

    ensure_dir(output_root)
    return PathConfig(base_dir=project_base_dir, data_root=data_root, output_root=output_root)


def get_device() -> torch.device:
    """Get torch device according to hardware availability."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
