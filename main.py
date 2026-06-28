"""CLI entry point for the refactored thesis feature-selection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import yaml

from src.config import ExperimentConfig
from src.pipeline import run_pipeline


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for configuration and resume behavior."""
    parser = argparse.ArgumentParser(description="Run benchmark pipeline from config.yml")
    parser.add_argument("--config", default="config.yml", help="Path to configuration YAML file")
    parser.add_argument(
        "--resume-output-dir",
        default=None,
        help="Existing output directory to resume from using saved checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    """Read configuration from config.yml and execute the full pipeline."""
    args = _parse_args()
    base_dir = Path.cwd()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (base_dir / config_path).resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Initialize configuration with values from the YAML file
    if args.resume_output_dir:
        config_data["resume_output_dir"] = str(Path(args.resume_output_dir).expanduser().resolve())

    config = ExperimentConfig(**config_data)

    outputs = run_pipeline(base_dir=base_dir, config=config)
    
    # Copy the config file to the generated output directory
    output_dir = outputs.get("output_dir")
    if output_dir:
        shutil.copy2(config_path, output_dir / "config.yml")
        
    print("\nPipeline completed successfully.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
