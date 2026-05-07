"""CLI entry point for the refactored thesis feature-selection pipeline."""

from __future__ import annotations

from pathlib import Path
import shutil
import yaml

from src.config import ExperimentConfig
from src.pipeline import run_pipeline


def main() -> None:
    """Read configuration from config.yml and execute the full pipeline."""
    base_dir = Path.cwd()
    config_path = base_dir / "config.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    # Initialize configuration with values from the YAML file
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
